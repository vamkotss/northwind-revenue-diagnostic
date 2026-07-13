"""Export a star schema for Power BI.

WHY NOT JUST POINT POWER BI AT data/processed?
----------------------------------------------
Because it would work, badly, and you would spend two days not understanding why.

Power BI's engine is built for a STAR SCHEMA: narrow FACT tables (the numbers)
surrounded by DIMENSION tables (the things you slice by). Point it at wide,
denormalised analysis tables instead and you get:

  - measures that are slow, because every filter scans a fat table
  - totals that are subtly wrong, because the same customer attribute lives in
    three tables and they disagree
  - a date filter that does not work properly, because Power BI needs a real
    calendar table and cannot invent one

THE MODEL WE BUILD
------------------
    dim_date         one row per DAY. Marked as the date table in Power BI.
    dim_customer     one row per customer. Everything you slice by.
    dim_plan         one row per plan tier, with its list price.

    fact_mrr         one row per customer per MONTH. The spine.
    fact_movement    one row per month. Churn / contraction / expansion.
    fact_forecast    one row per future week, with its interval.
    fact_reconciliation  the ledger bridge, so the dashboard can PROVE it ties.

Every fact joins to the dimensions on a key. Nothing is duplicated. Power BI's
engine is happy, the measures are fast, and the totals are right.

THE RECONCILIATION TABLE IS NOT DECORATION
------------------------------------------
Most dashboards ask you to trust them. This one ships the bridge that proves the
numbers tie to the billing ledger, on a page, where the CFO can look at it.

That page is the difference between a dashboard people check and a dashboard
people believe.

Run:  python -m northwind.export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from northwind.decompose import build_mrr_panel, decompose_all_months
from northwind.forecast import backtest, build_weekly_mrr, empirical_intervals, score_recent
from northwind.forecast import forecast_forward as build_forecast
from northwind.generate import PLANS
from northwind.metrics import load_contract
from northwind.reconcile import reconcile_all_months, reconcile_ledger

AS_OF = pd.Timestamp("2026-06-30")


# ---------------------------------------------------------------------------
# DIMENSIONS
# ---------------------------------------------------------------------------


def build_dim_date(start: str = "2024-01-01", end: str = "2026-12-31") -> pd.DataFrame:
    """A proper calendar table. One row per day. Power BI requires this.

    WHY YOU CANNOT SKIP THIS. Power BI's time intelligence functions - "same
    period last year", "year to date", "rolling 12 months" - do not work on a
    date column inside a fact table. They need a dedicated, GAPLESS calendar
    marked as the date table.

    Skip it and your year-over-year comparison silently returns blank for every
    month where nothing happened to be sold. No error. Just blanks, and a chart
    with holes in it that you will blame on the data.
    """
    days = pd.date_range(start, end, freq="D")

    dim = pd.DataFrame({"date": days})

    dim["date_key"] = dim["date"].dt.strftime("%Y%m%d").astype(int)
    dim["year"] = dim["date"].dt.year
    dim["quarter"] = "Q" + dim["date"].dt.quarter.astype(str)
    dim["month_num"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.strftime("%b")
    dim["month_start"] = dim["date"].dt.to_period("M").dt.to_timestamp()

    # A sortable year-month label. Power BI sorts text alphabetically unless you
    # give it something to sort BY - "Apr" comes before "Jan" otherwise.
    dim["year_month"] = dim["date"].dt.strftime("%Y-%m")
    dim["year_month_sort"] = dim["date"].dt.strftime("%Y%m").astype(int)

    dim["week_start"] = dim["date"] - pd.to_timedelta(dim["date"].dt.dayofweek, unit="D")
    dim["is_month_start"] = dim["date"].dt.is_month_start

    return dim


def build_dim_customer(customers: pd.DataFrame) -> pd.DataFrame:
    """One row per customer. Everything the dashboard slices by lives here.

    Already cleaned: segments are canonical, nulls are an explicit 'Unknown'
    category. If we shipped the raw version, every slicer would show 'SMB',
    'smb', and ' SMB ' as three separate options - and the user would quietly
    conclude the dashboard is broken. They would be right.
    """
    dim = customers.copy()

    dim["signup_date"] = pd.to_datetime(dim["signup_date"])
    dim["signup_month"] = dim["signup_date"].dt.to_period("M").dt.to_timestamp()

    # Cohort: the quarter they joined. The single most useful slicer in SaaS.
    dim["signup_cohort"] = (
        dim["signup_date"].dt.year.astype(str)
        + "-Q"
        + dim["signup_date"].dt.quarter.astype(str)
    )

    return dim[
        [
            "customer_id",
            "company_name",
            "segment",
            "industry",
            "region",
            "acquisition_channel",
            "signup_date",
            "signup_month",
            "signup_cohort",
        ]
    ]


def build_dim_plan() -> pd.DataFrame:
    """One row per plan tier. Tiny, and it earns its place.

    It gives the dashboard a sort order. Without it, Power BI displays plan tiers
    alphabetically - Enterprise, Growth, Starter - which puts the most expensive
    plan first and reads as nonsense to anyone who knows the product.
    """
    return pd.DataFrame(
        [
            {"plan_tier": "Starter", "list_price": PLANS["Starter"], "tier_rank": 1},
            {"plan_tier": "Growth", "list_price": PLANS["Growth"], "tier_rank": 2},
            {"plan_tier": "Enterprise", "list_price": PLANS["Enterprise"], "tier_rank": 3},
        ]
    )


# ---------------------------------------------------------------------------
# FACTS
# ---------------------------------------------------------------------------


def build_fact_mrr(subs: pd.DataFrame) -> pd.DataFrame:
    """The spine. One row per customer per month, with their MRR.

    Narrow on purpose: keys, one measure, one attribute. Everything else is a
    join away in the dimensions. That is what makes the engine fast, and it is
    the discipline that keeps the totals correct.
    """
    panel = build_mrr_panel(subs)

    fact = panel.copy()
    fact["date_key"] = fact["month"].dt.strftime("%Y%m%d").astype(int)

    return fact[["date_key", "customer_id", "plan_tier", "mrr"]]


def build_fact_movement(subs: pd.DataFrame) -> pd.DataFrame:
    """One row per month: where the revenue went. Churn, contraction, expansion.

    This is the table behind the waterfall chart - the one that answers the CFO's
    actual question rather than merely restating it.
    """
    panel = build_mrr_panel(subs)
    movement = decompose_all_months(panel)

    fact = movement.copy()
    fact["date_key"] = fact["month"].dt.strftime("%Y%m%d").astype(int)

    return fact


def build_fact_forecast(subs: pd.DataFrame) -> pd.DataFrame:
    """The 13-week forecast, with the intervals the backtest earned.

    Shipped WITH its bounds. A forecast on a dashboard without an interval is an
    invitation to over-read it, and someone always does.
    """
    series = build_weekly_mrr(subs)
    results = backtest(series)

    winner = score_recent(results).iloc[0]["model"]
    intervals = empirical_intervals(results, winner)

    forward = build_forecast(series, intervals, winner)

    fact = forward.copy()
    fact["date_key"] = fact["week"].dt.strftime("%Y%m%d").astype(int)
    fact["model"] = winner

    # The actuals, so the chart can draw history and forecast on one line.
    actual = series.copy()
    actual["date_key"] = actual["week"].dt.strftime("%Y%m%d").astype(int)
    actual = actual.rename(columns={"mrr": "actual"})[["date_key", "week", "actual"]]

    return pd.concat([actual, fact], ignore_index=True)


def build_fact_reconciliation(
    raw_invoices: pd.DataFrame,
    invoices: pd.DataFrame,
    duplicates: pd.DataFrame,
    orphans: pd.DataFrame,
    subs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The proof. Both bridges, on a page, where anyone can check our working.

    Most dashboards ask to be trusted. This one shows that every dollar it
    displays ties to the billing ledger, and shows exactly where the differences
    come from.

    Put this on its own tab and link to it from the front page. It is the reason
    the other tabs get believed.
    """
    contract = load_contract()

    ledger = reconcile_ledger(raw_invoices, invoices, duplicates, orphans)

    monthly = reconcile_all_months(subs, invoices, contract)
    monthly["date_key"] = monthly["month"].dt.strftime("%Y%m%d").astype(int)

    return ledger, monthly


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------


def export(raw_dir: Path, processed_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Build every table and write it as CSV for Power BI."""
    from northwind.clean import parse_amount

    customers = pd.read_parquet(processed_dir / "customers.parquet")
    subs = pd.read_parquet(processed_dir / "subscriptions.parquet")
    invoices = pd.read_parquet(processed_dir / "invoices.parquet")
    duplicates = pd.read_parquet(processed_dir / "quarantine_duplicate_invoices.parquet")
    orphans = pd.read_parquet(processed_dir / "quarantine_orphan_invoices.parquet")

    raw_invoices = pd.read_csv(raw_dir / "invoices.csv")
    raw_invoices["amount"] = raw_invoices["amount"].map(parse_amount)

    ledger, recon_monthly = build_fact_reconciliation(
        raw_invoices, invoices, duplicates, orphans, subs
    )

    tables = {
        "dim_date": build_dim_date(),
        "dim_customer": build_dim_customer(customers),
        "dim_plan": build_dim_plan(),
        "fact_mrr": build_fact_mrr(subs),
        "fact_movement": build_fact_movement(subs),
        "fact_forecast": build_fact_forecast(subs),
        "fact_reconciliation_ledger": ledger,
        "fact_reconciliation_monthly": recon_monthly,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    print("STAR SCHEMA FOR POWER BI\n")
    for name, frame in tables.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)

        kind = "DIM " if name.startswith("dim") else "FACT"
        print(f"  [{kind}] {name:<30} {len(frame):>8,} rows  x {len(frame.columns)} cols")

    print(f"\n  Written to {out_dir}/")
    print("\n  Import all eight into Power BI, then follow docs/powerbi/BUILD.md")

    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Power BI star schema.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/powerbi"))
    args = parser.parse_args()

    export(args.raw, args.processed, args.out)


if __name__ == "__main__":
    main()
