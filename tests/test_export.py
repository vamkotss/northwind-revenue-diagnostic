"""Tests for the Power BI star schema.

WHY TEST AN EXPORT?
-------------------
Because every failure mode here is SILENT.

Power BI does not throw errors when your data model is wrong. It draws a chart
with holes in it. It returns blanks for a year-over-year comparison. It reports
an MRR twelve times larger than reality and formats it beautifully.

You do not find out from an error message. You find out in a meeting.

So the constraints Power BI silently depends on get asserted here, in Python,
where they fail loudly and in CI:

  - the date dimension has NO GAPS (time intelligence dies quietly without this)
  - every fact key EXISTS in its dimension (orphan keys vanish from every visual)
  - the keys are UNIQUE in the dimensions (duplicates silently multiply totals)
  - the reconciliation still ties (the whole dashboard rests on that page)
"""

from __future__ import annotations

import pandas as pd
import pytest

from northwind.clean import clean, parse_amount
from northwind.export import (
    build_dim_customer,
    build_dim_date,
    build_dim_plan,
    build_fact_forecast,
    build_fact_movement,
    build_fact_mrr,
    build_fact_reconciliation,
)
from northwind.generate import SEED, generate


@pytest.fixture(scope="module")
def star(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    processed = tmp_path_factory.mktemp("processed")

    generate(raw, seed=SEED)
    clean(raw, processed)

    customers = pd.read_parquet(processed / "customers.parquet")
    subs = pd.read_parquet(processed / "subscriptions.parquet")
    invoices = pd.read_parquet(processed / "invoices.parquet")
    duplicates = pd.read_parquet(processed / "quarantine_duplicate_invoices.parquet")
    orphans = pd.read_parquet(processed / "quarantine_orphan_invoices.parquet")

    raw_invoices = pd.read_csv(raw / "invoices.csv")
    raw_invoices["amount"] = raw_invoices["amount"].map(parse_amount)

    ledger, recon = build_fact_reconciliation(
        raw_invoices, invoices, duplicates, orphans, subs
    )

    return {
        "dim_date": build_dim_date(),
        "dim_customer": build_dim_customer(customers),
        "dim_plan": build_dim_plan(),
        "fact_mrr": build_fact_mrr(subs),
        "fact_movement": build_fact_movement(subs),
        "fact_forecast": build_fact_forecast(subs),
        "ledger": ledger,
        "recon": recon,
    }


# ---------------------------------------------------------------------------
# 1. THE DATE DIMENSION
# ---------------------------------------------------------------------------


def test_date_dimension_has_no_gaps(star):
    """A GAPLESS calendar. Power BI's time intelligence depends on it absolutely.

    One missing day and SAMEPERIODLASTYEAR starts returning blanks - not errors,
    BLANKS - and your year-over-year chart quietly develops holes that you will
    spend an afternoon blaming on the data.
    """
    dim = star["dim_date"]

    expected = pd.date_range(dim["date"].min(), dim["date"].max(), freq="D")

    assert len(dim) == len(expected), "the calendar has gaps"
    assert (dim["date"].to_numpy() == expected.to_numpy()).all()


def test_date_keys_are_unique(star):
    """A duplicate key in a dimension silently multiplies every total joined to it.

    No error. No warning. Just a number that is exactly twice what it should be,
    presented with total confidence.
    """
    assert not star["dim_date"]["date_key"].duplicated().any()


def test_month_labels_have_a_sort_column(star):
    """Without one, Power BI sorts months alphabetically and 'Apr' precedes 'Jan'."""
    dim = star["dim_date"]

    assert "year_month" in dim.columns
    assert "year_month_sort" in dim.columns

    # The sort column must actually order the labels correctly.
    ordered = dim.drop_duplicates("year_month").sort_values("year_month_sort")
    assert ordered["year_month"].is_monotonic_increasing


def test_the_calendar_covers_the_forecast(star):
    """The forecast runs into the future. The calendar must reach it.

    If the date table stops at today, every forecast row joins to nothing and
    disappears from the chart entirely. The visual will look fine - just empty
    on the right-hand side.
    """
    last_forecast = pd.to_datetime(star["fact_forecast"]["week"]).max()
    last_calendar = star["dim_date"]["date"].max()

    assert last_calendar >= last_forecast, (
        "the calendar ends before the forecast does - the forecast will vanish"
    )


# ---------------------------------------------------------------------------
# 2. REFERENTIAL INTEGRITY
# ---------------------------------------------------------------------------


def test_dimension_keys_are_unique(star):
    """One row per customer, one per plan. A duplicate here breaks every total."""
    assert not star["dim_customer"]["customer_id"].duplicated().any()
    assert not star["dim_plan"]["plan_tier"].duplicated().any()


def test_every_fact_customer_exists_in_the_dimension(star):
    """No orphan keys. An orphan silently disappears from every visual.

    This is the same class of bug as the orphan invoices in the cleaning layer,
    and it fails the same way: quietly, with a plausible-looking number.
    """
    known = set(star["dim_customer"]["customer_id"])
    fact_customers = set(star["fact_mrr"]["customer_id"])

    orphans = fact_customers - known

    assert not orphans, f"{len(orphans)} customers in fact_mrr are missing from dim_customer"


def test_every_fact_date_exists_in_the_calendar(star):
    """Same rule, for dates."""
    known = set(star["dim_date"]["date_key"])

    for fact_name in ["fact_mrr", "fact_movement"]:
        keys = set(star[fact_name]["date_key"])
        missing = keys - known

        assert not missing, f"{fact_name} has {len(missing)} date keys not in dim_date"


def test_every_fact_plan_exists_in_the_dimension(star):
    """And for plan tiers."""
    known = set(star["dim_plan"]["plan_tier"])
    used = set(star["fact_mrr"]["plan_tier"].dropna())

    assert used <= known, f"unknown plan tiers in fact_mrr: {used - known}"


# ---------------------------------------------------------------------------
# 3. THE FACTS ARE SANE
# ---------------------------------------------------------------------------


def test_fact_mrr_is_narrow(star):
    """A fact table is keys and measures. Nothing else.

    Every extra descriptive column is a chance for the same attribute to exist in
    two places and disagree - and a slower engine besides. Attributes belong in
    dimensions.
    """
    columns = set(star["fact_mrr"].columns)

    assert columns == {"date_key", "customer_id", "plan_tier", "mrr"}, (
        f"fact_mrr has drifted wide: {sorted(columns)}"
    )


def test_fact_mrr_is_one_row_per_customer_per_month(star):
    """The declared grain is the actual grain.

    If a customer appears twice in one month, every MRR total is inflated and
    nothing in Power BI will tell you.
    """
    fact = star["fact_mrr"]

    duplicates = fact.duplicated(subset=["date_key", "customer_id"]).sum()

    assert duplicates == 0, f"{duplicates} duplicate customer-months in fact_mrr"


def test_mrr_is_never_negative(star):
    """Contracted MRR cannot be below zero. If it is, the cleaning layer failed."""
    assert (star["fact_mrr"]["mrr"] >= 0).all()


def test_the_movement_table_carries_the_finding(star):
    """The headline numbers are in the table the dashboard reads."""
    movement = star["fact_movement"]

    for column in ["nrr", "grr", "churn_rate", "contraction_rate", "expansion_rate"]:
        assert column in movement.columns, f"fact_movement is missing {column}"

    # And the story survives the round trip into the export.
    assert movement["nrr"].max() > 1.04
    assert movement["nrr"].iloc[-1] < 0.97


def test_the_forecast_carries_its_interval(star):
    """A forecast on a dashboard without bounds is an invitation to over-read it."""
    forecast = star["fact_forecast"]

    future = forecast[forecast["forecast"].notna()]

    assert len(future) == 13
    assert (future["lower"] <= future["forecast"]).all()
    assert (future["forecast"] <= future["upper"]).all()


def test_the_forecast_includes_history(star):
    """History and forecast on one table, so they draw as one line."""
    forecast = star["fact_forecast"]

    assert forecast["actual"].notna().sum() > 50, "no history to draw"
    assert forecast["forecast"].notna().sum() == 13


# ---------------------------------------------------------------------------
# 4. THE PROOF PAGE STILL PROVES SOMETHING
# ---------------------------------------------------------------------------


def test_the_reconciliation_page_still_ties(star):
    """The dashboard ships a page claiming every month reconciles. It had better.

    This is the page that makes the other three believable. If the residual is
    not zero, the claim on the page is a lie - and one visible lie discredits
    every number next to it.
    """
    recon = star["recon"]

    assert (recon["residual"].abs() < 0.01).all(), (
        "the reconciliation page would display a non-zero residual - "
        "the dashboard's central claim is false"
    )


def test_the_ledger_bridge_is_exported(star):
    """The raw-to-clean bridge makes it onto the page."""
    ledger = star["ledger"]

    residual = ledger[ledger["line"] == "UNEXPLAINED RESIDUAL"].iloc[0]

    assert abs(residual["dollars"]) < 0.01
    assert residual["rows"] == 0
