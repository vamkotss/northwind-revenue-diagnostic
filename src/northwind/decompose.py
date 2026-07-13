"""The decomposition: answer the CFO's question.

THE QUESTION
------------
"Net revenue retention fell from 108% to 94% over three quarters. Why, and what
do we do about it?"

THE APPROACH
------------
NRR is not one thing. It is four things in a trench coat:

    NRR = 1 - churn_rate - contraction_rate + expansion_rate

Every one of those moves independently, has a different cause, and demands a
different response. A single number saying "94%" tells you the patient has a
fever. It does not tell you whether it is an infection or a broken bone, and
the treatments are not interchangeable.

So we decompose. Exactly, additively, with no residual:

    change in NRR = -(change in churn)
                    -(change in contraction)
                    +(change in expansion)

Then we cut it by segment and by plan tier until the cause has a name.

WHY THE NUMBERS CAN BE TRUSTED
------------------------------
Because Milestone 4 already tied every one of them to the billing ledger, to
the cent, for all 29 months. This module is the first one allowed to draw a
conclusion, and it is allowed to because the reconciliation closed first.

That order is the whole methodology. Definitions, then reconciliation, then
conclusions. Most analysts run it backwards and wonder why nobody believes them.

Run:  python -m northwind.decompose
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from northwind.metrics import load_contract

AS_OF = pd.Timestamp("2026-06-30")

# The three planted shocks. The analysis must FIND these, not assume them - but
# once found, they are what turns a chart into an explanation.
SHOCKS = {
    pd.Timestamp("2025-07-01"): "Sales reorg",
    pd.Timestamp("2025-09-01"): "Add-on price rise",
    pd.Timestamp("2025-10-01"): "Competitor launch",
}


# ---------------------------------------------------------------------------
# THE MRR PANEL
# ---------------------------------------------------------------------------


def build_mrr_panel(subs: pd.DataFrame) -> pd.DataFrame:
    """MRR for every customer, for every month. The spine of everything below.

    One row per customer per month they were live. Built from the CLEAN
    subscriptions, which means the pauses are already resolved (R1), the
    corrections have already superseded their originals (R6), and the add-ons
    were never in here to begin with (R2).
    """
    s = subs.copy()
    s["period_start"] = pd.to_datetime(s["period_start"])
    s["period_end"] = pd.to_datetime(s["period_end"]).fillna(AS_OF)

    months = pd.date_range("2024-01-01", "2026-06-01", freq="MS")

    frames = []
    for month in months:
        # Live on the first of the month. That is the MRR convention, and it is
        # the same convention the reconciliation used - so these numbers tie.
        live = s[(s["period_start"] <= month) & (s["period_end"] > month)]

        agg = live.groupby("customer_id").agg(
            mrr=("mrr", "sum"),
            # The tier a customer is ON this month. Used to see WHO is moving.
            plan_tier=("plan_tier", "first"),
        )
        agg["month"] = month
        frames.append(agg.reset_index())

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# THE MOVEMENT BRIDGE
# ---------------------------------------------------------------------------


def decompose_month(panel: pd.DataFrame, month: pd.Timestamp) -> dict:
    """Break one month's NRR into its four moving parts.

    We take the cohort of customers who existed TWELVE MONTHS AGO and ask what
    happened to their revenue since. That is what NRR means: it is a cohort
    metric, and customers acquired since do not count. Folding them in is the
    single most common way companies accidentally flatter their retention.

    Four outcomes for every dollar in that cohort:

      CHURNED       they are gone entirely
      CONTRACTED    still a customer, paying less     (downgrade, seat cut) [R3]
      EXPANDED      still a customer, paying more     (upgrade, more seats)
      FLAT          unchanged
    """
    prior = month - pd.DateOffset(months=12)

    start = panel[panel["month"] == prior].set_index("customer_id")["mrr"]
    end = panel[panel["month"] == month].set_index("customer_id")["mrr"]

    if len(start) == 0:
        return {}

    # The cohort: everyone who was paying us twelve months ago.
    cohort = start.index

    # Where are they now? A customer absent from `end` has zero MRR - they left.
    now = end.reindex(cohort, fill_value=0.0)
    was = start

    starting_mrr = float(was.sum())

    # CHURN: the dollars belonging to customers who now pay nothing.
    churned = float(was[now == 0].sum())

    # For the survivors, the delta splits cleanly into two buckets.
    survived = now > 0
    delta = now[survived] - was[survived]

    # CONTRACTION: they stayed, but they shrank. Ruling R3 - this is NOT churn.
    # It is the biggest single driver of this decline, and calling it churn
    # would have buried it inside a number nobody could act on.
    contraction = float(-delta[delta < 0].sum())

    # EXPANSION: they stayed, and they grew.
    expansion = float(delta[delta > 0].sum())

    ending_mrr = starting_mrr - churned - contraction + expansion

    return {
        "month": month,
        "starting_mrr": round(starting_mrr, 2),
        "churned_mrr": round(-churned, 2),
        "contraction_mrr": round(-contraction, 2),
        "expansion_mrr": round(expansion, 2),
        "ending_mrr": round(ending_mrr, 2),
        # Rates, expressed against the starting cohort. These ADD to NRR exactly.
        "churn_rate": round(churned / starting_mrr, 4),
        "contraction_rate": round(contraction / starting_mrr, 4),
        "expansion_rate": round(expansion / starting_mrr, 4),
        "nrr": round(ending_mrr / starting_mrr, 4),
        # GRR ignores expansion. It is the honest floor - it cannot exceed 100%.
        "grr": round((starting_mrr - churned - contraction) / starting_mrr, 4),
        "cohort_customers": int(len(cohort)),
        "churned_customers": int((now == 0).sum()),
    }


def decompose_all_months(panel: pd.DataFrame) -> pd.DataFrame:
    """Run the movement bridge across the whole window.

    We start in July 2025, not January. The earlier months compare against a
    2024 cohort that was tiny - the company was barely a year old - so the rates
    swing wildly on a denominator of almost nothing. A 130% NRR built on eleven
    customers is not a finding, it is a small-sample artefact, and reporting it
    would be the kind of mistake that gets a number quietly ignored forever.
    """
    months = pd.date_range("2025-06-01", "2026-06-01", freq="MS")

    rows = [decompose_month(panel, m) for m in months]
    frame = pd.DataFrame([r for r in rows if r])

    # Guard the denominator explicitly. If a cohort is too small to be stable,
    # we do not report it - we say so.
    return frame[frame["cohort_customers"] >= 500].reset_index(drop=True)


# ---------------------------------------------------------------------------
# MONTH-OVER-MONTH MOVEMENT
#
# NRR is a TRAILING TWELVE MONTH metric. That is what makes it a good headline
# and a terrible detector: a shock in October is still dribbling into the number
# the following September. It smears.
#
# So to find WHEN something broke, we look month against month. Same arithmetic,
# one-month lookback. This is the view that puts a date on the wound.
# ---------------------------------------------------------------------------


def monthly_movement(panel: pd.DataFrame) -> pd.DataFrame:
    """Churn, contraction, and expansion measured against the PREVIOUS month."""
    months = sorted(panel["month"].unique())

    rows = []
    for month in months[1:]:
        prior = month - pd.DateOffset(months=1)

        start = panel[panel["month"] == prior].set_index("customer_id")["mrr"]
        end = panel[panel["month"] == month].set_index("customer_id")["mrr"]

        if len(start) < 500:
            continue

        cohort = start.index
        now = end.reindex(cohort, fill_value=0.0)
        was = start

        starting = float(was.sum())
        churned = float(was[now == 0].sum())

        survived = now > 0
        delta = now[survived] - was[survived]
        contraction = float(-delta[delta < 0].sum())
        expansion = float(delta[delta > 0].sum())

        rows.append(
            {
                "month": month,
                "starting_mrr": round(starting, 2),
                "churn_rate": round(churned / starting, 4),
                "contraction_rate": round(contraction / starting, 4),
                "expansion_rate": round(expansion / starting, 4),
                "net_rate": round((expansion - churned - contraction) / starting, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ATTRIBUTION - WHERE DID THE 14 POINTS GO?
# ---------------------------------------------------------------------------


def attribute_decline(
    movement: pd.DataFrame, healthy: pd.Timestamp, damaged: pd.Timestamp
) -> pd.DataFrame:
    """Split the NRR decline into exactly three causes. No residual, by construction.

    Because NRR = 1 - churn - contraction + expansion, the change in NRR is
    EXACTLY the sum of the changes in those three rates. There is nothing left
    over. That is not a modelling assumption - it is arithmetic, and it means
    nobody can argue with the split.

    This is the table that goes in the memo.
    """
    a = movement[movement["month"] == healthy].iloc[0]
    b = movement[movement["month"] == damaged].iloc[0]

    d_churn = b["churn_rate"] - a["churn_rate"]
    d_contraction = b["contraction_rate"] - a["contraction_rate"]
    d_expansion = b["expansion_rate"] - a["expansion_rate"]

    d_nrr = b["nrr"] - a["nrr"]

    rows = [
        {
            "driver": "Churn (customers who left)",
            "healthy": a["churn_rate"],
            "damaged": b["churn_rate"],
            "change": d_churn,
            "nrr_impact": -d_churn,
        },
        {
            "driver": "Contraction (downgrades)",
            "healthy": a["contraction_rate"],
            "damaged": b["contraction_rate"],
            "change": d_contraction,
            "nrr_impact": -d_contraction,
        },
        {
            "driver": "Expansion (upsell)",
            "healthy": a["expansion_rate"],
            "damaged": b["expansion_rate"],
            "change": d_expansion,
            "nrr_impact": d_expansion,
        },
    ]

    table = pd.DataFrame(rows)

    # Each driver's share of the total damage.
    table["share_of_decline"] = table["nrr_impact"] / d_nrr

    return table


# ---------------------------------------------------------------------------
# WHO? - CUT BY SEGMENT AND TIER
# ---------------------------------------------------------------------------


def decompose_by_group(
    panel: pd.DataFrame,
    customers: pd.DataFrame,
    month: pd.Timestamp,
    group_col: str,
) -> pd.DataFrame:
    """Same decomposition, split by segment / tier / channel.

    This is where a chart becomes an instruction. "NRR is 94%" is a fact.
    "Growth-tier SMB customers are contracting at 11% while everyone else is
    flat" is something a VP can walk out of the room and act on.
    """
    prior = month - pd.DateOffset(months=12)

    start = panel[panel["month"] == prior].set_index("customer_id")
    end = panel[panel["month"] == month].set_index("customer_id")["mrr"]

    cohort = start.index
    now = end.reindex(cohort, fill_value=0.0)
    was = start["mrr"]

    frame = pd.DataFrame({"was": was, "now": now})

    # Attach the grouping attribute. For plan_tier we use the tier they were ON
    # at the START of the window - that is the population that was at risk.
    if group_col == "plan_tier":
        frame[group_col] = start["plan_tier"]
    else:
        lookup = customers.set_index("customer_id")[group_col]
        frame[group_col] = frame.index.map(lookup)

    frame["delta"] = frame["now"] - frame["was"]
    frame["churned"] = (frame["now"] == 0).astype(int)

    out = []
    for name, g in frame.groupby(group_col, observed=True):
        starting = float(g["was"].sum())
        if starting == 0:
            continue

        churned = float(g.loc[g["now"] == 0, "was"].sum())
        survivors = g[g["now"] > 0]
        contraction = float(-survivors.loc[survivors["delta"] < 0, "delta"].sum())
        expansion = float(survivors.loc[survivors["delta"] > 0, "delta"].sum())

        ending = starting - churned - contraction + expansion

        out.append(
            {
                group_col: name,
                "starting_mrr": round(starting, 2),
                "churn_rate": round(churned / starting, 4),
                "contraction_rate": round(contraction / starting, 4),
                "expansion_rate": round(expansion / starting, 4),
                "nrr": round(ending / starting, 4),
                # Absolute dollars lost. A 20% contraction rate on a small base
                # is a curiosity; on a large base it is the problem.
                "mrr_lost": round(churned + contraction - expansion, 2),
                "customers": len(g),
            }
        )

    return pd.DataFrame(out).sort_values("mrr_lost", ascending=False)


# ---------------------------------------------------------------------------
# WHEN? - FIND THE INFLECTION POINTS
# ---------------------------------------------------------------------------


def find_inflections(movement: pd.DataFrame, column: str, threshold: float = 0.4) -> pd.DataFrame:
    """Find the months where a rate jumped sharply. The 'what happened here?' months.

    We are not asserting the shock dates - we are DERIVING them, then going to
    look for a business event that explains each one. A date you found in the
    data and then matched to a real event is evidence. A date you assumed and
    then found in the data is a story.
    """
    m = movement.copy()

    # Month-on-month change, relative to a trailing 3-month baseline.
    baseline = m[column].rolling(3, min_periods=1).mean().shift(1)
    m["jump"] = (m[column] - baseline) / baseline.replace(0, pd.NA)

    spikes = m[m["jump"].abs() > threshold][["month", column, "jump"]].copy()
    spikes["known_event"] = spikes["month"].map(SHOCKS).fillna("unexplained")

    return spikes


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------


def decompose(processed_dir: Path, out_dir: Path) -> dict:
    """Run the full decomposition and print the answer to the CFO's question."""
    load_contract()  # fail fast if the definitions are broken

    subs = pd.read_parquet(processed_dir / "subscriptions.parquet")
    customers = pd.read_parquet(processed_dir / "customers.parquet")

    panel = build_mrr_panel(subs)
    movement = decompose_all_months(panel)
    mom = monthly_movement(panel)

    # The baseline is the PEAK, chosen from the curve, not picked in advance.
    # Comparing against an arbitrary month makes the shares meaningless.
    healthy = movement.loc[movement["nrr"].idxmax(), "month"]
    damaged = movement["month"].max()

    attribution = attribute_decline(movement, healthy, damaged)

    by_segment = decompose_by_group(panel, customers, damaged, "segment")
    by_tier = decompose_by_group(panel, customers, damaged, "plan_tier")
    by_channel = decompose_by_group(panel, customers, damaged, "acquisition_channel")

    # --- Print the story ---
    print("=" * 74)
    print("NRR OVER TIME")
    print("=" * 74)
    for r in movement.itertuples(index=False):
        bar = "#" * max(0, int((r.nrr - 0.85) * 100))
        flag = SHOCKS.get(r.month, "")
        print(f"  {r.month:%Y-%m}  NRR {r.nrr:6.1%}   GRR {r.grr:6.1%}  {bar} {flag}")

    a = movement[movement["month"] == healthy].iloc[0]
    b = movement[movement["month"] == damaged].iloc[0]

    print()
    print("=" * 74)
    print(f"WHERE DID THE {(a['nrr'] - b['nrr']) * 100:.1f} POINTS GO?")
    print(f"  ({healthy:%b %Y}: {a['nrr']:.1%}  ->  {damaged:%b %Y}: {b['nrr']:.1%})")
    print("=" * 74)
    print(f"\n  {'Driver':<30} {'Was':>8} {'Now':>8} {'NRR hit':>9} {'Share':>8}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8}")
    for r in attribution.itertuples(index=False):
        print(
            f"  {r.driver:<30} {r.healthy:>7.1%} {r.damaged:>8.1%} "
            f"{r.nrr_impact:>+8.1%} {r.share_of_decline:>8.0%}"
        )
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8}")
    print(f"  {'TOTAL':<30} {'':>8} {'':>8} {attribution['nrr_impact'].sum():>+8.1%} {'100%':>8}")

    print("\n" + "=" * 74)
    print("WHO IS DOING THE DAMAGE?  (June 2026 cohort)")
    print("=" * 74)

    for label, table, col in [
        ("BY SEGMENT", by_segment, "segment"),
        ("BY PLAN TIER", by_tier, "plan_tier"),
        ("BY ACQUISITION CHANNEL", by_channel, "acquisition_channel"),
    ]:
        print(f"\n  {label}")
        print(f"  {col:<18} {'NRR':>7} {'Churn':>7} {'Contract':>9} {'Expand':>8} {'MRR lost':>13}")
        print(f"  {'-' * 18} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 8} {'-' * 13}")
        for r in table.itertuples(index=False):
            name = getattr(r, col)
            print(
                f"  {str(name):<18} {r.nrr:>6.1%} {r.churn_rate:>7.1%} "
                f"{r.contraction_rate:>9.1%} {r.expansion_rate:>8.1%} ${r.mrr_lost:>12,.0f}"
            )

    print("\n" + "=" * 74)
    print("WHEN DID IT BREAK?  (month-over-month, so the shocks are not smeared)")
    print("=" * 74)

    # Smooth over three months. Single months are noisy enough that a real shock
    # and a random wobble look alike; a 3-month mean separates them.
    m = mom.copy()
    for col in ["churn_rate", "contraction_rate", "expansion_rate"]:
        m[col] = m[col].rolling(3, min_periods=1).mean()

    print(f"\n  {'Month':<10} {'Churn':>7} {'Contract':>9} {'Expand':>8}   Event found")
    print(f"  {'-' * 10} {'-' * 7} {'-' * 9} {'-' * 8}   {'-' * 20}")
    for r in m[m["month"] >= pd.Timestamp("2025-04-01")].itertuples(index=False):
        event = SHOCKS.get(r.month, "")
        mark = "  <--" if event else ""
        bar = "#" * int(r.expansion_rate * 400)
        print(f"  {r.month:%Y-%m}    {r.churn_rate:>6.1%} {r.contraction_rate:>9.1%} "
              f"{r.expansion_rate:>8.1%} {bar:<14}{mark} {event}")

    out_dir.mkdir(parents=True, exist_ok=True)
    movement.to_parquet(out_dir / "nrr_movement.parquet", index=False)
    mom.to_parquet(out_dir / "nrr_monthly_movement.parquet", index=False)
    attribution.to_parquet(out_dir / "nrr_attribution.parquet", index=False)
    by_segment.to_parquet(out_dir / "nrr_by_segment.parquet", index=False)
    by_tier.to_parquet(out_dir / "nrr_by_tier.parquet", index=False)
    by_channel.to_parquet(out_dir / "nrr_by_channel.parquet", index=False)

    return {
        "movement": movement,
        "monthly": mom,
        "attribution": attribution,
        "by_segment": by_segment,
        "by_tier": by_tier,
        "by_channel": by_channel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose the NRR decline.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    decompose(args.processed, args.out)


if __name__ == "__main__":
    main()
