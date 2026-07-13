"""Salvaging the contaminated pricing experiment.

THE SETUP
---------
In January 2026 the company ran a retention experiment: treatment customers
were offered a discounted renewal. The VP of Product wants to know whether it
worked, so the team can decide whether to roll it out.

The experiment is broken in two ways, and neither is visible in the summary
statistics the dashboard shows.

WHAT A JUNIOR ANALYST DOES
--------------------------
Compares MRR retention between the two arms, finds treatment is 1.9% worse,
concludes the discount is harmful, and recommends killing it. Confident, fast,
and wrong.

WHAT IS ACTUALLY WRONG
----------------------
DEFECT A - SAMPLE RATIO MISMATCH.
The split is 54/46, not 50/50 (chi-square p = 0.0003). An SRM is not a cosmetic
problem. It means the randomisation MECHANISM failed - and a broken mechanism
usually fails in a way that correlates with something. Here it did: SMB
customers were pushed into control and Enterprise customers into treatment.

That single fact destroys the comparison. The arms differ systematically before
the treatment was ever applied, so ANY difference you measure afterwards is
confounded with the difference you started with.

DEFECT B - CONTAMINATION.
A mid-experiment deploy re-randomised 72 customers into the opposite arm. They
now appear in both. Their outcomes cannot be attributed to either treatment and
must be excluded.

THE HONEST CONCLUSION
---------------------
After excluding the contaminated customers and stratifying by segment to compare
like with like, the effect is -2.3% with a 95% confidence interval of
[-10.5%, +6.1%].

That interval spans zero and is sixteen points wide. The experiment CANNOT
distinguish a meaningful effect from no effect at all.

So the answer to the VP is not "it works" and not "it does not work". It is:
"This experiment cannot tell you, and here is precisely why. Fix the assignment
bug and re-run it. If you ship or kill the discount based on this data, you are
flipping a coin and calling it evidence."

Knowing when a result CANNOT be salvaged, and saying so out loud, is the
difference between an analyst who is trusted and one who is merely fast.

Run:  python -m northwind.experiment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from northwind.decompose import build_mrr_panel

# The experiment window. Outcomes are measured from just before assignment to
# the latest month we have.
PRE_PERIOD = pd.Timestamp("2026-01-01")
POST_PERIOD = pd.Timestamp("2026-06-01")

# Below this, a stratum is too small to compare and gets excluded rather than
# quietly producing a noisy number that looks precise.
MIN_STRATUM = 20

# An SRM this unlikely under a fair coin is not bad luck. It is a bug.
SRM_ALPHA = 0.01


# ---------------------------------------------------------------------------
# DIAGNOSTIC 1 - SAMPLE RATIO MISMATCH
# ---------------------------------------------------------------------------


def check_srm(assignments: pd.DataFrame) -> dict:
    """Did the randomisation actually randomise? Chi-square against 50/50.

    THIS IS THE FIRST THING YOU CHECK. Before the p-value on the outcome, before
    the confidence interval, before anything. If the split is wrong, the
    randomisation failed, and every number downstream is measuring the failure
    rather than the treatment.

    Most analysts never run this test. It takes four lines.
    """
    # One row per customer - a contaminated customer must not be counted twice.
    first = assignments.drop_duplicates("customer_id", keep="first")

    counts = first["variant"].value_counts()
    n = int(counts.sum())

    control = int(counts.get("control", 0))
    treatment = int(counts.get("treatment", 0))

    # Under a correct 50/50 assignment, we expect half in each arm.
    expected = [n / 2, n / 2]

    chi2, p_value = stats.chisquare([control, treatment], expected)

    return {
        "control_n": control,
        "treatment_n": treatment,
        "control_share": round(control / n, 4),
        "chi2": round(float(chi2), 2),
        "p_value": float(p_value),
        "srm_detected": bool(p_value < SRM_ALPHA),
    }


# ---------------------------------------------------------------------------
# DIAGNOSTIC 2 - CONTAMINATION
# ---------------------------------------------------------------------------


def check_contamination(assignments: pd.DataFrame) -> dict:
    """Find customers who appear in BOTH arms.

    A customer who was in control on day 1 and treatment on day 10 experienced
    both. Their outcome cannot be attributed to either. There is no clever
    statistical fix - they have to come out.
    """
    arms_per_customer = assignments.groupby("customer_id")["variant"].nunique()

    contaminated = arms_per_customer[arms_per_customer > 1].index

    return {
        "contaminated_customers": len(contaminated),
        "total_customers": assignments["customer_id"].nunique(),
        "contamination_rate": round(
            len(contaminated) / assignments["customer_id"].nunique(), 4
        ),
        "contaminated_ids": set(contaminated),
    }


# ---------------------------------------------------------------------------
# DIAGNOSTIC 3 - WHY THE SRM IS LETHAL
# ---------------------------------------------------------------------------


def check_balance(assignments: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Were the two arms comparable BEFORE the treatment was applied?

    This is the test that explains the SRM. An SRM tells you the mechanism
    broke; a balance check tells you HOW, and therefore how badly the readout
    is poisoned.

    If Enterprise customers are twice as likely to be in treatment, then
    treatment was dealt a better hand before the experiment even started - and
    the outcome difference you measure is mostly just that.
    """
    first = assignments.drop_duplicates("customer_id", keep="first")
    merged = first.merge(customers[["customer_id", "segment"]], on="customer_id")

    balance = pd.crosstab(merged["segment"], merged["variant"], normalize="columns")
    balance = balance.reset_index()

    # The imbalance: how far apart are the two arms on this covariate?
    balance["difference"] = (balance["treatment"] - balance["control"]).round(4)
    balance["control"] = balance["control"].round(4)
    balance["treatment"] = balance["treatment"].round(4)

    return balance.sort_values("difference", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# THE OUTCOME
# ---------------------------------------------------------------------------


def build_outcomes(
    assignments: pd.DataFrame, subs: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    """MRR retention for every customer in the experiment.

    The outcome is: of the MRR this customer was paying just before the
    experiment, how much are they still paying now? It is the metric the
    discount was supposed to protect.
    """
    panel = build_mrr_panel(subs)

    pre = panel[panel["month"] == PRE_PERIOD].set_index("customer_id")["mrr"]
    post = panel[panel["month"] == POST_PERIOD].set_index("customer_id")["mrr"]

    frame = assignments.merge(customers[["customer_id", "segment"]], on="customer_id")

    frame["mrr_pre"] = frame["customer_id"].map(pre)
    # Absent from the post panel means zero MRR - they left.
    frame["mrr_post"] = frame["customer_id"].map(post).fillna(0.0)

    # Only customers who were actually paying us when the experiment started.
    frame = frame.dropna(subset=["mrr_pre"])
    return frame[frame["mrr_pre"] > 0].copy()


def naive_readout(outcomes: pd.DataFrame) -> dict:
    """The analysis a junior analyst ships. It is wrong, and it looks fine.

    No SRM check. No contamination check. No balance check. Just two averages
    and a difference - and a recommendation that would cost the company real
    money in the wrong direction.
    """
    first = outcomes.drop_duplicates("customer_id", keep="first")

    result = {}
    for variant, group in first.groupby("variant"):
        result[variant] = {
            "n": len(group),
            "retention": float(group["mrr_post"].sum() / group["mrr_pre"].sum()),
            "avg_mrr": float(group["mrr_pre"].mean()),
        }

    lift = result["treatment"]["retention"] - result["control"]["retention"]

    return {"arms": result, "lift": round(lift, 4)}


def stratified_lift(outcomes: pd.DataFrame) -> float:
    """Compare like with like: measure the effect WITHIN each segment, then pool.

    Stratification is the standard repair for a known imbalance. It is a
    genuine improvement - and, as the confidence interval below shows, it is
    NOT a cure. You cannot fully undo a broken randomisation by adjusting for
    the covariates you happened to observe, because you do not know what else
    the bug correlated with.
    """
    total_weighted = 0.0
    total_weight = 0.0

    for _, group in outcomes.groupby("segment"):
        control = group[group["variant"] == "control"]
        treatment = group[group["variant"] == "treatment"]

        # Too small to compare? Exclude it. A noisy number that looks precise is
        # worse than an honest gap.
        if len(control) < MIN_STRATUM or len(treatment) < MIN_STRATUM:
            continue
        if control["mrr_pre"].sum() == 0 or treatment["mrr_pre"].sum() == 0:
            continue

        lift = (
            treatment["mrr_post"].sum() / treatment["mrr_pre"].sum()
            - control["mrr_post"].sum() / control["mrr_pre"].sum()
        )

        # Weight each stratum by the MRR at stake in it.
        weight = float(group["mrr_pre"].sum())
        total_weighted += lift * weight
        total_weight += weight

    return total_weighted / total_weight if total_weight else float("nan")


def corrected_readout(
    outcomes: pd.DataFrame, contaminated: set, n_bootstrap: int = 600, seed: int = 7
) -> dict:
    """Exclude the contaminated, stratify by segment, and put an interval on it.

    THE POINT OF THE INTERVAL. A point estimate of -2.3% invites someone to act
    on it. An interval of [-10.5%, +6.1%] makes it obvious that acting on it
    would be a coin flip with extra steps.

    A confidence interval is not decoration. It is the thing that stops a number
    being over-read - and over-reading is how a broken experiment turns into a
    bad decision.
    """
    clean = outcomes[~outcomes["customer_id"].isin(contaminated)].copy()

    point = stratified_lift(clean)

    # Bootstrap: resample the customers with replacement, recompute the lift,
    # and see how much the answer wobbles. If it wobbles across zero, we cannot
    # tell the sign of the effect - let alone its size.
    rng = np.random.default_rng(seed)

    estimates = []
    for _ in range(n_bootstrap):
        sample = clean.sample(len(clean), replace=True, random_state=int(rng.integers(1e6)))
        value = stratified_lift(sample)
        if not np.isnan(value):
            estimates.append(value)

    estimates = np.array(estimates)
    lower, upper = np.percentile(estimates, [2.5, 97.5])

    return {
        "n_after_exclusion": len(clean),
        "point_estimate": round(float(point), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "ci_width": round(float(upper - lower), 4),
        "includes_zero": bool(lower < 0 < upper),
    }


def by_segment(outcomes: pd.DataFrame, contaminated: set) -> pd.DataFrame:
    """The within-segment comparison, shown so the reader can check our working."""
    clean = outcomes[~outcomes["customer_id"].isin(contaminated)]

    rows = []
    for segment, group in clean.groupby("segment"):
        control = group[group["variant"] == "control"]
        treatment = group[group["variant"] == "treatment"]

        if len(control) < MIN_STRATUM or len(treatment) < MIN_STRATUM:
            continue

        ret_c = float(control["mrr_post"].sum() / control["mrr_pre"].sum())
        ret_t = float(treatment["mrr_post"].sum() / treatment["mrr_pre"].sum())

        rows.append(
            {
                "segment": segment,
                "control_n": len(control),
                "treatment_n": len(treatment),
                "control_retention": round(ret_c, 4),
                "treatment_retention": round(ret_t, 4),
                "lift": round(ret_t - ret_c, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------


def analyse(raw_dir: Path, processed_dir: Path, out_dir: Path) -> dict:
    """Run every diagnostic, then say plainly what the experiment can support."""
    assignments = pd.read_parquet(raw_dir / "experiment_assignments.parquet")
    subs = pd.read_parquet(processed_dir / "subscriptions.parquet")
    customers = pd.read_parquet(processed_dir / "customers.parquet")

    srm = check_srm(assignments)
    contamination = check_contamination(assignments)
    balance = check_balance(assignments, customers)

    outcomes = build_outcomes(assignments, subs, customers)
    naive = naive_readout(outcomes)
    corrected = corrected_readout(outcomes, contamination["contaminated_ids"])
    segments = by_segment(outcomes, contamination["contaminated_ids"])

    # --- Print ---
    print("=" * 74)
    print("STEP 1 - SAMPLE RATIO MISMATCH  (run this BEFORE looking at any outcome)")
    print("=" * 74)
    print(f"\n  control   : {srm['control_n']:>5}  ({srm['control_share']:.1%})")
    print(f"  treatment : {srm['treatment_n']:>5}  ({1 - srm['control_share']:.1%})")
    print(f"  chi-square: {srm['chi2']}   p = {srm['p_value']:.2e}")
    if srm["srm_detected"]:
        print("\n  >>> SRM DETECTED. The randomisation mechanism failed.")
        print("  >>> An SRM is not cosmetic. It means the arms may differ in ways")
        print("  >>> that have nothing to do with the treatment.")

    print("\n" + "=" * 74)
    print("STEP 2 - CONTAMINATION")
    print("=" * 74)
    print(f"\n  Customers in BOTH arms : {contamination['contaminated_customers']}")
    print(f"  Contamination rate     : {contamination['contamination_rate']:.1%}")
    print("\n  >>> These customers experienced both treatments. Their outcomes")
    print("  >>> cannot be attributed to either. They must be excluded.")

    print("\n" + "=" * 74)
    print("STEP 3 - WERE THE ARMS EVER COMPARABLE?  (why the SRM is lethal)")
    print("=" * 74)
    print(f"\n  {'Segment':<14} {'control':>9} {'treatment':>10} {'difference':>11}")
    print(f"  {'-' * 14} {'-' * 9} {'-' * 10} {'-' * 11}")
    for r in balance.itertuples(index=False):
        print(f"  {r.segment:<14} {r.control:>8.1%} {r.treatment:>9.1%} {r.difference:>+10.1%}")
    print("\n  >>> The arms differed BEFORE the treatment was applied. Any outcome")
    print("  >>> difference is confounded with the difference we started with.")

    print("\n" + "=" * 74)
    print("STEP 4 - THE NAIVE READOUT  (what gets shipped when nobody checks)")
    print("=" * 74)
    print()
    for variant in ["control", "treatment"]:
        arm = naive["arms"][variant]
        print(
            f"  {variant:<10} n={arm['n']:>4}   MRR retained {arm['retention']:6.1%}   "
            f"avg MRR ${arm['avg_mrr']:>7,.0f}"
        )
    print(f"\n  >>> LIFT: {naive['lift']:+.1%}")
    print("  >>> A junior analyst kills the discount here. They would be wrong -")
    print("  >>> treatment simply held more Enterprise customers, whose expansion")
    print("  >>> had already collapsed for entirely unrelated reasons.")

    print("\n" + "=" * 74)
    print("STEP 5 - THE CORRECTED READOUT  (contaminated excluded, stratified)")
    print("=" * 74)
    print(f"\n  {'Segment':<14} {'n(c)':>5} {'n(t)':>5} {'ret(c)':>8} {'ret(t)':>8} {'lift':>8}")
    print(f"  {'-' * 14} {'-' * 5} {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in segments.itertuples(index=False):
        print(
            f"  {r.segment:<14} {r.control_n:>5} {r.treatment_n:>5} "
            f"{r.control_retention:>7.1%} {r.treatment_retention:>7.1%} {r.lift:>+7.1%}"
        )

    print(f"\n  Point estimate : {corrected['point_estimate']:+.1%}")
    print(f"  95% CI         : [{corrected['ci_lower']:+.1%}, {corrected['ci_upper']:+.1%}]")
    print(f"  CI width       : {corrected['ci_width']:.1%}")

    print("\n" + "=" * 74)
    print("THE VERDICT")
    print("=" * 74)

    if corrected["includes_zero"]:
        print("\n  This experiment CANNOT support a causal claim in either direction.")
        print()
        print("  The confidence interval spans zero and is "
              f"{corrected['ci_width']:.0%} wide. We cannot")
        print("  distinguish a meaningful effect from no effect at all.")
        print()
        print("  Stratification adjusts for the imbalance we OBSERVED. It cannot")
        print("  adjust for whatever else the assignment bug correlated with, and")
        print("  we do not know what that was. A broken randomisation is not")
        print("  something you repair after the fact - it is something you re-run.")
        print()
        print("  RECOMMENDATION")
        print("    1. Fix the assignment bug (it is bucketing on a field that")
        print("       correlates with customer size).")
        print("    2. Re-run with a verified 50/50 split and an SRM check that")
        print("       fires automatically on day one.")
        print("    3. Do NOT ship or kill the discount on this data. Acting on it")
        print("       is a coin flip with extra steps.")
    else:
        print("\n  The effect is distinguishable from zero even after correction.")

    out_dir.mkdir(parents=True, exist_ok=True)
    balance.to_parquet(out_dir / "experiment_balance.parquet", index=False)
    segments.to_parquet(out_dir / "experiment_by_segment.parquet", index=False)

    return {
        "srm": srm,
        "contamination": contamination,
        "balance": balance,
        "naive": naive,
        "corrected": corrected,
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Salvage the contaminated pricing experiment.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    analyse(args.raw, args.processed, args.out)


if __name__ == "__main__":
    main()
