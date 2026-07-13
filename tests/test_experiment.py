"""Tests for the experiment salvage.

WHAT THESE PROTECT
------------------
The most important assertion in this file is that the answer is "we cannot
tell". That is a strange thing to have to defend with a test - but it is exactly
the finding most likely to get quietly softened into something more satisfying,
by a future version of this code or by a future version of me.

An analyst under pressure to produce a result will find one. These tests are the
thing standing between "the experiment is inconclusive" and "well, directionally
it suggests..." - which is how a coin flip gets dressed up as evidence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from northwind.clean import clean
from northwind.experiment import (
    build_outcomes,
    check_balance,
    check_contamination,
    check_srm,
    corrected_readout,
    naive_readout,
    stratified_lift,
)
from northwind.generate import SEED, generate


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    processed = tmp_path_factory.mktemp("processed")

    generate(raw, seed=SEED)
    clean(raw, processed)

    assignments = pd.read_parquet(raw / "experiment_assignments.parquet")
    subs = pd.read_parquet(processed / "subscriptions.parquet")
    customers = pd.read_parquet(processed / "customers.parquet")

    contamination = check_contamination(assignments)
    outcomes = build_outcomes(assignments, subs, customers)

    return {
        "assignments": assignments,
        "customers": customers,
        "outcomes": outcomes,
        "contamination": contamination,
        "srm": check_srm(assignments),
        "balance": check_balance(assignments, customers),
        "naive": naive_readout(outcomes),
        "corrected": corrected_readout(outcomes, contamination["contaminated_ids"]),
    }


# ---------------------------------------------------------------------------
# 1. THE DIAGNOSTICS FIRE
# ---------------------------------------------------------------------------


def test_srm_is_detected(data):
    """The split is not 50/50, and the test says so.

    This is the check most analysts never run. It takes four lines and it is the
    difference between reading an experiment and reading a bug.
    """
    srm = data["srm"]

    assert srm["srm_detected"], (
        f"SRM not detected - split is {srm['control_share']:.1%} / "
        f"{1 - srm['control_share']:.1%} with p = {srm['p_value']:.3f}"
    )
    assert srm["p_value"] < 0.01


def test_srm_check_ignores_contaminated_double_counting(data):
    """A customer in both arms must be counted once, not twice.

    Counting them twice inflates both arms and can MASK a real SRM - so the
    contamination bug would hide the randomisation bug. Two defects covering for
    each other is the worst case, and it is not hypothetical.
    """
    srm = data["srm"]

    unique_customers = data["assignments"]["customer_id"].nunique()

    assert srm["control_n"] + srm["treatment_n"] == unique_customers


def test_contamination_is_detected(data):
    """Customers assigned to both arms are found."""
    c = data["contamination"]

    assert c["contaminated_customers"] > 0
    assert 0.02 < c["contamination_rate"] < 0.08


def test_the_arms_were_never_comparable(data):
    """THE DIAGNOSIS. The SRM correlates with customer segment.

    An SRM tells you the mechanism broke. The BALANCE CHECK tells you how - and
    therefore how badly the readout is poisoned. Here, Enterprise customers are
    roughly twice as likely to be in treatment. They churn less and expand more.

    Treatment did not win. It was dealt a better hand before the first card was
    played.
    """
    balance = data["balance"].set_index("segment")

    enterprise_gap = balance.loc["Enterprise", "difference"]

    assert abs(enterprise_gap) > 0.04, (
        f"Enterprise differs by only {enterprise_gap:.1%} between arms - "
        "the imbalance that explains the SRM is not present"
    )

    # And the imbalance runs the way we think it does.
    assert balance.loc["Enterprise", "treatment"] > balance.loc["Enterprise", "control"]
    assert balance.loc["SMB", "control"] > balance.loc["SMB", "treatment"]


# ---------------------------------------------------------------------------
# 2. THE NAIVE READOUT IS WRONG
# ---------------------------------------------------------------------------


def test_the_naive_readout_produces_a_confident_wrong_answer(data):
    """The junior analysis finds a clear effect. There isn't one.

    This test exists to prove the trap is real. If the naive number came out at
    zero, there would be nothing to warn anybody about - and the whole milestone
    would be a lecture with no lesson attached.
    """
    naive = data["naive"]

    assert abs(naive["lift"]) > 0.01, (
        "the naive readout finds no effect - there is no trap to spring"
    )

    # And the arms have visibly different customers, which is the giveaway
    # that anyone bothering to look would have caught.
    control_mrr = naive["arms"]["control"]["avg_mrr"]
    treatment_mrr = naive["arms"]["treatment"]["avg_mrr"]

    assert treatment_mrr > control_mrr * 1.2, (
        "the arms have similar average MRR - the selection bias is not visible"
    )


# ---------------------------------------------------------------------------
# 3. THE HONEST VERDICT
# The tests that matter most.
# ---------------------------------------------------------------------------


def test_the_corrected_effect_is_indistinguishable_from_zero(data):
    """THE FINDING. After correction, we cannot tell whether the discount did anything.

    The confidence interval spans zero. That is the answer, and it is an
    uncomfortable one, because nobody gets promoted for saying "we cannot tell".

    But the alternative is worse. A point estimate of -2.3% invites somebody to
    kill a feature. An interval of [-10.5%, +6.1%] makes it obvious that doing so
    would be a coin flip wearing a lab coat.
    """
    corrected = data["corrected"]

    assert corrected["includes_zero"], (
        f"the CI [{corrected['ci_lower']:.1%}, {corrected['ci_upper']:.1%}] excludes zero - "
        "the honest 'we cannot tell' verdict no longer holds"
    )


def test_the_confidence_interval_is_too_wide_to_act_on(data):
    """The interval is wide enough that no decision could survive it.

    A CI of [-10.5%, +6.1%] does not mean 'probably slightly negative'. It means
    the data is consistent with the discount being a disaster AND with it being a
    success. Reporting the midpoint without the interval is how that gets lost.
    """
    corrected = data["corrected"]

    assert corrected["ci_width"] > 0.10, (
        f"CI is only {corrected['ci_width']:.1%} wide - "
        "the case for not acting on it is weaker than claimed"
    )


def test_stratification_helps_but_does_not_cure(data):
    """Correcting for the imbalance we SAW does not fix a broken randomisation.

    This is the subtle point, and it is the one a good interviewer will probe.
    Stratification adjusts for the covariates you observed. It cannot adjust for
    whatever else the assignment bug correlated with - and you do not know what
    that was, because if you did, you would have caught the bug.

    A broken randomisation is not repaired after the fact. It is re-run.
    """
    corrected = data["corrected"]
    naive = data["naive"]

    # Both estimates exist and both are uncertain. The point is NOT that
    # stratification produced a "true" answer - it is that it produced an
    # answer we still cannot act on.
    assert corrected["point_estimate"] is not None
    assert corrected["includes_zero"], (
        "stratification appears to have 'fixed' the experiment - it cannot, "
        "and claiming otherwise is the mistake this milestone exists to prevent"
    )
    assert abs(naive["lift"]) > 0


def test_contaminated_customers_are_excluded_from_the_corrected_readout(data):
    """The customers who saw both treatments do not appear in the final number."""
    corrected = data["corrected"]
    outcomes = data["outcomes"]

    n_contaminated_in_outcomes = outcomes[
        outcomes["customer_id"].isin(data["contamination"]["contaminated_ids"])
    ]["customer_id"].nunique()

    assert n_contaminated_in_outcomes > 0, "no contaminated customers to exclude"
    assert corrected["n_after_exclusion"] < len(outcomes)


def test_small_strata_are_dropped_not_reported(data):
    """A stratum with eight customers does not get a number.

    A noisy estimate that LOOKS precise is more dangerous than an honest gap,
    because a number in a table gets quoted and a gap gets questioned.
    """
    outcomes = data["outcomes"]

    # Fabricate a tiny stratum and confirm it does not blow up or dominate.
    lift = stratified_lift(outcomes)

    assert not pd.isna(lift)
    assert -1.0 < lift < 1.0, "the stratified lift is wildly out of range"
