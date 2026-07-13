"""Tests for the NRR decomposition.

WHAT THESE PROTECT
------------------
The decomposition is the first module allowed to draw a conclusion. Everything
before it was plumbing. So these tests guard two things:

1. THE ARITHMETIC IS EXACT. NRR splits into churn, contraction and expansion
   with NO residual. Not approximately - exactly. If a residual appears, some
   dollar has been double-counted or lost, and the conclusion is worthless.

2. THE HEADLINE FINDING IS REAL. Churn IMPROVED while NRR collapsed. That is
   counterintuitive, it is the whole story, and it is the thing an interviewer
   will push on. It needs a test, so that if a future change quietly breaks it,
   we find out from CI rather than from a hiring manager.
"""

from __future__ import annotations

import pandas as pd
import pytest

from northwind.clean import clean
from northwind.decompose import (
    attribute_decline,
    build_mrr_panel,
    decompose_all_months,
    decompose_by_group,
    decompose_month,
    monthly_movement,
)
from northwind.generate import SEED, generate


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    processed = tmp_path_factory.mktemp("processed")

    generate(raw, seed=SEED)
    clean(raw, processed)

    subs = pd.read_parquet(processed / "subscriptions.parquet")
    customers = pd.read_parquet(processed / "customers.parquet")
    panel = build_mrr_panel(subs)

    return {
        "subs": subs,
        "customers": customers,
        "panel": panel,
        "movement": decompose_all_months(panel),
        "mom": monthly_movement(panel),
    }


# ---------------------------------------------------------------------------
# 1. THE ARITHMETIC IS EXACT
# ---------------------------------------------------------------------------


def test_the_movement_bridge_closes_exactly(data):
    """starting - churn - contraction + expansion = ending. Every month. To the cent.

    This is not a modelling assumption. It is arithmetic. If it fails, a dollar
    has been double-counted or lost somewhere in the split - and every
    conclusion built on top of it is unsafe.
    """
    for row in data["movement"].itertuples(index=False):
        expected = (
            row.starting_mrr + row.churned_mrr + row.contraction_mrr + row.expansion_mrr
        )
        assert row.ending_mrr == pytest.approx(expected, abs=0.02), (
            f"{row.month:%Y-%m} does not close: "
            f"ending {row.ending_mrr:,.2f} vs bridge {expected:,.2f}"
        )


def test_the_rates_sum_to_nrr(data):
    """NRR = 1 - churn_rate - contraction_rate + expansion_rate. Identically."""
    for row in data["movement"].itertuples(index=False):
        reconstructed = 1 - row.churn_rate - row.contraction_rate + row.expansion_rate

        assert row.nrr == pytest.approx(reconstructed, abs=0.001), (
            f"{row.month:%Y-%m}: NRR {row.nrr:.4f} does not equal the sum of its parts"
        )


def test_attribution_has_no_residual(data):
    """The three drivers account for 100% of the decline. Nothing left over.

    Because NRR is an exact identity, the change in NRR is EXACTLY the sum of
    the changes in its three components. Nobody can argue with the split, which
    is precisely why it belongs in the memo.
    """
    movement = data["movement"]

    healthy = movement.loc[movement["nrr"].idxmax(), "month"]
    damaged = movement["month"].max()

    table = attribute_decline(movement, healthy, damaged)

    a = movement[movement["month"] == healthy].iloc[0]
    b = movement[movement["month"] == damaged].iloc[0]

    total_decline = b["nrr"] - a["nrr"]

    assert table["nrr_impact"].sum() == pytest.approx(total_decline, abs=0.002), (
        "the three drivers do not add up to the total NRR change - "
        "there is an unexplained residual in the attribution"
    )
    assert table["share_of_decline"].sum() == pytest.approx(1.0, abs=0.01)


def test_grr_never_exceeds_one_hundred_percent(data):
    """GRR ignores expansion, so it is the honest floor. It cannot exceed 100%.

    If it does, expansion has leaked into a metric that is defined to exclude it,
    and the 'floor' is no longer a floor.
    """
    assert (data["movement"]["grr"] <= 1.0).all(), "GRR exceeded 100% - expansion leaked in"


def test_grr_is_always_below_nrr(data):
    """NRR includes expansion; GRR does not. NRR must therefore be the larger."""
    m = data["movement"]
    assert (m["nrr"] >= m["grr"]).all()


# ---------------------------------------------------------------------------
# 2. THE HEADLINE FINDING
# The counterintuitive one. This is the story.
# ---------------------------------------------------------------------------


def test_nrr_actually_collapsed(data):
    """The thing the CFO is asking about genuinely happened."""
    movement = data["movement"]

    peak = movement["nrr"].max()
    latest = movement.iloc[-1]["nrr"]

    assert peak > 1.04, f"NRR never got healthy - peaked at {peak:.1%}"
    assert latest < 0.97, f"NRR did not collapse - ended at {latest:.1%}"
    assert (peak - latest) > 0.08, "the decline is too small to be the story"


def test_churn_did_not_cause_the_decline(data):
    """THE FINDING. Churn IMPROVED while NRR collapsed.

    This is the whole project in one assertion, and it explains everything else:

      - Sales watch LOGO CHURN. Logo churn got better. So Sales reported that
        things were fine, and they were not lying.
      - Finance watch DOLLARS. Dollars were haemorrhaging. So Finance reported
        a crisis, and they were not lying either.
      - Both were right. Nobody had written down that NRR has three moving parts
        and only one of them was being watched.

    An analyst who computes churn, sees it improving, and reports "retention is
    healthy" gets this exactly backwards - and would have told the board to do
    nothing while the company bled out.
    """
    movement = data["movement"]

    healthy = movement.loc[movement["nrr"].idxmax()]
    damaged = movement.iloc[-1]

    assert damaged["churn_rate"] <= healthy["churn_rate"] + 0.01, (
        f"churn rose from {healthy['churn_rate']:.1%} to {damaged['churn_rate']:.1%} - "
        "the counterintuitive finding no longer holds, and the story has changed"
    )


def test_expansion_collapse_is_the_largest_driver(data):
    """Lost upsell, not lost customers, is what killed NRR.

    This is what turns a chart into an instruction. 'NRR is 94%' tells a VP
    nothing. 'Your upsell motion stopped working in July and that is 87% of the
    damage' tells them exactly which meeting to hold on Monday.
    """
    movement = data["movement"]

    healthy = movement.loc[movement["nrr"].idxmax(), "month"]
    damaged = movement["month"].max()

    table = attribute_decline(movement, healthy, damaged).set_index("driver")

    expansion_hit = table.loc["Expansion (upsell)", "nrr_impact"]
    contraction_hit = table.loc["Contraction (downgrades)", "nrr_impact"]
    churn_hit = table.loc["Churn (customers who left)", "nrr_impact"]

    assert expansion_hit < contraction_hit, "expansion is not the biggest driver"
    assert expansion_hit < churn_hit, "expansion is not the biggest driver"
    assert expansion_hit < -0.05, (
        f"the expansion collapse is only {expansion_hit:.1%} - too small to be the story"
    )


def test_contraction_is_the_second_driver_and_is_not_churn(data):
    """RULING R3. Downgrades are contraction, and they matter.

    Had we counted downgrades as churn - as Sales wanted - this entire driver
    would have been swallowed into the churn number and become invisible. The
    ruling is what kept it findable.
    """
    movement = data["movement"]

    healthy = movement.loc[movement["nrr"].idxmax()]
    damaged = movement.iloc[-1]

    assert damaged["contraction_rate"] > healthy["contraction_rate"] * 1.5, (
        "contraction did not materially worsen - check R3 is being applied"
    )


# ---------------------------------------------------------------------------
# 3. WHO AND WHEN
# ---------------------------------------------------------------------------


def test_the_damage_is_concentrated_in_smb(data):
    """Not every segment is bleeding. SMB is.

    A finding that applies to everyone equally is not a finding, it is a
    background condition. The value is in the asymmetry.
    """
    by_segment = decompose_by_group(
        data["panel"], data["customers"], pd.Timestamp("2026-06-01"), "segment"
    ).set_index("segment")

    assert by_segment.loc["SMB", "nrr"] < by_segment.loc["Enterprise", "nrr"], (
        "SMB is not worse than Enterprise - the segment story does not hold"
    )
    assert by_segment.loc["SMB", "mrr_lost"] > 0, "SMB is not losing money"


def test_the_month_over_month_view_finds_the_expansion_collapse(data):
    """A trailing-12-month metric smears a shock across a year. This does not.

    NRR is a good headline and a terrible detector. To put a DATE on the wound,
    you compare month against month - and then the collapse is unmistakable.
    """
    mom = data["mom"]

    before = mom[
        (mom["month"] >= pd.Timestamp("2025-04-01")) & (mom["month"] < pd.Timestamp("2025-08-01"))
    ]["expansion_rate"].mean()

    after = mom[mom["month"] >= pd.Timestamp("2025-11-01")]["expansion_rate"].mean()

    assert after < before * 0.6, (
        f"monthly expansion only fell from {before:.1%} to {after:.1%} - "
        "the collapse is not visible in the month-over-month view"
    )


def test_small_cohorts_are_excluded(data):
    """We refuse to report a rate built on a denominator of almost nothing.

    Early 2025 compares against a 2024 cohort of a few dozen customers. A 130%
    NRR on eleven customers is a small-sample artefact, not a finding, and
    publishing it is how a number gets quietly ignored forever.
    """
    assert (data["movement"]["cohort_customers"] >= 500).all()


def test_a_single_month_decomposes_correctly(data):
    """The bridge works in isolation, not just as part of the full run."""
    result = decompose_month(data["panel"], pd.Timestamp("2026-06-01"))

    assert result["starting_mrr"] > 0
    assert result["churned_mrr"] < 0        # losses are negative
    assert result["contraction_mrr"] < 0    # losses are negative
    assert result["expansion_mrr"] > 0      # gains are positive
    assert 0.5 < result["nrr"] < 1.5        # sane range
