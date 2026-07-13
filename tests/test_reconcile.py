"""Tests for the reconciliation.

THE STANDARD THESE ENFORCE
--------------------------
The unexplained residual must be ZERO. Every month. To the cent.

Not "small". Not "immaterial". Not "within tolerance for a portfolio project".
Zero.

A residual of $340 sounds harmless right up until someone in the room asks what
it is, and the honest answer is "I do not know" - at which point every other
number you presented becomes suspect, because you have just demonstrated that
your pipeline can lose money without telling you.

An unexplained residual is not a rounding error. It is a defect you have not
found yet, wearing a disguise.

WHY THESE TESTS ARE THE POINT
-----------------------------
Anyone can write a reconciliation that closes once, on the month they happened
to check. These tests close it for all 29 months, and they fail loudly the
moment a future change breaks the logic - which is exactly when you want to
find out, rather than in front of a CFO.
"""

from __future__ import annotations

import pandas as pd
import pytest

from northwind.clean import clean, parse_amount
from northwind.generate import SEED, generate
from northwind.metrics import load_contract
from northwind.reconcile import (
    TOLERANCE,
    reconcile_all_months,
    reconcile_ledger,
    reconcile_month,
)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Generate, clean, and hand back everything the reconciliation needs."""
    raw = tmp_path_factory.mktemp("raw")
    processed = tmp_path_factory.mktemp("processed")

    generate(raw, seed=SEED)
    clean(raw, processed)

    raw_invoices = pd.read_csv(raw / "invoices.csv")
    raw_invoices["amount"] = raw_invoices["amount"].map(parse_amount)

    return {
        "contract": load_contract(),
        "raw_invoices": raw_invoices,
        "invoices": pd.read_parquet(processed / "invoices.parquet"),
        "subs": pd.read_parquet(processed / "subscriptions.parquet"),
        "duplicates": pd.read_parquet(processed / "quarantine_duplicate_invoices.parquet"),
        "orphans": pd.read_parquet(processed / "quarantine_orphan_invoices.parquet"),
    }


@pytest.fixture(scope="module")
def monthly(data):
    """The full monthly bridge, computed once."""
    return reconcile_all_months(data["subs"], data["invoices"], data["contract"])


# ---------------------------------------------------------------------------
# BRIDGE 1 - LEDGER INTEGRITY
# ---------------------------------------------------------------------------


def test_ledger_bridge_closes(data):
    """Raw invoices, minus what we removed, equals the clean invoices. Exactly.

    If this fails, rows are disappearing between raw and clean - and the total
    that comes out the other end will still look completely plausible. That is
    what makes silent row loss the most dangerous bug in analytics.
    """
    bridge = reconcile_ledger(
        data["raw_invoices"], data["invoices"], data["duplicates"], data["orphans"]
    )

    residual = bridge[bridge["line"] == "UNEXPLAINED RESIDUAL"].iloc[0]

    assert abs(residual["dollars"]) < TOLERANCE, (
        f"${residual['dollars']:,.2f} of invoice value is unaccounted for "
        "between the raw export and the clean table"
    )
    assert residual["rows"] == 0, f"{residual['rows']} invoice rows vanished without a name"


def test_quarantined_dollars_are_material(data):
    """The quarantine is not a technicality - it is holding real money.

    $536k of invoices point at customers who do not exist. A LEFT JOIN would
    have erased them and understated revenue with no warning whatsoever. The
    fact that this number is LARGE is precisely why quarantining beats dropping.
    """
    orphan_dollars = float(data["orphans"]["amount"].sum())
    dupe_dollars = float(data["duplicates"]["amount"].sum())

    assert orphan_dollars > 100_000, "orphans should hold material value"
    assert dupe_dollars > 100_000, "duplicates should hold material value"


# ---------------------------------------------------------------------------
# BRIDGE 2 - CONTRACTED TO BILLED
# The tests that actually matter.
# ---------------------------------------------------------------------------


def test_every_month_reconciles_to_zero(monthly):
    """THE TEST. All 29 months tie to the ledger to the cent.

    Closing for one month proves nothing - it could be luck, or two errors
    happening to cancel each other out. It has to close for every month, or the
    logic is not right and we simply have not noticed yet.
    """
    failures = monthly[monthly["residual"].abs() > TOLERANCE]

    assert len(failures) == 0, (
        f"{len(failures)} months do not reconcile:\n"
        f"{failures[['month', 'residual']].to_string(index=False)}\n"
        "An unexplained residual is not a rounding error - it is an undiscovered defect."
    )


def test_the_bridge_covers_the_whole_analysis_window(monthly):
    """We reconciled every month we intend to draw conclusions about.

    Reconciling only the months that happen to tie is not reconciliation.
    """
    assert len(monthly) >= 28, f"only {len(monthly)} months reconciled"

    assert monthly["month"].min() <= pd.Timestamp("2024-03-01")
    assert monthly["month"].max() >= pd.Timestamp("2026-05-01")


def test_bridge_lines_are_all_material(monthly):
    """Every line in the bridge is doing real work.

    A reconciling item that is always zero is not an explanation - it is a
    decoration, and it hides the fact that you never understood the gap.
    """
    latest = monthly.iloc[-1]

    assert latest["less_not_invoiced"] < 0, "the 'not invoiced' line is empty"
    assert latest["plus_invoiced_not_active"] > 0, "the mid-month-start line is empty"
    assert latest["plus_refunds_base"] < 0, "the refunds line is empty"
    assert latest["plus_addons"] > 0, "the add-on line is empty"


def test_contracted_and_billed_genuinely_differ(monthly):
    """The gap is real, not an artefact.

    If contracted MRR equalled billed revenue exactly, there would be nothing to
    reconcile and this milestone would be theatre. The two numbers differ for
    legitimate reasons, and naming those reasons is the entire exercise.
    """
    latest = monthly.iloc[-1]

    gap = latest["billed_base"] - latest["contracted_mrr"]

    assert abs(gap) > 10_000, (
        f"contracted and billed differ by only ${gap:,.2f} - "
        "suspiciously close, check the bridge is actually doing anything"
    )


def test_addons_are_excluded_from_contracted_mrr(data, monthly):
    """RULING R2. Add-ons are billed but never counted as MRR.

    They appear as a bridge line BELOW billed base revenue, never inside
    contracted MRR. If they leaked into MRR, retention would have looked
    artificially healthy at exactly the moment it was collapsing - because the
    September 2025 price rise lifted add-on revenue by ~45% overnight.
    """
    assert data["contract"].addons_count_as_mrr is False

    latest = monthly.iloc[-1]

    # billed_total = billed_base + add-ons. If add-ons had been folded into MRR,
    # this identity would break.
    assert latest["billed_total"] == pytest.approx(
        latest["billed_base"] + latest["plus_addons"], abs=1.0
    )


def test_refunds_land_in_the_month_they_belong_to(data):
    """RULING R5. A refund reduces the month of the invoice it reverses.

    Booked in the month it was ISSUED instead, the damage from the September
    2025 price rise would have smeared forward into later months - flattering
    the period the CFO is asking about and hiding the trend entirely.
    """
    inv = data["invoices"].copy()
    inv["effective_date"] = pd.to_datetime(inv["effective_date"])
    inv["issued_date"] = pd.to_datetime(inv["issued_date"])

    refunds = inv[inv["reverses_invoice_id"].notna()]

    # Every refund must be booked no later than the day it was issued.
    assert (refunds["effective_date"] <= refunds["issued_date"]).all()

    # And a meaningful number must actually have MOVED, or R5 is a no-op.
    moved = (refunds["effective_date"] < refunds["issued_date"]).sum()
    assert moved > 100, f"only {moved} refunds were restated - R5 is barely doing anything"


# ---------------------------------------------------------------------------
# ROBUSTNESS
# ---------------------------------------------------------------------------


def test_a_single_month_reconciles_in_isolation(data):
    """The bridge is not relying on some accident of the full date range."""
    subs = data["subs"].copy()
    subs["period_start"] = pd.to_datetime(subs["period_start"])
    subs["period_end"] = pd.to_datetime(subs["period_end"]).fillna(pd.Timestamp("2026-06-30"))

    inv = data["invoices"].copy()
    inv["effective_date"] = pd.to_datetime(inv["effective_date"])

    result = reconcile_month(subs, inv, pd.Timestamp("2025-10-01"), data["contract"])

    assert abs(result["residual"]) < TOLERANCE


def test_reconciliation_survives_the_shock_months(monthly):
    """The three planted shocks do not break the bridge.

    July 2025 (sales reorg), September 2025 (price rise), October 2025
    (competitor launch). These are the months where churn, downgrades, and
    refunds all spike at once - precisely where a fragile reconciliation would
    fall apart, and precisely the months the CFO will scrutinise hardest.
    """
    shock_months = [
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-09-01"),
        pd.Timestamp("2025-10-01"),
    ]

    for month in shock_months:
        row = monthly[monthly["month"] == month]
        assert len(row) == 1, f"{month:%Y-%m} missing from the bridge"
        assert abs(float(row["residual"].iloc[0])) < TOLERANCE, (
            f"{month:%Y-%m} does not reconcile - and it is one of the months "
            "the CFO will look at first"
        )
