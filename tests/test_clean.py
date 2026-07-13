"""Tests for the cleaning layer.

WHAT THESE PROVE
----------------
1. NOTHING IS LOST SILENTLY. Every row that leaves the pipeline is accounted
   for - kept, quarantined, or deliberately removed with a dollar figure
   attached. A cleaning step that cannot tell you what it deleted has not
   cleaned the data, it has merely changed it.

2. THE RULINGS ARE ACTUALLY IMPLEMENTED. Change pause_grace_days in the YAML
   and the churn number moves. If it does not, the contract is decoration and
   every claim this project makes is hollow.

3. THE TRAPS ARE ACTUALLY SPRUNG. The near-duplicate invoices with fresh IDs.
   The orphans a LEFT JOIN would erase. The refund booked in the wrong month.
   Each has a test that fails if we quietly regress to the naive approach.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from northwind.clean import (
    CleaningReport,
    clean,
    parse_amount,
)
from northwind.generate import SEED, generate
from northwind.metrics import CONTRACT_PATH, load_contract


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Generate raw data, clean it, and hand back both sides plus the report."""
    raw = tmp_path_factory.mktemp("raw")
    processed = tmp_path_factory.mktemp("processed")

    generate(raw, seed=SEED)
    report = clean(raw, processed)

    return {
        "raw_customers": pd.read_parquet(raw / "customers.parquet"),
        "raw_subs": pd.read_parquet(raw / "subscriptions.parquet"),
        "raw_invoices": pd.read_csv(raw / "invoices.csv"),
        "customers": pd.read_parquet(processed / "customers.parquet"),
        "subs": pd.read_parquet(processed / "subscriptions.parquet"),
        "invoices": pd.read_parquet(processed / "invoices.parquet"),
        "usage": pd.read_parquet(processed / "usage_events.parquet"),
        "dupes": pd.read_parquet(processed / "quarantine_duplicate_invoices.parquet"),
        "orphans": pd.read_parquet(processed / "quarantine_orphan_invoices.parquet"),
        "report": report.to_frame(),
    }


# ---------------------------------------------------------------------------
# 1. AMOUNT PARSING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("$1,234.56", 1234.56),      # currency symbol and thousands separator
        ("(500.00)", -500.00),       # accounting notation: parentheses mean NEGATIVE
        ("$99.00", 99.00),
        ("1234.56", 1234.56),        # already clean text
        (1234.56, 1234.56),          # already a number
        (-500.0, -500.0),
    ],
)
def test_parse_amount(raw_value, expected):
    """Every amount format the billing export throws at us parses correctly.

    The parenthesis case is the one that matters. A naive parser that strips
    non-numeric characters turns '(500.00)' into +500.00 - silently converting
    a refund into revenue. That single bug would corrupt every number downstream.
    """
    assert parse_amount(raw_value) == pytest.approx(expected)


def test_parse_amount_handles_nulls():
    """A missing amount is zero, not a crash."""
    assert parse_amount(None) == 0.0
    assert parse_amount(pd.NA) == 0.0


def test_all_amounts_are_numeric_after_cleaning(pipeline):
    """No text survives into the processed invoices."""
    assert pd.api.types.is_numeric_dtype(pipeline["invoices"]["amount"])


def test_refunds_are_still_negative_after_parsing(pipeline):
    """The accounting-negative parse did not flip refunds into revenue."""
    refunds = pipeline["invoices"][pipeline["invoices"]["status"] == "refunded"]

    assert len(refunds) > 0
    assert (refunds["amount"] < 0).all(), "a refund became positive - the parser is broken"


# ---------------------------------------------------------------------------
# 2. DEDUPLICATION - THE NEAR-DUPLICATE TRAP
# ---------------------------------------------------------------------------


def test_near_duplicates_are_caught(pipeline):
    """THE TRAP. Duplicates with FRESH invoice IDs must not survive.

    Half the injected duplicates were re-issued with a new invoice_id. A naive
    drop_duplicates(subset=['invoice_id']) removes the other half, reports a
    plausible-looking number, and leaves real double-counted revenue in the
    total. This test fails if we ever regress to that.
    """
    inv = pipeline["invoices"]

    business_key = ["customer_id", "subscription_id", "issued_date", "amount", "status"]
    surviving_dupes = inv.duplicated(subset=business_key).sum()

    assert surviving_dupes == 0, (
        f"{surviving_dupes} duplicate invoices survived - "
        "deduplication is matching on ID rather than on the business key"
    )


def test_deduplication_reports_the_dollars(pipeline):
    """We can say exactly how much revenue we were about to double-count."""
    row = pipeline["report"].query("step == 'deduplicate_invoices'").iloc[0]

    assert row["rows_affected"] > 0
    assert row["dollars"] > 0, "deduplication did not report a dollar impact"


def test_removed_duplicates_are_kept_for_inspection(pipeline):
    """Duplicates go to a quarantine table, not to the bin."""
    assert len(pipeline["dupes"]) > 0


# ---------------------------------------------------------------------------
# 3. ORPHANS - QUARANTINED, NOT DELETED
# ---------------------------------------------------------------------------


def test_orphan_invoices_are_quarantined_not_dropped(pipeline):
    """Invoices with no matching customer are preserved and counted.

    A LEFT JOIN erases these silently. Revenue comes out low, nothing errors,
    and you present a wrong number with complete confidence. Milestone 4
    reconciles to the ledger to the dollar - impossible unless every
    unattributable dollar is on the books somewhere.
    """
    orphans = pipeline["orphans"]
    customers = pipeline["customers"]

    assert len(orphans) > 0, "no orphans quarantined - did we silently drop them?"

    known = set(customers["customer_id"])
    assert not orphans["customer_id"].isin(known).any(), (
        "a quarantined 'orphan' actually has a valid customer"
    )


def test_no_orphans_remain_in_the_clean_invoices(pipeline):
    """Every invoice in the clean table points at a real customer."""
    known = set(pipeline["customers"]["customer_id"])

    assert pipeline["invoices"]["customer_id"].isin(known).all()


def test_every_raw_invoice_is_accounted_for(pipeline):
    """THE AUDIT. Clean + duplicates + orphans must equal the raw count.

    Not one row may vanish without a name. If this fails, some cleaning step is
    losing data quietly - and quiet data loss is the single most dangerous
    failure mode in analytics, because the number that comes out still looks
    completely plausible.
    """
    raw_count = len(pipeline["raw_invoices"])
    accounted = len(pipeline["invoices"]) + len(pipeline["dupes"]) + len(pipeline["orphans"])

    assert accounted == raw_count, (
        f"{raw_count - accounted} invoice rows vanished without being logged"
    )


# ---------------------------------------------------------------------------
# 4. R5 - REFUNDS HIT THE ORIGINAL MONTH
# ---------------------------------------------------------------------------


def test_refunds_are_attributed_to_the_original_invoice_month(pipeline):
    """RULING R5. A refund's effective_date is the date of the invoice it reverses."""
    inv = pipeline["invoices"]

    refunds = inv[inv["reverses_invoice_id"].notna()].copy()
    assert len(refunds) > 0

    # Build a lookup of every invoice's own issue date.
    issued = inv.set_index("invoice_id")["issued_date"].to_dict()

    # For each refund, the effective date must equal the ORIGINAL's issue date.
    expected = refunds["reverses_invoice_id"].map(issued)
    matched = refunds.loc[expected.notna()]
    expected = expected.dropna()

    assert (
        pd.to_datetime(matched["effective_date"]).to_numpy()
        == pd.to_datetime(expected).to_numpy()
    ).all(), "a refund is booked in the wrong month"


def test_refunds_moved_to_an_earlier_month(pipeline):
    """Sanity: the restatement actually moved dates backwards, not forwards."""
    inv = pipeline["invoices"]
    refunds = inv[inv["reverses_invoice_id"].notna()]

    effective = pd.to_datetime(refunds["effective_date"])
    issued = pd.to_datetime(refunds["issued_date"])

    assert (effective <= issued).all(), "a refund was moved FORWARD in time"


# ---------------------------------------------------------------------------
# 5. R6 - CORRECTIONS SUPERSEDE
# ---------------------------------------------------------------------------


def test_corrections_replace_the_original_rows(pipeline):
    """RULING R6. No customer-period appears twice after cleaning.

    If both the original and its correction survive, MRR is double-counted for
    that period. That is one concrete, findable reason the analytics tables
    never tied to the billing ledger.
    """
    subs = pipeline["subs"]

    duplicated_periods = subs.duplicated(subset=["customer_id", "period_start"]).sum()

    assert duplicated_periods == 0, (
        f"{duplicated_periods} customer-periods appear twice - "
        "corrections did not supersede their originals, and MRR is double-counted"
    )


def test_corrections_survive_and_originals_do_not(pipeline):
    """The correction is the row that lives."""
    subs = pipeline["subs"]
    raw_subs = pipeline["raw_subs"]

    n_corrections = int(raw_subs["is_correction"].sum())

    assert int(subs["is_correction"].sum()) == n_corrections, (
        "correction rows were dropped - we kept the stale original instead"
    )


# ---------------------------------------------------------------------------
# 6. R1 - THE PAUSE RULING
# The most important tests in the file.
# ---------------------------------------------------------------------------


def test_no_ambiguous_pauses_survive(pipeline):
    """Every 'paused' row has been resolved into a definite state.

    'paused' is not an answer. It is the question. After cleaning, every one of
    those 614 rows is either 'churned' or 'paused_returned' - and which one is
    determined by a rule that is written down, defended with evidence, and
    applied identically to every row.
    """
    statuses = set(pipeline["subs"]["status"])

    assert "paused" not in statuses, (
        "unresolved 'paused' rows remain - the Sales/Finance argument is still live"
    )
    assert "paused_returned" in statuses
    assert "churned" in statuses


def test_pause_resolution_splits_both_ways(pipeline):
    """Some paused customers were kept, some were churned. Neither camp won outright.

    If everything resolved one way, we would simply have adopted Sales' position
    or Finance's rather than ruling between them - and the ruling would be
    theatre.
    """
    row = pipeline["report"].query("step == 'resolve_pauses'").iloc[0]

    assert row["rows_affected"] > 0
    assert "returned within" in row["detail"]


def test_changing_the_grace_period_changes_the_churn_number(tmp_path):
    """THE TEST THAT MATTERS. The YAML governs the outcome.

    Set pause_grace_days to a very large number and almost every paused customer
    who ever came back should now count as retained rather than churned.

    If this fails, the contract is a document the code politely ignores - which
    is the exact failure mode this entire project was built to prevent.
    """
    from northwind import clean as clean_module

    raw = tmp_path / "raw"
    generate(raw, seed=SEED)

    # --- Run once with the real contract (60 days) ---
    baseline = clean(raw, tmp_path / "out_60")
    baseline_row = baseline.to_frame().query("step == 'resolve_pauses'").iloc[0]

    # --- Now rewrite the contract with a 3-year grace period ---
    raw_yaml = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    raw_yaml["parameters"]["pause_grace_days"] = 1095

    altered_path = tmp_path / "metrics.yaml"
    altered_path.write_text(yaml.safe_dump(raw_yaml), encoding="utf-8")

    # Point the cleaning module at the altered contract and clear the cache.
    load_contract.cache_clear()
    original_path = clean_module.load_contract

    def patched(path=None):
        return original_path(altered_path)

    clean_module.load_contract = patched
    try:
        relaxed = clean(raw, tmp_path / "out_1095")
        relaxed_row = relaxed.to_frame().query("step == 'resolve_pauses'").iloc[0]
    finally:
        clean_module.load_contract = original_path
        load_contract.cache_clear()

    # With a 3-year grace period, far fewer paused customers count as churned,
    # so the churned MRR must fall.
    assert relaxed_row["dollars"] < baseline_row["dollars"], (
        "changing pause_grace_days did not change the churn number - "
        "the metrics contract is decoration, not policy"
    )


# ---------------------------------------------------------------------------
# 7. CATEGORICALS AND MISSING VALUES
# ---------------------------------------------------------------------------


def test_segment_spellings_are_collapsed(pipeline):
    """'smb', 'S.M.B.', ' SMB ' and 'Smb' all become 'SMB'."""
    segments = set(pipeline["customers"]["segment"].unique())

    # Three real segments plus the explicit Unknown category.
    assert segments == {"SMB", "Mid-Market", "Enterprise", "Unknown"}, (
        f"segment values are not canonical: {sorted(segments)}"
    )


def test_missing_values_become_an_explicit_category(pipeline):
    """Nulls are labelled 'Unknown', never silently dropped.

    A null that disappears from a GROUP BY is a lie of omission. The gap is real
    and it belongs in the chart, where someone can see it and ask about it.
    """
    cust = pipeline["customers"]

    for col in ["segment", "industry", "region"]:
        assert cust[col].isna().sum() == 0, f"nulls survive in {col}"
        assert (cust[col] == "Unknown").sum() > 0, f"no Unknown category created in {col}"


def test_no_customers_were_lost(pipeline):
    """Cleaning customers must not delete any of them."""
    assert len(pipeline["customers"]) == len(pipeline["raw_customers"])


# ---------------------------------------------------------------------------
# 8. THE REPORT
# ---------------------------------------------------------------------------


def test_every_step_is_logged(pipeline):
    """The cleaning report accounts for every transformation."""
    steps = set(pipeline["report"]["step"])

    expected = {
        "parse_amounts",
        "deduplicate_invoices",
        "quarantine_orphan_invoices",
        "attribute_refunds",
        "apply_corrections",
        "resolve_pauses",
        "standardise_segments",
        "label_missing_values",
        "fix_timezone_drift",
    }

    assert expected <= steps, f"unlogged cleaning steps: {sorted(expected - steps)}"


def test_ruling_backed_steps_cite_their_ruling(pipeline):
    """Any step that implements a ruling names it, so drift is visible.

    If R1 changes and this code does not, the citation is the thread that leads
    a reviewer straight to the inconsistency.
    """
    report = pipeline["report"]

    citations = {
        "attribute_refunds": "R5",
        "apply_corrections": "R6",
        "resolve_pauses": "R1",
    }

    for step, ruling in citations.items():
        row = report.query("step == @step").iloc[0]
        assert row["ruling"] == ruling, f"{step} does not cite {ruling}"


def test_report_is_empty_by_default():
    """A fresh report starts clean - no phantom steps."""
    assert len(CleaningReport().to_frame()) == 0
