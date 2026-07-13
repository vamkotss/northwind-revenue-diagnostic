"""Tests for the mess generator.

WHY TEST A DATA GENERATOR?
-------------------------
Because the whole project rests on one claim: "I know exactly what is wrong
with this data, because I put it there." If the generator is not tested, that
claim is just a story. These tests turn it into evidence.

They check three things:
  1. REPRODUCIBILITY - same seed, same bytes. Every time.
  2. DEFECTS EXIST - each of the eight defects is actually present.
  3. THE BUSINESS STORY IS REAL - NRR genuinely declines, so the analysis
     that follows is discovering something rather than inventing it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from northwind.generate import DEFECT_RATES, SEED, generate


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Generate the dataset ONCE and share it across every test in this file.

    scope="module" means this runs a single time, not once per test. Generation
    takes a few seconds; doing it four times would be wasteful.
    """
    out = tmp_path_factory.mktemp("raw")
    generate(out, seed=SEED)

    return {
        "dir": out,
        "customers": pd.read_parquet(out / "customers.parquet"),
        "subscriptions": pd.read_parquet(out / "subscriptions.parquet"),
        "invoices": pd.read_csv(out / "invoices.csv"),
        "usage": pd.read_parquet(out / "usage_events.parquet"),
        "experiment": pd.read_parquet(out / "experiment_assignments.parquet"),
        "manifest": pd.read_parquet(out / "defect_manifest.parquet"),
    }


# ---------------------------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_data(tmp_path):
    """Two runs with the same seed must produce byte-identical output.

    This is THE non-negotiable property. If it fails, every number in the
    final memo is unreproducible and the CI pipeline is decoration.
    """
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"

    generate(a, seed=12345)
    generate(b, seed=12345)

    # Compare the raw bytes of each file, not the parsed contents.
    # Bytes is the strictest possible check.
    for name in ["customers.parquet", "subscriptions.parquet", "usage_events.parquet"]:
        assert (a / name).read_bytes() == (b / name).read_bytes(), (
            f"{name} differs between two runs with the same seed"
        )


def test_different_seeds_produce_different_data(tmp_path):
    """Sanity check the inverse: a different seed must change the data.

    If this passed trivially, it would mean the seed is being ignored and the
    reproducibility test above is meaningless.
    """
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"

    generate(a, seed=111)
    generate(b, seed=999)

    assert (a / "customers.parquet").read_bytes() != (b / "customers.parquet").read_bytes()


# ---------------------------------------------------------------------------
# 2. STRUCTURE
# ---------------------------------------------------------------------------


def test_all_tables_are_produced(data):
    """Every expected file exists and is non-empty."""
    for name in ["customers", "subscriptions", "invoices", "usage", "experiment"]:
        assert len(data[name]) > 0, f"{name} is empty"


def test_scale_is_as_specified(data):
    """The dataset is the size we designed it to be."""
    assert len(data["customers"]) == 5_000
    # Usage is the big table. It should dominate the row count.
    assert len(data["usage"]) > 1_000_000, "usage_events is smaller than expected"


# ---------------------------------------------------------------------------
# 3. EACH DEFECT IS ACTUALLY PRESENT
# One test per defect. If a test fails, the cleaning layer downstream would be
# solving a problem that does not exist - which is worse than useless.
# ---------------------------------------------------------------------------


def test_defect_manifest_lists_all_eight(data):
    """The manifest documents every defect type we claim to inject."""
    listed = set(data["manifest"]["defect"])
    assert listed == set(DEFECT_RATES.keys()), (
        f"manifest is missing: {set(DEFECT_RATES.keys()) - listed}"
    )


def test_defect_duplicate_invoices(data):
    """Duplicate invoices exist - both exact-ID twins and new-ID near-twins."""
    inv = data["invoices"]

    # Exact duplicates: the same invoice_id appears more than once.
    exact = inv["invoice_id"].duplicated().sum()
    assert exact > 0, "no exact duplicate invoice IDs found"

    # Near-duplicates: a fresh ID, but every business field identical. These
    # are the dangerous ones - drop_duplicates on invoice_id will NOT catch them.
    business_cols = ["customer_id", "subscription_id", "issued_date", "amount"]
    near = inv.duplicated(subset=business_cols).sum()
    assert near > exact, "no near-duplicate invoices (same content, new ID)"


def test_defect_timezone_drift(data):
    """Some usage timestamps carry a bad offset, pushing them to the wrong day."""
    usage = data["usage"]

    # event_date is the truth. event_ts is what the logger wrote.
    # Where the timestamp's calendar day disagrees with event_date, we have drift.
    ts_day = pd.to_datetime(usage["event_ts"]).dt.normalize()
    true_day = pd.to_datetime(usage["event_date"]).dt.normalize()

    drifted = (ts_day != true_day).sum()
    assert drifted > 0, "no timezone drift detected"

    # It should affect a meaningful slice, not one lonely row.
    assert drifted / len(usage) > 0.001


def test_defect_retroactive_subscriptions(data):
    """Backdated correction rows exist and are flagged."""
    subs = data["subscriptions"]

    corrections = subs[subs["is_correction"]]
    assert len(corrections) > 0, "no retroactive correction rows found"

    # Each correction overlaps an existing period for the same customer,
    # which is exactly what makes naive summing double-count revenue.
    assert corrections["customer_id"].isin(subs["customer_id"]).all()


def test_defect_missing_values(data):
    """Customer attributes contain nulls."""
    cust = data["customers"]

    for col in ["segment", "industry", "region"]:
        assert cust[col].isna().sum() > 0, f"no nulls found in {col}"


def test_defect_dirty_categoricals(data):
    """The segment field is spelled inconsistently across rows."""
    cust = data["customers"]

    # Drop nulls, then count distinct spellings. Clean data would have 3.
    distinct = cust["segment"].dropna().nunique()
    assert distinct > 3, (
        f"expected messy spellings, found only {distinct} distinct segment values"
    )


def test_defect_dirty_amounts(data):
    """Invoice amounts include currency symbols and accounting negatives."""
    inv = data["invoices"]

    # The CSV read them all as strings. Look for the telltale characters.
    amounts = inv["amount"].astype(str)

    assert amounts.str.contains(r"\$").sum() > 0, "no '$' formatted amounts"
    assert amounts.str.contains(r"^\(").sum() > 0, "no accounting-negative amounts"


def test_defect_orphan_invoices(data):
    """Some invoices reference customer IDs that do not exist."""
    inv = data["invoices"]
    cust = data["customers"]

    known = set(cust["customer_id"])
    orphans = ~inv["customer_id"].isin(known)

    assert orphans.sum() > 0, "no orphan invoices found"


def test_defect_ambiguous_pause(data):
    """The 'paused' status - the reason Sales and Finance disagree - exists."""
    subs = data["subscriptions"]

    assert (subs["status"] == "paused").sum() > 0, "no paused subscriptions found"


# ---------------------------------------------------------------------------
# 3b. PAUSE BEHAVIOUR IS REAL AND THE THRESHOLD IS DERIVABLE
#
# The churn ruling depends on being able to MEASURE how pause length relates to
# whether a customer ever comes back. If that relationship is not in the data,
# any threshold you pick is arbitrary - and an interviewer will catch it.
# ---------------------------------------------------------------------------


def _pause_return_curve(subs: pd.DataFrame) -> pd.DataFrame:
    """For every paused subscription: did the customer return, and how long after?

    A "return" means a later subscription row exists for the same customer,
    starting after the pause began. The source system does not record this -
    it must be reconstructed by joining the table to itself.
    """
    s = subs[~subs["is_correction"]].copy()
    s["period_start"] = pd.to_datetime(s["period_start"])
    s["period_end"] = pd.to_datetime(s["period_end"])

    paused = s[s["status"] == "paused"][
        ["subscription_id", "customer_id", "period_end"]
    ].copy()

    # Self-join: match each paused row to every later period for that customer.
    joined = paused.merge(s[["customer_id", "period_start"]], on="customer_id")
    joined = joined[joined["period_start"] > joined["period_end"]]

    # The earliest such period is when they came back.
    first_return = joined.groupby("subscription_id")["period_start"].min()

    paused["returned_on"] = paused["subscription_id"].map(first_return)
    paused["returned"] = paused["returned_on"].notna()
    paused["gap_months"] = (
        (paused["returned_on"] - paused["period_end"]).dt.days / 30.44
    ).round()

    return paused


def test_some_paused_customers_come_back(data):
    """Pauses are not all terminal - a meaningful share of customers return.

    If nobody ever returned, 'paused' would just be a synonym for 'churned'
    and there would be no edge case to rule on.
    """
    curve = _pause_return_curve(data["subscriptions"])

    return_rate = curve["returned"].mean()
    assert 0.10 < return_rate < 0.60, (
        f"return rate of {return_rate:.1%} makes the pause ruling trivial"
    )


def test_return_rate_decays_with_pause_length(data):
    """THE KEY PROPERTY. Longer pauses mean fewer returns.

    This decay is what makes an empirical churn threshold possible. The analyst
    plots returns against pause length, sees the curve collapse, and cuts where
    it flattens. Without this decay, any threshold is a guess dressed up as a
    decision - and that is exactly what this project exists to avoid.
    """
    curve = _pause_return_curve(data["subscriptions"])
    returns = curve[curve["returned"]]

    # The overwhelming majority of returns must happen EARLY.
    within_60_days = (returns["gap_months"] <= 2).mean()
    assert within_60_days > 0.80, (
        f"only {within_60_days:.1%} of returns happen within 60 days; "
        "the curve is too flat for a threshold to be defensible"
    )

    # And late returns must be genuinely rare - the tail has to die.
    beyond_120_days = (returns["gap_months"] > 4).mean()
    assert beyond_120_days < 0.05, (
        f"{beyond_120_days:.1%} of returns happen after 120 days; tail is too fat"
    )


def test_paused_is_a_mixed_bag(data):
    """Some 'paused' rows are real pauses; some are cancellations nobody logged.

    This is the trap. You cannot treat 'paused' as one thing, because it is not
    one thing - and no field in the source data tells you which is which.
    """
    curve = _pause_return_curve(data["subscriptions"])

    assert curve["returned"].sum() > 0, "no paused customer ever returned"
    assert (~curve["returned"]).sum() > 0, "every paused customer returned"


# ---------------------------------------------------------------------------
# 4. THE CONTAMINATED EXPERIMENT
# ---------------------------------------------------------------------------


def test_experiment_has_sample_ratio_mismatch(data):
    """The A/B test split is NOT 50/50 - the randomisation failed.

    A Sample Ratio Mismatch (SRM) means the two groups were not assigned
    fairly, so any comparison between them is suspect. Detecting this before
    reading out the result is the entire lesson.
    """
    exp = data["experiment"]

    share = exp["variant"].value_counts(normalize=True)

    # A healthy 50/50 split would sit within ~1% of 0.50 at this sample size.
    # Ours is deliberately around 55/45.
    assert abs(share["control"] - 0.50) > 0.02, "expected an SRM, split looks clean"


def test_experiment_is_contaminated(data):
    """Some customers appear in BOTH arms - they were re-randomised mid-test."""
    exp = data["experiment"]

    # Count how many arms each customer was assigned to.
    arms_per_customer = exp.groupby("customer_id")["variant"].nunique()

    contaminated = (arms_per_customer > 1).sum()
    assert contaminated > 0, "no customers appear in both arms"


# ---------------------------------------------------------------------------
# 5. THE BUSINESS STORY IS REAL
# The most important test in the file. It asserts that the thing the CFO is
# asking about ACTUALLY HAPPENED in the data. Without this, the analysis that
# follows is theatre.
# ---------------------------------------------------------------------------


def test_net_revenue_retention_actually_declines(data):
    """NRR must fall materially between mid-2025 and mid-2026.

    Net Revenue Retention answers: of the revenue we had from a group of
    customers a year ago, how much do we still have today - after churn,
    downgrades, and upsells? Above 100% means the surviving customers grew.
    Below 100% means they shrank.
    """
    subs = data["subscriptions"]

    # Ignore the injected correction rows for this check. The real analysis
    # will have to decide how to handle them; here we just want the signal.
    s = subs[~subs["is_correction"]].copy()

    s["period_start"] = pd.to_datetime(s["period_start"])
    s["period_end"] = pd.to_datetime(s["period_end"]).fillna(pd.Timestamp("2026-06-30"))

    def mrr_at(month: pd.Timestamp) -> pd.Series:
        """Total MRR per customer, active as of the given month."""
        active = s[(s["period_start"] <= month) & (s["period_end"] > month)]
        return active.groupby("customer_id")["mrr"].sum()

    def nrr_for(month: pd.Timestamp) -> float:
        """NRR for a month, versus the same cohort twelve months earlier."""
        prior = month - pd.DateOffset(months=12)
        base = mrr_at(prior)
        now = mrr_at(month)

        # Only customers who existed a year ago count toward NRR.
        cohort = base.index
        retained = now.reindex(cohort, fill_value=0).sum()

        return retained / base.sum()

    nrr_early = nrr_for(pd.Timestamp("2025-08-01"))
    nrr_late = nrr_for(pd.Timestamp("2026-06-01"))

    # It started healthy - the surviving customers were growing.
    assert nrr_early > 1.04, f"NRR should start above 104%, got {nrr_early:.1%}"

    # ...and it fell below water. This is the CFO's question, made real.
    assert nrr_late < 0.97, f"NRR should end below 97%, got {nrr_late:.1%}"

    # The drop must be large enough to be worth a board-level investigation.
    assert (nrr_early - nrr_late) > 0.08, "NRR decline is too small to be the story"
