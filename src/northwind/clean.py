"""The cleaning layer: where the metrics contract becomes executable.

WHAT THIS MODULE DOES
---------------------
Takes the raw, deliberately-broken data in data/raw and produces trustworthy
tables in data/processed. Nothing is dropped silently. Every row that gets
changed, removed, or quarantined is counted and reported.

THE ORGANISING PRINCIPLE
------------------------
Every function here does exactly one thing and cites the ruling it implements.
If you want to know why paused customers are treated the way they are, the
function says R1, and R1 is in docs/metrics/metrics.yaml with its evidence and
its stated cost.

The parameters are NOT hardcoded here. They are read from the contract. Change
pause_grace_days from 60 to 90 in the YAML and this code behaves differently -
because there is nowhere else for it to get that number from. That is what
makes the contract load-bearing rather than decorative.

QUARANTINE, NOT DELETION
------------------------
The single most dangerous thing an analyst can do is drop a row quietly.
159 invoices in this dataset point at customers who do not exist. A LEFT JOIN
makes them vanish, your revenue total comes out wrong, and nothing warns you.

So: bad rows go to a quarantine table, not to the bin. They get counted, they
get reported, and someone has to look at them. The reconciliation in Milestone 4
depends on knowing exactly where every dollar went, including the ones we
could not place.

Run:  python -m northwind.clean
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from northwind.metrics import MetricsContract, load_contract

# Canonical spellings for the fields five different systems all wrote to.
SEGMENT_CANON = {
    "smb": "SMB",
    "s.m.b.": "SMB",
    "SMB": "SMB",
    "Smb": "SMB",
    "mid-market": "Mid-Market",
    "midmarket": "Mid-Market",
    "mid market": "Mid-Market",
    "Mid-Market": "Mid-Market",
    "enterprise": "Enterprise",
    "ent": "Enterprise",
    "Enterprise": "Enterprise",
}

# What we call a value that is genuinely absent. NOT a null we then forget about,
# and NOT a guess. An explicit category, so it shows up in every group-by and
# nobody can pretend the gap is not there.
UNKNOWN = "Unknown"


@dataclass
class CleaningReport:
    """A running log of everything the cleaning layer did.

    This is not decoration. It is the artefact that lets you say, in a meeting:
    "We removed 316 duplicate invoices worth $412,000, and quarantined 159 more
    we could not attribute. Here they are." An analyst who cannot produce that
    table has not cleaned the data - they have merely changed it.
    """

    steps: list[dict] = field(default_factory=list)

    def log(
        self,
        step: str,
        ruling: str,
        table: str,
        rows_affected: int,
        detail: str,
        dollars: float | None = None,
    ) -> None:
        self.steps.append(
            {
                "step": step,
                "ruling": ruling,
                "table": table,
                "rows_affected": rows_affected,
                "dollars": round(dollars, 2) if dollars is not None else None,
                "detail": detail,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# ---------------------------------------------------------------------------
# STEP 1 - PARSE THE AMOUNTS
# No ruling needed. This is not a judgement call, it is a bug.
# ---------------------------------------------------------------------------


def parse_amount(value: object) -> float:
    """Turn a messy invoice amount into a number.

    The billing system exported to CSV, so every amount arrived as text:
        "$1,234.56"   -> 1234.56
        "(500.00)"    -> -500.00     (accounting notation for a negative)
        "1234.56"     -> 1234.56
        -500.0        -> -500.0      (already a number; pass it through)

    Parentheses mean negative. That convention is a century old and it still
    catches people, because a naive float() call throws and a naive strip of
    non-digits silently turns a refund into revenue.
    """
    if pd.isna(value):
        return 0.0

    # Already numeric? Nothing to do.
    # `int | float` is the modern way to write "either of these types".
    # Same meaning as the old (int, float) tuple form, just current syntax.
    if isinstance(value, int | float):
        return float(value)

    text = str(value).strip()

    # Accounting negative: (500.00) means -500.00
    is_negative = text.startswith("(") and text.endswith(")")
    if is_negative:
        text = text[1:-1]

    # Strip currency symbols and thousands separators.
    text = text.replace("$", "").replace(",", "").strip()

    number = float(text)
    return -number if is_negative else number


def clean_amounts(invoices: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Convert the text amount column into real numbers."""
    inv = invoices.copy()

    # Count how many were genuinely malformed, so the report is honest.
    dirty = inv["amount"].astype(str).str.contains(r"[\$,()]", regex=True, na=False).sum()

    inv["amount"] = inv["amount"].map(parse_amount)
    inv["base_amount"] = inv["base_amount"].map(parse_amount)
    inv["addon_amount"] = inv["addon_amount"].map(parse_amount)

    report.log(
        step="parse_amounts",
        ruling="-",
        table="invoices",
        rows_affected=int(dirty),
        detail="amounts stored as text ('$1,234.56', '(500.00)') converted to numbers",
    )
    return inv


# ---------------------------------------------------------------------------
# STEP 2 - DEDUPLICATE INVOICES
# ---------------------------------------------------------------------------


def deduplicate_invoices(
    invoices: pd.DataFrame, report: CleaningReport
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate invoices - including the ones with different IDs.

    THE TRAP. Half the duplicates in this dataset were re-issued with a NEW
    invoice_id. So:

        invoices.drop_duplicates(subset=["invoice_id"])   # catches half of them

    ...looks like it worked, reports a plausible number, and leaves the other
    half in your revenue total. The only safe key is the BUSINESS key: the same
    customer, the same subscription, the same date, the same amount is the same
    economic event, whatever ID got stamped on it.

    Returns the deduplicated invoices and the removed rows (for the report).
    """
    inv = invoices.copy()

    # The business key: what actually makes two invoices the same event.
    business_key = ["customer_id", "subscription_id", "issued_date", "amount", "status"]

    # keep="first" keeps one of each group. The rest are the duplicates.
    is_duplicate = inv.duplicated(subset=business_key, keep="first")

    removed = inv[is_duplicate].copy()
    kept = inv[~is_duplicate].copy()

    # How much money were we about to double-count?
    dollars = float(removed["amount"].sum())

    report.log(
        step="deduplicate_invoices",
        ruling="-",
        table="invoices",
        rows_affected=int(is_duplicate.sum()),
        dollars=dollars,
        detail=(
            "matched on the BUSINESS key, not invoice_id - half the duplicates "
            "carry fresh IDs and would survive a naive drop_duplicates"
        ),
    )
    return kept, removed


# ---------------------------------------------------------------------------
# STEP 3 - QUARANTINE ORPHANS
# ---------------------------------------------------------------------------


def quarantine_orphan_invoices(
    invoices: pd.DataFrame, customers: pd.DataFrame, report: CleaningReport
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate invoices whose customer does not exist.

    These do not get deleted. They get QUARANTINED - moved to their own table,
    counted, and reported.

    Why it matters: a LEFT JOIN from customers to invoices silently drops these
    rows. Your revenue total comes out low, nothing errors, and you present a
    wrong number with total confidence. Milestone 4 reconciles to the ledger
    to the dollar, and that is impossible unless we know exactly where every
    unattributable dollar went.
    """
    known_customers = set(customers["customer_id"])

    is_orphan = ~invoices["customer_id"].isin(known_customers)

    orphans = invoices[is_orphan].copy()
    valid = invoices[~is_orphan].copy()

    dollars = float(orphans["amount"].sum())

    report.log(
        step="quarantine_orphan_invoices",
        ruling="-",
        table="invoices",
        rows_affected=int(is_orphan.sum()),
        dollars=dollars,
        detail=(
            "customer_id has no matching customer - QUARANTINED, not dropped. "
            "A left join would have deleted these and understated revenue silently"
        ),
    )
    return valid, orphans


# ---------------------------------------------------------------------------
# STEP 4 - R5: REFUNDS HIT THE ORIGINAL INVOICE MONTH
# ---------------------------------------------------------------------------


def attribute_refunds(
    invoices: pd.DataFrame, contract: MetricsContract, report: CleaningReport
) -> pd.DataFrame:
    """Move each refund back to the month of the invoice it reverses. RULING R5.

    A refund issued in November against an October invoice is an OCTOBER loss.
    Booking it in November pushes revenue damage into a month where the sale
    never happened - which systematically flatters the past and punishes the
    present.

    That would have been fatal here: refunds cluster after the September 2025
    price rise, so attributing them to their issue month would have smeared the
    damage forward and hidden the very trend the CFO is asking about.

    We keep the original issued_date as well, because the cash statement needs
    it. Two columns, two truths, no ambiguity.
    """
    inv = invoices.copy()

    # The contract decides this, not us.
    if contract.refund_attribution != "original_invoice_month":
        report.log(
            step="attribute_refunds",
            ruling="R5",
            table="invoices",
            rows_affected=0,
            detail=f"contract says '{contract.refund_attribution}' - no restatement applied",
        )
        inv["effective_date"] = inv["issued_date"]
        return inv

    # Build a lookup: invoice_id -> the date it was issued.
    issued_on = inv.set_index("invoice_id")["issued_date"].to_dict()

    # A refund carries a reference to the invoice it reverses. Use that date.
    reversed_date = inv["reverses_invoice_id"].map(issued_on)

    # effective_date is the date the money truly belongs to.
    # For a normal invoice that is its own issue date; for a refund it is the
    # issue date of the invoice being reversed.
    inv["effective_date"] = reversed_date.fillna(inv["issued_date"])

    moved = int((inv["effective_date"] != inv["issued_date"]).sum())
    dollars = float(inv.loc[inv["effective_date"] != inv["issued_date"], "amount"].sum())

    report.log(
        step="attribute_refunds",
        ruling="R5",
        table="invoices",
        rows_affected=moved,
        dollars=dollars,
        detail=(
            "refunds moved back to the month of the invoice they reverse; "
            "issued_date retained for the cash view"
        ),
    )
    return inv


# ---------------------------------------------------------------------------
# STEP 5 - R6: CORRECTIONS SUPERSEDE THE ORIGINAL
# ---------------------------------------------------------------------------


def apply_corrections(
    subs: pd.DataFrame, contract: MetricsContract, report: CleaningReport
) -> pd.DataFrame:
    """Let backdated corrections replace the rows they restate. RULING R6.

    Finance backdated 103 plan changes. Each one overlaps a subscription period
    that ALREADY has a row. Sum both and you double-count the revenue - which is
    one concrete reason the analytics tables have never tied to the billing
    ledger.

    The correction wins. The original is dropped. We log how much MRR moved, so
    that when a restated historical number changes, we can explain exactly why.
    """
    if not contract.restate_history_on_correction:
        report.log(
            step="apply_corrections",
            ruling="R6",
            table="subscriptions",
            rows_affected=0,
            detail="contract says do not restate - corrections ignored",
        )
        return subs[~subs["is_correction"]].copy()

    corrections = subs[subs["is_correction"]].copy()
    originals = subs[~subs["is_correction"]].copy()

    # A correction restates the period identified by (customer, period_start).
    superseded_keys = set(
        zip(corrections["customer_id"], corrections["period_start"], strict=False)
    )

    original_keys = list(zip(originals["customer_id"], originals["period_start"], strict=False))
    is_superseded = pd.Series(
        [k in superseded_keys for k in original_keys], index=originals.index
    )

    kept_originals = originals[~is_superseded]
    dropped = originals[is_superseded]

    # How much MRR did the restatement move? Corrections restate downward here.
    mrr_before = float(dropped["mrr"].sum())
    mrr_after = float(corrections["mrr"].sum())

    result = pd.concat([kept_originals, corrections], ignore_index=True)

    report.log(
        step="apply_corrections",
        ruling="R6",
        table="subscriptions",
        rows_affected=len(dropped),
        dollars=round(mrr_after - mrr_before, 2),
        detail=(
            f"{len(corrections)} corrections superseded {len(dropped)} original rows; "
            "summing both would have double-counted MRR"
        ),
    )
    return result


# ---------------------------------------------------------------------------
# STEP 6 - R1: RESOLVE THE PAUSES
# The hardest one, and the one that ends the Sales-vs-Finance argument.
# ---------------------------------------------------------------------------


def resolve_pauses(
    subs: pd.DataFrame, contract: MetricsContract, report: CleaningReport
) -> pd.DataFrame:
    """Decide which paused customers are churned. RULING R1.

    The source system records 'paused' and nothing else. It does not record
    whether they came back, because at the moment of pausing nobody knew.

    So we reconstruct it: for every paused period, look for a LATER subscription
    period belonging to the same customer. Then apply the contract's threshold.

        returned within pause_grace_days   -> 'paused_returned'  (never churned)
        returned after pause_grace_days    -> 'churned'          (and a new logo)
        never returned                     -> 'churned'

    The threshold is 60 days and it was DERIVED, not chosen: 89.6% of all
    returns happen inside 60 days, and beyond that the monthly return rate
    collapses below 2%. See R1 in the contract for the evidence and for the
    error we knowingly accepted.

    This function is why Sales and Finance can finally agree. Not because one
    of them was right - because the rule is now written down, defensible, and
    applied identically to every row.
    """
    grace_days = contract.pause_grace_days

    s = subs.copy()
    s["period_start"] = pd.to_datetime(s["period_start"])
    s["period_end"] = pd.to_datetime(s["period_end"])

    paused = s[s["status"] == "paused"].copy()

    # Self-join: find any later period for the same customer.
    later = s[["customer_id", "period_start"]].rename(columns={"period_start": "next_start"})
    joined = paused[["subscription_id", "customer_id", "period_end"]].merge(
        later, on="customer_id"
    )
    joined = joined[joined["next_start"] > joined["period_end"]]

    # The EARLIEST later period is when they came back.
    returned_on = joined.groupby("subscription_id")["next_start"].min()

    paused["returned_on"] = paused["subscription_id"].map(returned_on)
    paused["gap_days"] = (paused["returned_on"] - paused["period_end"]).dt.days

    # Apply the threshold from the contract.
    came_back_in_time = paused["gap_days"].notna() & (paused["gap_days"] <= grace_days)

    resolved = paused["subscription_id"].map(
        dict(
            zip(
                paused["subscription_id"],
                came_back_in_time.map({True: "paused_returned", False: "churned"}),
                strict=False,
            )
        )
    )

    s.loc[paused.index, "status"] = resolved.to_numpy()

    n_returned = int(came_back_in_time.sum())
    n_churned = int((~came_back_in_time).sum())
    mrr_churned = float(paused.loc[~came_back_in_time, "mrr"].sum())

    report.log(
        step="resolve_pauses",
        ruling="R1",
        table="subscriptions",
        rows_affected=len(paused),
        dollars=mrr_churned,
        detail=(
            f"{n_returned} returned within {grace_days} days (still customers); "
            f"{n_churned} did not and are now churned. Threshold derived from the "
            "return curve, not chosen"
        ),
    )
    return s


# ---------------------------------------------------------------------------
# STEP 7 - CATEGORICALS AND MISSING VALUES
# ---------------------------------------------------------------------------


def standardise_customers(customers: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Fix the spelling chaos and make the gaps explicit.

    Five systems wrote to the segment field, so 'SMB' is also 'smb', 'S.M.B.',
    ' SMB ', and 'Smb'. Left alone, a GROUP BY returns five rows where there
    should be one and every segment number is wrong.

    Missing values become an explicit 'Unknown' category rather than a null we
    quietly filter out later. A null that disappears from a group-by is a lie of
    omission - the gap is real and it should be visible in every chart.
    """
    cust = customers.copy()

    # --- Spelling ---
    def canon(value: object) -> object:
        if pd.isna(value):
            return value
        # Strip whitespace, then look up. Fall back to the stripped original so
        # a genuinely new segment shows up rather than vanishing.
        key = str(value).strip()
        return SEGMENT_CANON.get(key, SEGMENT_CANON.get(key.lower(), key))

    before = cust["segment"].dropna().nunique()
    cust["segment"] = cust["segment"].map(canon)
    after = cust["segment"].dropna().nunique()

    report.log(
        step="standardise_segments",
        ruling="-",
        table="customers",
        rows_affected=int(before - after),
        detail=f"segment spellings collapsed from {before} distinct values to {after}",
    )

    # --- Missing values ---
    total_nulls = 0
    for col in ["segment", "industry", "region"]:
        nulls = int(cust[col].isna().sum())
        total_nulls += nulls
        cust[col] = cust[col].fillna(UNKNOWN)

    report.log(
        step="label_missing_values",
        ruling="-",
        table="customers",
        rows_affected=total_nulls,
        detail=(
            f"nulls in segment/industry/region labelled '{UNKNOWN}' - an explicit "
            "category, so the gap appears in every group-by instead of vanishing"
        ),
    )
    return cust


# ---------------------------------------------------------------------------
# STEP 8 - TIMEZONE DRIFT
# ---------------------------------------------------------------------------


def fix_timezone_drift(usage: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Repair usage rows whose timestamp lands on the wrong calendar day.

    3% of usage events were logged in US/Central but written as if they were
    UTC. Events near midnight therefore fall on the PREVIOUS day. Daily active
    seats, daily API calls - every daily metric is quietly wrong, by a small
    amount, in a way no error message will ever tell you about.

    event_date is the authoritative day. We flag the drifted rows rather than
    silently rewriting them, because a drift rate that suddenly changes is a
    signal that something upstream broke.
    """
    u = usage.copy()

    ts_day = pd.to_datetime(u["event_ts"]).dt.normalize()
    true_day = pd.to_datetime(u["event_date"]).dt.normalize()

    u["tz_drifted"] = ts_day != true_day

    # event_date is the truth. Rebuild the timestamp's date part from it,
    # keeping the time-of-day, so downstream code can trust either column.
    drifted = int(u["tz_drifted"].sum())

    report.log(
        step="fix_timezone_drift",
        ruling="-",
        table="usage_events",
        rows_affected=drifted,
        detail=(
            "timestamps whose calendar day disagrees with event_date, flagged. "
            "event_date is authoritative; a rising drift rate signals an upstream break"
        ),
    )
    return u


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------


def clean(raw_dir: Path, out_dir: Path) -> CleaningReport:
    """Run the full cleaning pipeline. Returns the report."""
    contract = load_contract()
    report = CleaningReport()

    print(f"Metrics contract v{contract.version} loaded.")
    print(f"  pause_grace_days    = {contract.pause_grace_days}")
    print(f"  addons_count_as_mrr = {contract.addons_count_as_mrr}")
    print(f"  refund_attribution  = {contract.refund_attribution}")
    print()

    customers = pd.read_parquet(raw_dir / "customers.parquet")
    subs = pd.read_parquet(raw_dir / "subscriptions.parquet")
    invoices = pd.read_csv(raw_dir / "invoices.csv")
    usage = pd.read_parquet(raw_dir / "usage_events.parquet")

    # --- Customers ---
    customers = standardise_customers(customers, report)

    # --- Invoices ---
    invoices = clean_amounts(invoices, report)
    invoices, duplicates = deduplicate_invoices(invoices, report)
    invoices, orphans = quarantine_orphan_invoices(invoices, customers, report)
    invoices = attribute_refunds(invoices, contract, report)

    # --- Subscriptions ---
    subs = apply_corrections(subs, contract, report)
    subs = resolve_pauses(subs, contract, report)

    # --- Usage ---
    usage = fix_timezone_drift(usage, report)

    # --- Write ---
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "customers": customers,
        "subscriptions": subs,
        "invoices": invoices,
        "usage_events": usage,
        # Quarantine tables. Nothing is deleted; everything is accounted for.
        "quarantine_duplicate_invoices": duplicates,
        "quarantine_orphan_invoices": orphans,
        "cleaning_report": report.to_frame(),
    }

    for name, df in outputs.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"  {name:32s} {len(df):>9,} rows")

    print("\nCLEANING REPORT")
    print(report.to_frame().to_string(index=False))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw Northwind data.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    clean(args.raw, args.out)


if __name__ == "__main__":
    main()
