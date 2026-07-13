"""Reconciliation: tie the subscription tables to the billing ledger, to the dollar.

THE MILESTONE MOST ANALYSTS SKIP
--------------------------------
The usual path is: compute NRR from the subscription table, put it in a deck,
present it. Then the CFO says "that does not tie to my ledger," and the meeting
is over. Nothing you said afterwards matters, because the number was never
trusted in the first place.

This module makes that impossible. It builds a BRIDGE from contracted MRR
(what analytics thinks was sold) to billed revenue (what finance actually
invoiced), and it accounts for every cent of the difference.

THE STANDARD
------------
The unexplained residual must be ZERO. Not small. Not "immaterial". Zero.

A residual of $340 sounds harmless until someone asks what it is, and the true
answer is "I do not know" - at which point every other number in the deck is
suspect too. An unexplained residual is not a rounding error; it is a hole in
the logic where an unknown defect is living.

THE TWO BRIDGES
---------------
1. LEDGER INTEGRITY. Raw invoices -> clean invoices.
   Where did the duplicates and orphans go? Every raw row is kept, quarantined,
   or removed with a dollar figure attached.

2. CONTRACTED -> BILLED. Subscription MRR -> invoiced revenue.
   These differ for LEGITIMATE reasons, not just defects:
     - add-ons are billed but are not MRR                       (ruling R2)
     - refunds reduce billed revenue but not contracted MRR     (ruling R5)
     - timing: a plan starting on the 12th is not active on the 1st
   Each is a named line. Nothing is hand-waved.

WHAT THIS BUYS YOU
------------------
You walk into the room and say: "The gap is $112,252. It is refunds of $139,010,
less $136,747 of subscriptions we did not invoice this month, plus $163,504 of
mid-month starts. Every dollar is accounted for."

That is not analysis. That is control - and control is what makes a number
something people act on rather than merely believe.

Run:  python -m northwind.reconcile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from northwind.metrics import MetricsContract, load_contract

# The last day of the company's history. An open-ended subscription runs to here.
AS_OF = pd.Timestamp("2026-06-30")

# Anything above this is a bug, not a rounding artefact. Floats are imprecise;
# reconciliations are not. A cent of slack absorbs float noise and nothing else.
TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# BRIDGE 1 - LEDGER INTEGRITY
# ---------------------------------------------------------------------------


def reconcile_ledger(
    raw_invoices: pd.DataFrame,
    clean_invoices: pd.DataFrame,
    duplicates: pd.DataFrame,
    orphans: pd.DataFrame,
) -> pd.DataFrame:
    """Prove that every raw invoice row ended up somewhere we can name.

    This is the boring one, and it is the one that saves you. If rows can vanish
    between raw and clean, then no downstream number can be trusted - and the
    failure is silent, because the total that comes out still looks plausible.
    """
    rows = [
        {
            "line": "Raw invoice rows (as exported by billing)",
            "rows": len(raw_invoices),
            "dollars": float(raw_invoices["amount"].sum()),
        },
        {
            "line": "  less: duplicates removed",
            "rows": -len(duplicates),
            "dollars": -float(duplicates["amount"].sum()),
        },
        {
            "line": "  less: orphans quarantined",
            "rows": -len(orphans),
            "dollars": -float(orphans["amount"].sum()),
        },
        {
            "line": "= Clean invoice rows",
            "rows": len(clean_invoices),
            "dollars": float(clean_invoices["amount"].sum()),
        },
    ]

    bridge = pd.DataFrame(rows)

    # The residual: does the arithmetic actually close?
    expected_rows = len(raw_invoices) - len(duplicates) - len(orphans)
    expected_dollars = (
        float(raw_invoices["amount"].sum())
        - float(duplicates["amount"].sum())
        - float(orphans["amount"].sum())
    )

    bridge = pd.concat(
        [
            bridge,
            pd.DataFrame(
                [
                    {
                        "line": "UNEXPLAINED RESIDUAL",
                        "rows": len(clean_invoices) - expected_rows,
                        "dollars": float(clean_invoices["amount"].sum()) - expected_dollars,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    return bridge


# ---------------------------------------------------------------------------
# BRIDGE 2 - CONTRACTED MRR TO BILLED REVENUE
# ---------------------------------------------------------------------------


def _contracted_mrr(subs: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """The subscriptions that were live on the first of the month.

    MRR is a point-in-time measure, and the convention is the 1st. That
    convention is exactly WHY the timing differences below exist - a plan that
    starts on the 12th gets invoiced this month but was never 'active on the
    1st'. That is not a defect. It is a definition doing its job.
    """
    return subs[(subs["period_start"] <= month) & (subs["period_end"] > month)]


def reconcile_month(
    subs: pd.DataFrame,
    invoices: pd.DataFrame,
    month: pd.Timestamp,
    contract: MetricsContract,
) -> dict:
    """Bridge contracted MRR to billed revenue for a single month.

    Every line is a real economic difference, not a fudge:

      contracted MRR            subscriptions live on the 1st
      - not invoiced            live on the 1st, but no invoice raised
      + invoiced, not live      mid-month starts and similar
      + refunds (base)          reverse revenue billed in an earlier month  [R5]
      = billed base revenue

      + add-ons                 billed, but NOT MRR                          [R2]
      = billed total
    """
    next_month = month + pd.DateOffset(months=1)

    active = _contracted_mrr(subs, month)
    contracted = float(active["mrr"].sum())

    # Invoices belonging to this month. effective_date, not issued_date - a
    # refund lives in the month of the invoice it reverses. That is ruling R5.
    in_month = invoices[
        (invoices["effective_date"] >= month) & (invoices["effective_date"] < next_month)
    ]

    # Split real charges from reversals.
    charges = in_month[in_month["reverses_invoice_id"].isna()]
    refunds = in_month[in_month["reverses_invoice_id"].notna()]

    active_subs = set(active["subscription_id"])
    charged_subs = set(charges["subscription_id"])

    # LINE 1: live on the 1st, but we raised no invoice. Contracted, not billed.
    not_invoiced = float(
        active[~active["subscription_id"].isin(charged_subs)]["mrr"].sum()
    )

    # LINE 2: invoiced, but not live on the 1st. Billed, not contracted.
    # Mostly mid-month starts. Legitimate, and it must be named.
    invoiced_not_active = float(
        charges[~charges["subscription_id"].isin(active_subs)]["base_amount"].sum()
    )

    # LINE 3: refunds. They reduce billed revenue; they never touched MRR.  [R5]
    refund_base = float(refunds["base_amount"].sum())

    # LINE 4: usage add-ons. Billed, but not recurring, so not MRR.         [R2]
    addons = float(in_month["addon_amount"].sum())
    if contract.addons_count_as_mrr:
        # If the contract ever changes its mind, this line disappears from the
        # bridge because add-ons would already be inside contracted MRR.
        addons = 0.0

    billed_base = float(in_month["base_amount"].sum())
    billed_total = float(in_month["amount"].sum())

    # Does it close?
    expected_base = contracted - not_invoiced + invoiced_not_active + refund_base
    residual = billed_base - expected_base

    return {
        "month": month,
        "contracted_mrr": round(contracted, 2),
        "less_not_invoiced": round(-not_invoiced, 2),
        "plus_invoiced_not_active": round(invoiced_not_active, 2),
        "plus_refunds_base": round(refund_base, 2),
        "billed_base": round(billed_base, 2),
        "residual": round(residual, 2),
        "plus_addons": round(addons, 2),
        "billed_total": round(billed_total, 2),
        "invoice_count": len(in_month),
        "active_subscriptions": len(active),
    }


def reconcile_all_months(
    subs: pd.DataFrame,
    invoices: pd.DataFrame,
    contract: MetricsContract,
    start: str = "2024-02-01",
    end: str = "2026-06-01",
) -> pd.DataFrame:
    """Run the bridge for every month in the analysis window.

    A bridge that closes for ONE month proves nothing - it might be luck, or a
    coincidence of offsetting errors. It has to close for every month, or the
    logic is not actually right.
    """
    s = subs.copy()
    s["period_start"] = pd.to_datetime(s["period_start"])
    s["period_end"] = pd.to_datetime(s["period_end"]).fillna(AS_OF)

    inv = invoices.copy()
    inv["effective_date"] = pd.to_datetime(inv["effective_date"])

    months = pd.date_range(start=start, end=end, freq="MS")

    return pd.DataFrame([reconcile_month(s, inv, m, contract) for m in months])


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------


def reconcile(raw_dir: Path, processed_dir: Path, out_dir: Path) -> dict:
    """Run both bridges, write them out, and print the verdict."""
    contract = load_contract()

    raw_invoices = pd.read_csv(raw_dir / "invoices.csv")
    # The raw amounts are text. Parse them so the ledger bridge can add up.
    from northwind.clean import parse_amount

    raw_invoices["amount"] = raw_invoices["amount"].map(parse_amount)

    clean_invoices = pd.read_parquet(processed_dir / "invoices.parquet")
    duplicates = pd.read_parquet(processed_dir / "quarantine_duplicate_invoices.parquet")
    orphans = pd.read_parquet(processed_dir / "quarantine_orphan_invoices.parquet")
    subs = pd.read_parquet(processed_dir / "subscriptions.parquet")

    # --- Bridge 1 ---
    ledger = reconcile_ledger(raw_invoices, clean_invoices, duplicates, orphans)

    print("=" * 72)
    print("BRIDGE 1 - LEDGER INTEGRITY  (raw invoices -> clean invoices)")
    print("=" * 72)
    for row in ledger.itertuples(index=False):
        print(f"  {row.line:<44} {row.rows:>8,}  ${row.dollars:>15,.2f}")

    # --- Bridge 2 ---
    monthly = reconcile_all_months(subs, clean_invoices, contract)

    print()
    print("=" * 72)
    print("BRIDGE 2 - CONTRACTED MRR -> BILLED REVENUE  (monthly)")
    print("=" * 72)

    latest = monthly.iloc[-1]
    print(f"\n  Most recent month: {latest['month']:%Y-%m}\n")
    print(f"  {'Contracted MRR (subscriptions live on the 1st)':<46} "
          f"${latest['contracted_mrr']:>14,.2f}")
    print(f"  {'  less: live but not invoiced':<46} ${latest['less_not_invoiced']:>14,.2f}")
    print(f"  {'  plus: invoiced but not live on the 1st':<46} "
          f"${latest['plus_invoiced_not_active']:>14,.2f}")
    print(f"  {'  plus: refunds, base portion  [R5]':<46} "
          f"${latest['plus_refunds_base']:>14,.2f}")
    print(f"  {'-' * 46} {'-' * 15}")
    print(f"  {'= Billed base revenue (the ledger)':<46} ${latest['billed_base']:>14,.2f}")
    print(f"  {'  plus: usage add-ons, not MRR  [R2]':<46} ${latest['plus_addons']:>14,.2f}")
    print(f"  {'= Billed total':<46} ${latest['billed_total']:>14,.2f}")

    # --- The verdict ---
    worst = monthly["residual"].abs().max()
    failures = monthly[monthly["residual"].abs() > TOLERANCE]

    print()
    print("=" * 72)
    print(f"  Months reconciled       : {len(monthly)}")
    print(f"  Largest residual        : ${worst:,.2f}")
    print(f"  Months that do not tie  : {len(failures)}")
    print("=" * 72)

    if len(failures) == 0:
        print("\n  RECONCILED. Every month ties to the ledger to the cent.\n")
    else:
        print("\n  DOES NOT RECONCILE. Unexplained residual in:\n")
        print(failures[["month", "residual"]].to_string(index=False))
        print("\n  An unexplained residual is not a rounding error. It is a defect")
        print("  we have not found yet. Do not report any number until it closes.\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(out_dir / "reconciliation_ledger.parquet", index=False)
    monthly.to_parquet(out_dir / "reconciliation_monthly.parquet", index=False)

    return {"ledger": ledger, "monthly": monthly, "worst_residual": float(worst)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile subscriptions to the billing ledger.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    reconcile(args.raw, args.processed, args.out)


if __name__ == "__main__":
    main()
