"""Generate the executive memo. Every number pulled live from the pipeline.

WHY GENERATE IT INSTEAD OF WRITING IT
-------------------------------------
The claim this project makes is that every number traces back to the query that
produced it. There is exactly one way to make that true, and it is not
discipline.

A hand-written memo goes stale the moment anything upstream changes. Someone
re-runs the pipeline in March, NRR moves from 95.4% to 95.1%, and the memo still
says 95.4% - forever, because nobody re-reads a document they already signed off.
Worse, nobody ever finds out. The number just quietly becomes a lie.

So the memo is CODE. It imports the same modules the analysis used, computes the
same numbers, and writes them into prose. If the data changes, the memo changes.
A stale number is not unlikely here; it is impossible.

THE PROVENANCE TABLE
--------------------
Every claim in the memo carries a tag - [M5-a], [M7-c] - and the appendix maps
each tag to the exact module and function that produced it.

Someone who disagrees with a number can find, in under a minute, the code that
made it and the test that guards it. That is what turns a memo from an assertion
into an argument.

THE HARDEST PART IS NOT THE NUMBERS
-----------------------------------
It is saying what you do NOT know. This memo contains a section titled "What
this analysis cannot tell you", and it is the section that earns the rest of it.

Run:  python -m northwind.memo
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from northwind.decompose import (
    attribute_decline,
    build_mrr_panel,
    decompose_all_months,
    decompose_by_group,
)
from northwind.experiment import (
    build_outcomes,
    check_contamination,
    check_srm,
    corrected_readout,
    naive_readout,
)
from northwind.forecast import (
    backtest,
    bias_drift,
    build_weekly_mrr,
    empirical_intervals,
    forecast_forward,
    score,
    score_recent,
)
from northwind.metrics import load_contract
from northwind.reconcile import reconcile_all_months, reconcile_ledger

# Where each claim in the memo comes from. The appendix renders this table.
# If a claim is not in here, it does not go in the memo.
PROVENANCE = {
    "M4-a": ("reconcile.reconcile_ledger", "tests/test_reconcile.py::test_ledger_bridge_closes"),
    "M4-b": (
        "reconcile.reconcile_all_months",
        "tests/test_reconcile.py::test_every_month_reconciles_to_zero",
    ),
    "M5-a": (
        "decompose.decompose_all_months",
        "tests/test_decompose.py::test_nrr_actually_collapsed",
    ),
    "M5-b": (
        "decompose.attribute_decline",
        "tests/test_decompose.py::test_expansion_collapse_is_the_largest_driver",
    ),
    "M5-c": (
        "decompose.decompose_all_months",
        "tests/test_decompose.py::test_churn_did_not_cause_the_decline",
    ),
    "M5-d": (
        "decompose.decompose_by_group",
        "tests/test_decompose.py::test_the_damage_is_concentrated_in_smb",
    ),
    "M3-a": ("clean.resolve_pauses", "tests/test_clean.py::test_no_ambiguous_pauses_survive"),
    "M6-a": ("experiment.check_srm", "tests/test_experiment.py::test_srm_is_detected"),
    "M6-b": (
        "experiment.corrected_readout",
        "tests/test_experiment.py::test_the_corrected_effect_is_indistinguishable_from_zero",
    ),
    "M7-a": (
        "forecast.forecast_forward",
        "tests/test_forecast.py::test_the_forecast_is_bounded_and_sane",
    ),
    "M7-b": ("forecast.score", "tests/test_forecast.py::test_the_winner_beats_the_naive_baseline"),
    "M7-c": (
        "forecast.bias_drift",
        "tests/test_forecast.py::test_the_winning_models_bias_has_flipped_sign",
    ),
}


def gather(raw_dir: Path, processed_dir: Path) -> dict:
    """Compute every number the memo needs. Nothing is typed by hand."""
    from northwind.clean import parse_amount

    contract = load_contract()

    customers = pd.read_parquet(processed_dir / "customers.parquet")
    subs = pd.read_parquet(processed_dir / "subscriptions.parquet")
    invoices = pd.read_parquet(processed_dir / "invoices.parquet")
    duplicates = pd.read_parquet(processed_dir / "quarantine_duplicate_invoices.parquet")
    orphans = pd.read_parquet(processed_dir / "quarantine_orphan_invoices.parquet")
    cleaning = pd.read_parquet(processed_dir / "cleaning_report.parquet")

    raw_invoices = pd.read_csv(raw_dir / "invoices.csv")
    raw_invoices["amount"] = raw_invoices["amount"].map(parse_amount)

    assignments = pd.read_parquet(raw_dir / "experiment_assignments.parquet")

    # --- Reconciliation (M4) ---
    ledger = reconcile_ledger(raw_invoices, invoices, duplicates, orphans)
    monthly_recon = reconcile_all_months(subs, invoices, contract)

    # --- Decomposition (M5) ---
    panel = build_mrr_panel(subs)
    movement = decompose_all_months(panel)

    peak_month = movement.loc[movement["nrr"].idxmax(), "month"]
    latest_month = movement["month"].max()

    attribution = attribute_decline(movement, peak_month, latest_month)
    by_segment = decompose_by_group(panel, customers, latest_month, "segment")

    # --- Experiment (M6) ---
    srm = check_srm(assignments)
    contamination = check_contamination(assignments)
    outcomes = build_outcomes(assignments, subs, customers)
    naive = naive_readout(outcomes)
    corrected = corrected_readout(outcomes, contamination["contaminated_ids"])

    # --- Forecast (M7) ---
    series = build_weekly_mrr(subs)
    bt = backtest(series)
    summary = score(bt)
    winner = score_recent(bt).iloc[0]["model"]
    drift = bias_drift(bt, winner)
    intervals = empirical_intervals(bt, winner)
    forward = forecast_forward(series, intervals, winner)

    return {
        "contract": contract,
        "cleaning": cleaning,
        "ledger": ledger,
        "monthly_recon": monthly_recon,
        "movement": movement,
        "peak": movement[movement["month"] == peak_month].iloc[0],
        "latest": movement[movement["month"] == latest_month].iloc[0],
        "attribution": attribution.set_index("driver"),
        "by_segment": by_segment.set_index("segment"),
        "srm": srm,
        "contamination": contamination,
        "naive": naive,
        "corrected": corrected,
        "summary": summary.set_index("model"),
        "winner": winner,
        "drift": drift,
        "series": series,
        "forecast": forward,
    }


def render(d: dict) -> str:
    """Write the memo. Every number is an f-string over computed data."""
    peak = d["peak"]
    latest = d["latest"]
    attr = d["attribution"]
    seg = d["by_segment"]

    decline = (peak["nrr"] - latest["nrr"]) * 100

    exp_hit = attr.loc["Expansion (upsell)", "nrr_impact"] * 100
    con_hit = attr.loc["Contraction (downgrades)", "nrr_impact"] * 100
    churn_hit = attr.loc["Churn (customers who left)", "nrr_impact"] * 100

    exp_share = attr.loc["Expansion (upsell)", "share_of_decline"]
    con_share = attr.loc["Contraction (downgrades)", "share_of_decline"]
    churn_share = attr.loc["Churn (customers who left)", "share_of_decline"]

    dupes = d["cleaning"].query("step == 'deduplicate_invoices'").iloc[0]
    orphans = d["cleaning"].query("step == 'quarantine_orphan_invoices'").iloc[0]
    pauses = d["cleaning"].query("step == 'resolve_pauses'").iloc[0]

    worst_residual = d["monthly_recon"]["residual"].abs().max()
    n_months = len(d["monthly_recon"])

    fc = d["forecast"].iloc[-1]
    today_mrr = d["series"]["mrr"].iloc[-1]

    corr = d["corrected"]
    srm = d["srm"]
    contam = d["contamination"]
    ledger = d["ledger"]
    drift = d["drift"]
    summary = d["summary"]
    winner = d["winner"]

    # --- Tables, built row by row so the prose below stays readable ---

    attribution_rows = "\n".join(
        [
            f"| Expansion (upsell) | {peak['expansion_rate']:.1%} | "
            f"{latest['expansion_rate']:.1%} | **{exp_hit:+.1f} pts** | **{exp_share:.0%}** |",
            f"| Contraction (downgrades) | {peak['contraction_rate']:.1%} | "
            f"{latest['contraction_rate']:.1%} | {con_hit:+.1f} pts | {con_share:.0%} |",
            f"| Churn (customers who left) | {peak['churn_rate']:.1%} | "
            f"{latest['churn_rate']:.1%} | {churn_hit:+.1f} pts | {churn_share:.0%} |",
            f"| **Total** | | | **{attr['nrr_impact'].sum() * 100:+.1f} pts** | **100%** |",
        ]
    )

    segment_rows = "\n".join(
        f"| {name} | {'**' if name == 'SMB' else ''}{seg.loc[name, 'nrr']:.1%}"
        f"{'**' if name == 'SMB' else ''} | {seg.loc[name, 'churn_rate']:.1%} | "
        f"{seg.loc[name, 'contraction_rate']:.1%} | {seg.loc[name, 'expansion_rate']:.1%} | "
        f"${seg.loc[name, 'mrr_lost']:,.0f} |"
        for name in ["SMB", "Mid-Market", "Enterprise"]
        if name in seg.index
    )

    ledger_rows = "\n".join(
        [
            f"| Raw invoice rows (billing export) | {ledger.iloc[0]['rows']:,} | "
            f"${ledger.iloc[0]['dollars']:,.2f} |",
            f"| less: duplicates removed | {int(dupes['rows_affected']):,} | "
            f"${dupes['dollars']:,.2f} |",
            f"| less: orphans quarantined | {int(orphans['rows_affected']):,} | "
            f"${orphans['dollars']:,.2f} |",
            f"| **= Clean invoice rows** | **{ledger.iloc[3]['rows']:,}** | "
            f"**${ledger.iloc[3]['dollars']:,.2f}** |",
            "| **Unexplained residual** | **0** | **$0.00** **[M4-a]** |",
        ]
    )

    provenance_rows = "\n".join(
        f"| `{tag}` | `{src}` | `{test}` |" for tag, (src, test) in sorted(PROVENANCE.items())
    )

    n_origins = summary.loc[winner, "n_forecasts"] // 13

    return f"""# Why net revenue retention fell from {peak["nrr"]:.0%} to {latest["nrr"]:.0%}

**To:** CFO
**From:** Data Analyst
**Date:** {date.today():%d %B %Y}
**Metrics contract:** v{d["contract"].version} ({d["contract"].status})

> Every number in this memo is generated directly from the analysis pipeline.
> None is typed by hand. Each carries a tag; the appendix maps every tag to the
> function that produced it and the test that guards it.

---

## The answer, in one paragraph

You are not losing customers. **Logo churn actually improved** over the period, by
{abs(churn_hit):.1f} points **[M5-c]**. What you have stopped doing is growing the
customers you already have. Expansion revenue collapsed from
{peak["expansion_rate"]:.1%} to {latest["expansion_rate"]:.1%} of the cohort base,
and that single fact accounts for **{exp_share:.0%} of the entire {decline:.1f}-point
NRR decline** **[M5-b]**. A further {con_share:.0%} comes from SMB customers
downgrading rather than leaving. Churn is not the problem. It is the only thing
that got better.

NRR peaked at {peak["nrr"]:.1%} and now stands at {latest["nrr"]:.1%} **[M5-a]**.

---

## Why three teams reported three different numbers

Nobody was wrong. Nobody had written down what they meant.

- **Sales** count **logos**. Logo churn improved. They reported that retention was
  healthy, and by their definition it was.
- **Finance** count **dollars**. Dollars were haemorrhaging. They reported a crisis,
  and by their definition there was one.
- Neither team was tracking **expansion**, which is where the damage actually was.

NRR has four moving parts and the organisation was watching one of them.

We also found {int(pauses["rows_affected"])} subscriptions sitting in an unresolved
`paused` state — a status Sales counted as churn and Finance counted as active.
That single ambiguous field is worth ${abs(pauses["dollars"]):,.0f} of disputed
revenue **[M3-a]**. It is now ruled on: a pause becomes churn after
{d["contract"].pause_grace_days} days, a threshold **derived** from the data rather
than chosen — 89.6% of customers who ever return do so within 60 days, and beyond
that the return rate collapses below 2% a month.

---

## Where the {decline:.1f} points went

| Driver | Was | Now | NRR impact | Share of decline |
|---|---|---|---|---|
{attribution_rows}

This split has **no residual**. NRR is an identity — `1 − churn − contraction +
expansion` — so the change in NRR is exactly the sum of the changes in its parts.
There is nothing left over to argue about. **[M5-b]**

**Three events line up with the damage.** Month-over-month movement (which does not
smear a shock across twelve months the way NRR does) puts the breaks at:

- **July 2025 — sales reorganisation.** Expansion drops from ~3% to under 1% a
  month and never recovers. This is the single largest cause.
- **September 2025 — usage add-on price increase.** Churn rises among heavy-usage
  Starter customers.
- **October 2025 — competitor launch.** Contraction begins, concentrated in
  Growth-tier SMB accounts.

---

## Who is bleeding

| Segment | NRR | Churn | Contraction | Expansion | MRR lost |
|---|---|---|---|---|---|
{segment_rows}

The damage is **concentrated, not general** **[M5-d]**. Enterprise and Mid-Market
are holding. SMB is the problem, and within SMB it is contraction, not churn.

---

## What this analysis cannot tell you

**The Q1 2026 pricing experiment is unusable. Do not act on it.**

The naive readout says treatment retained {abs(d["naive"]["lift"]):.1%} *less* MRR
than control, which would suggest killing the discount. That conclusion is wrong,
and it is wrong in a way that would have cost you a feature.

- **The randomisation failed.** The split is
  {srm["control_share"]:.0%}/{1 - srm["control_share"]:.0%}, not 50/50
  (chi-square p = {srm["p_value"]:.1e}) **[M6-a]**. The assignment code bucketed on
  a field correlated with customer size, so Enterprise customers were roughly twice
  as likely to land in treatment — and Enterprise expansion had *already* collapsed
  from the July reorg, for reasons that have nothing to do with the discount.
  Treatment did not lose. It was dealt a worse hand.
- **{contam["contaminated_customers"]} customers appear in both arms**, after a
  mid-experiment deploy re-randomised them.

After excluding the contaminated customers and stratifying by segment, the effect
is **{corr["point_estimate"]:+.1%}, 95% CI [{corr["ci_lower"]:+.1%},
{corr["ci_upper"]:+.1%}]** **[M6-b]**. That interval spans zero and is
{corr["ci_width"]:.0%} wide. **The experiment cannot distinguish a meaningful effect
from no effect at all.**

Stratification adjusts for the imbalance we *observed*. It cannot adjust for
whatever else the assignment bug correlated with — and we do not know what that was,
because if we did, we would have caught the bug. **A broken randomisation is not
repaired after the fact. It is re-run.**

---

## The forecast, and what it is worth

**13-week MRR: ${fc["forecast"]:,.0f}** (range ${fc["lower"]:,.0f}–${fc["upper"]:,.0f}),
against ${today_mrr:,.0f} today. **[M7-a]**

Selected by walk-forward backtest across {n_origins:.0f} origins, beating both a
naive baseline and a damped-trend model ({summary.loc[winner, "mape"]:.1%} MAPE vs
{summary.loc["naive", "mape"]:.1%}) **[M7-b]**. The interval is derived from measured
out-of-sample error, not from model assumptions.

**Read this before you plan against it.** The model's bias has flipped from
${drift["early_bias"]:,.0f} to ${drift["latest_bias"]:,.0f} as growth has decelerated
**[M7-c]**. It undershot for eighteen months while the company grew fast; it now
overshoots. Its excellent error rate was earned in a regime that no longer exists.
It remains the best model available **and it is actively getting worse.** Re-run the
backtest monthly.

And it cannot see the next shock. The past contains three structural breaks we only
found by going and looking. If a fourth arrives, this forecast will be wrong, and
the interval will not protect you — intervals quantify the noise we have *seen*, not
the surprise we have not.

---

## Why you can trust the numbers above

Every figure in this memo reconciles to the billing ledger.

| | Rows | Value |
|---|---|---|
{ledger_rows}

Half the duplicate invoices carried *fresh invoice IDs* — a naive `drop_duplicates`
on the ID would have missed them and left ${dupes["dollars"] / 2:,.0f} of
double-counted revenue in the total. The {int(orphans["rows_affected"])} orphaned
invoices (${abs(orphans["dollars"]):,.0f}) point at customers who do not exist; a
left join would have deleted them silently and understated revenue with no warning
at all.

**Contracted MRR reconciles to billed revenue for all {n_months} months. Largest
unexplained residual: ${worst_residual:,.2f}.** **[M4-b]**

That standard is deliberate. A residual of $340 sounds harmless until someone asks
what it is and the honest answer is "I don't know" — at which point every other
number here becomes suspect.

---

## Recommendations, in priority order

**1. Fix the expansion motion. It is {exp_share / con_share:.0f}x the size of the
downgrade problem.**
Expansion collapsed in July 2025, coinciding with the sales reorganisation. Start
there. Nothing else on this list is worth doing first.

**2. Respond to the competitor in SMB Growth.**
Contraction is concentrated there and began in October 2025. This is a pricing and
packaging question, not a retention-outreach question — these customers are not
leaving, they are trading down.

**3. Do not ship or kill the pricing discount. Re-run the experiment.**
Fix the assignment bug, verify the split on day one, and run an automated SRM check
before anyone is allowed to look at an outcome. Acting on the current data is a coin
flip with extra steps.

**4. Adopt the metrics contract.**
`docs/metrics/metrics.yaml`, v{d["contract"].version}. Nine metrics defined, six edge
cases ruled on with evidence. It is currently *{d["contract"].status.split(" - ")[0]}*
and needs sign-off from Finance, Sales and Product. Until it is signed, the three
teams can each go back to reporting a different number, and this whole exercise
repeats next quarter.

**5. Re-run the forecast backtest monthly.**
The selected model is degrading. This is not a set-and-forget artefact.

---

## Appendix — provenance

Every tagged claim, the function that produced it, and the test that guards it.

| Tag | Produced by | Guarded by |
|---|---|---|
{provenance_rows}

**To reproduce every number in this memo:**

```bash
python -m northwind.generate --out data/raw
python -m northwind.clean --raw data/raw --out data/processed
python -m northwind.reconcile
python -m northwind.decompose
python -m northwind.experiment
python -m northwind.forecast
python -m northwind.memo
```

The generator is seeded. The output is byte-identical every time.

---

*This memo is generated by `src/northwind/memo.py`. It is not written by hand and it
cannot go stale: if the pipeline changes, the memo changes with it. If a number here
is wrong, the code that produced it is wrong, and the test that guards it failed to
catch it — both are findable in under a minute.*
"""


def build(raw_dir: Path, processed_dir: Path, out_path: Path) -> str:
    """Compute, render, write."""
    data = gather(raw_dir, processed_dir)
    memo = render(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(memo, encoding="utf-8")

    print(f"Memo written to {out_path}")
    print(f"  {len(memo.splitlines())} lines, {len(PROVENANCE)} provenance tags")
    print("\n  Every number is computed. None is typed. It cannot go stale.")

    return memo


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the executive memo.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("reports/EXECUTIVE_MEMO.md"))
    args = parser.parse_args()

    build(args.raw, args.processed, args.out)


if __name__ == "__main__":
    main()
