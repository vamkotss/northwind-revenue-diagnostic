# Why net revenue retention fell from 106% to 95%

**To:** CFO
**From:** Data Analyst
**Date:** 13 July 2026
**Metrics contract:** v1.0 (Proposed - pending sign-off by CFO, VP Sales, VP Product)

> Every number in this memo is generated directly from the analysis pipeline.
> None is typed by hand. Each carries a tag; the appendix maps every tag to the
> function that produced it and the test that guards it.

---

## The answer, in one paragraph

You are not losing customers. **Logo churn actually improved** over the period, by
1.3 points **[M5-c]**. What you have stopped doing is growing the
customers you already have. Expansion revenue collapsed from
19.7% to 10.4% of the cohort base,
and that single fact accounts for **87% of the entire 10.8-point
NRR decline** **[M5-b]**. A further 25% comes from SMB customers
downgrading rather than leaving. Churn is not the problem. It is the only thing
that got better.

NRR peaked at 106.1% and now stands at 95.4% **[M5-a]**.

---

## Why three teams reported three different numbers

Nobody was wrong. Nobody had written down what they meant.

- **Sales** count **logos**. Logo churn improved. They reported that retention was
  healthy, and by their definition it was.
- **Finance** count **dollars**. Dollars were haemorrhaging. They reported a crisis,
  and by their definition there was one.
- Neither team was tracking **expansion**, which is where the damage actually was.

NRR has four moving parts and the organisation was watching one of them.

We also found 614 subscriptions sitting in an unresolved
`paused` state — a status Sales counted as churn and Finance counted as active.
That single ambiguous field is worth $1,007,746 of disputed
revenue **[M3-a]**. It is now ruled on: a pause becomes churn after
60 days, a threshold **derived** from the data rather
than chosen — 89.6% of customers who ever return do so within 60 days, and beyond
that the return rate collapses below 2% a month.

---

## Where the 10.8 points went

| Driver | Was | Now | NRR impact | Share of decline |
|---|---|---|---|---|
| Expansion (upsell) | 19.7% | 10.4% | **-9.3 pts** | **87%** |
| Contraction (downgrades) | 1.4% | 4.2% | -2.7 pts | 25% |
| Churn (customers who left) | 12.2% | 10.8% | +1.3 pts | -12% |
| **Total** | | | **-10.7 pts** | **100%** |

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
| SMB | **90.7%** | 12.6% | 6.4% | 9.6% | $325,280 |
| Mid-Market | 100.9% | 9.5% | 1.3% | 11.7% | $-18,668 |
| Enterprise | 102.7% | 6.0% | 2.1% | 10.9% | $-23,638 |

The damage is **concentrated, not general** **[M5-d]**. Enterprise and Mid-Market
are holding. SMB is the problem, and within SMB it is contraction, not churn.

---

## What this analysis cannot tell you

**The Q1 2026 pricing experiment is unusable. Do not act on it.**

The naive readout says treatment retained 1.9% *less* MRR
than control, which would suggest killing the discount. That conclusion is wrong,
and it is wrong in a way that would have cost you a feature.

- **The randomisation failed.** The split is
  54%/46%, not 50/50
  (chi-square p = 3.1e-04) **[M6-a]**. The assignment code bucketed on
  a field correlated with customer size, so Enterprise customers were roughly twice
  as likely to land in treatment — and Enterprise expansion had *already* collapsed
  from the July reorg, for reasons that have nothing to do with the discount.
  Treatment did not lose. It was dealt a worse hand.
- **72 customers appear in both arms**, after a
  mid-experiment deploy re-randomised them.

After excluding the contaminated customers and stratifying by segment, the effect
is **-2.3%, 95% CI [-10.5%,
+6.1%]** **[M6-b]**. That interval spans zero and is
17% wide. **The experiment cannot distinguish a meaningful effect
from no effect at all.**

Stratification adjusts for the imbalance we *observed*. It cannot adjust for
whatever else the assignment bug correlated with — and we do not know what that was,
because if we did, we would have caught the bug. **A broken randomisation is not
repaired after the fact. It is re-run.**

---

## The forecast, and what it is worth

**13-week MRR: $16,188,892** (range $14,943,559–$18,062,845),
against $13,852,728 today. **[M7-a]**

Selected by walk-forward backtest across 16 origins, beating both a
naive baseline and a damped-trend model (4.4% MAPE vs
11.6%) **[M7-b]**. The interval is derived from measured
out-of-sample error, not from model assumptions.

**Read this before you plan against it.** The model's bias has flipped from
$-832,288 to $729,934 as growth has decelerated
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
| Raw invoice rows (billing export) | 51,587 | $174,532,024.69 |
| less: duplicates removed | 306 | $1,192,238.12 |
| less: orphans quarantined | 154 | $536,505.67 |
| **= Clean invoice rows** | **51,127** | **$172,803,280.90** |
| **Unexplained residual** | **0** | **$0.00** **[M4-a]** |

Half the duplicate invoices carried *fresh invoice IDs* — a naive `drop_duplicates`
on the ID would have missed them and left $596,119 of
double-counted revenue in the total. The 154 orphaned
invoices ($536,506) point at customers who do not exist; a
left join would have deleted them silently and understated revenue with no warning
at all.

**Contracted MRR reconciles to billed revenue for all 29 months. Largest
unexplained residual: $0.00.** **[M4-b]**

That standard is deliberate. A residual of $340 sounds harmless until someone asks
what it is and the honest answer is "I don't know" — at which point every other
number here becomes suspect.

---

## Recommendations, in priority order

**1. Fix the expansion motion. It is 3x the size of the
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
`docs/metrics/metrics.yaml`, v1.0. Nine metrics defined, six edge
cases ruled on with evidence. It is currently *Proposed*
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
| `M3-a` | `clean.resolve_pauses` | `tests/test_clean.py::test_no_ambiguous_pauses_survive` |
| `M4-a` | `reconcile.reconcile_ledger` | `tests/test_reconcile.py::test_ledger_bridge_closes` |
| `M4-b` | `reconcile.reconcile_all_months` | `tests/test_reconcile.py::test_every_month_reconciles_to_zero` |
| `M5-a` | `decompose.decompose_all_months` | `tests/test_decompose.py::test_nrr_actually_collapsed` |
| `M5-b` | `decompose.attribute_decline` | `tests/test_decompose.py::test_expansion_collapse_is_the_largest_driver` |
| `M5-c` | `decompose.decompose_all_months` | `tests/test_decompose.py::test_churn_did_not_cause_the_decline` |
| `M5-d` | `decompose.decompose_by_group` | `tests/test_decompose.py::test_the_damage_is_concentrated_in_smb` |
| `M6-a` | `experiment.check_srm` | `tests/test_experiment.py::test_srm_is_detected` |
| `M6-b` | `experiment.corrected_readout` | `tests/test_experiment.py::test_the_corrected_effect_is_indistinguishable_from_zero` |
| `M7-a` | `forecast.forecast_forward` | `tests/test_forecast.py::test_the_forecast_is_bounded_and_sane` |
| `M7-b` | `forecast.score` | `tests/test_forecast.py::test_the_winner_beats_the_naive_baseline` |
| `M7-c` | `forecast.bias_drift` | `tests/test_forecast.py::test_the_winning_models_bias_has_flipped_sign` |

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
