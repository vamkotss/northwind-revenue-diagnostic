# Building the Northwind dashboard in Power BI

You have eight CSVs in `data/powerbi/`. This document turns them into a
dashboard a CFO would act on.

Read the trap section first. It is the thing that breaks most SaaS dashboards,
it produces no error message, and it will not be obvious that anything is wrong.

---

## The trap: MRR is a stock, not a flow

Revenue is a **flow**. If you bill $1m in January and $1m in February, you have
billed $2m. Adding them up is correct.

MRR is a **stock**. It is the amount you are contracted to bill *at a moment in
time*, like a bank balance. If your MRR is $13m in January and $13m in February,
your MRR is **$13m** — not $26m.

Now: `fact_mrr` has one row per customer per month. So this measure —

```dax
Total MRR = SUM(fact_mrr[mrr])
```

— is correct when a single month is selected, and **catastrophically wrong** the
moment someone clicks "2026" in the year slicer. It will cheerfully report an
MRR of $80m and Power BI will not warn you, because nothing is broken. You asked
it to add up a column and it added up the column.

Statisticians call this a **semi-additive measure**: it adds up across customers,
but not across time. The fix is to tell Power BI to only look at the last date in
whatever period is selected.

```dax
Total MRR =
CALCULATE(
    SUM( fact_mrr[mrr] ),
    LASTDATE( dim_date[date] )
)
```

Every SaaS dashboard with a suspiciously large MRR figure has this bug. Now
yours does not.

---

## Step 1 — Load the data

1. Open **Power BI Desktop**
2. **Home** → **Get data** → **Text/CSV**
3. Import all eight files from `data/powerbi/`:
   - `dim_date.csv`
   - `dim_customer.csv`
   - `dim_plan.csv`
   - `fact_mrr.csv`
   - `fact_movement.csv`
   - `fact_forecast.csv`
   - `fact_reconciliation_ledger.csv`
   - `fact_reconciliation_monthly.csv`
4. Click **Load** (not Transform — the cleaning already happened in Python,
   where it is tested and version-controlled, which is where it belongs)

---

## Step 2 — Mark the date table

**Do not skip this.** Power BI's time intelligence functions do not work without
it, and they fail *silently* — you get blanks, not errors.

1. In the left sidebar, click the **Table view** icon
2. Select **dim_date**
3. **Table tools** ribbon → **Mark as date table**
4. Choose the **`date`** column → **OK**

---

## Step 3 — Build the relationships

Click the **Model view** icon (third one down the left sidebar). Drag to create
these five relationships:

| From (fact) | To (dimension) | Cardinality |
|---|---|---|
| `fact_mrr[date_key]` | `dim_date[date_key]` | Many-to-one |
| `fact_mrr[customer_id]` | `dim_customer[customer_id]` | Many-to-one |
| `fact_mrr[plan_tier]` | `dim_plan[plan_tier]` | Many-to-one |
| `fact_movement[date_key]` | `dim_date[date_key]` | Many-to-one |
| `fact_forecast[date_key]` | `dim_date[date_key]` | Many-to-one |

**Every relationship must be Many-to-one, single direction, filtering from the
dimension into the fact.** That is what a star schema is. If Power BI offers you
"both directions", say no — bidirectional filters create ambiguity, and ambiguity
in a data model shows up as totals that are wrong in ways nobody can reproduce.

The reconciliation tables stay unrelated. They are a proof, not a slice.

---

## Step 4 — Sort the labels correctly

Two columns will sort alphabetically unless you tell them otherwise, and
alphabetical is wrong for both.

**Month labels** (otherwise "Apr" comes before "Jan"):
1. Table view → **dim_date** → select the **`year_month`** column
2. **Column tools** → **Sort by column** → **`year_month_sort`**

**Plan tiers** (otherwise Enterprise, the most expensive, appears first):
1. Table view → **dim_plan** → select the **`plan_tier`** column
2. **Column tools** → **Sort by column** → **`tier_rank`**

---

## Step 5 — The measures

Create these in **Modeling** → **New measure**. Paste one at a time.

### Core

```dax
Total MRR =
-- MRR is a STOCK, not a flow. LASTDATE takes the value at the END of whatever
-- period is selected, instead of summing every month inside it.
-- Remove LASTDATE and a full-year selection reports 12x the real MRR.
CALCULATE(
    SUM( fact_mrr[mrr] ),
    LASTDATE( dim_date[date] )
)
```

```dax
Total ARR =
-- Annual Recurring Revenue. MRR times twelve. Nothing more.
[Total MRR] * 12
```

```dax
Active Customers =
-- Semi-additive for the same reason: a customer active in Jan and Feb is ONE
-- customer, not two. DISTINCTCOUNT alone would still overcount across months,
-- so we pin to the last date as well.
CALCULATE(
    DISTINCTCOUNT( fact_mrr[customer_id] ),
    LASTDATE( dim_date[date] )
)
```

```dax
Average MRR per Customer =
-- DIVIDE, not "/". DIVIDE returns blank on a zero denominator instead of an
-- error, so a slicer selection with no customers shows an empty cell rather
-- than breaking the whole visual.
DIVIDE( [Total MRR], [Active Customers] )
```

### Retention

`fact_movement` already contains NRR, GRR, and the three component rates,
computed and tested in Python. These measures surface them.

**Why not compute NRR in DAX?** Because the definition lives in
`docs/metrics/metrics.yaml`, the Python implements it, and 112 tests prove the
implementation is right. Re-implementing that logic in DAX would create a second
definition that can silently drift from the first — which is the exact disease
this whole project was built to cure. The dashboard reads the number. It does not
invent its own.

```dax
NRR =
-- Net Revenue Retention. Above 100% means the customers you kept grew.
-- Below 100% means they shrank.
CALCULATE(
    SUM( fact_movement[nrr] ),
    LASTDATE( dim_date[date] )
)
```

```dax
GRR =
-- Gross Revenue Retention. Ignores expansion, so it can never exceed 100%.
-- The honest floor.
CALCULATE(
    SUM( fact_movement[grr] ),
    LASTDATE( dim_date[date] )
)
```

```dax
Churn Rate =
CALCULATE( SUM( fact_movement[churn_rate] ), LASTDATE( dim_date[date] ) )
```

```dax
Contraction Rate =
-- Downgrades. NOT churn - see ruling R3. The customer is still a customer.
CALCULATE( SUM( fact_movement[contraction_rate] ), LASTDATE( dim_date[date] ) )
```

```dax
Expansion Rate =
-- The one that collapsed. 87% of the NRR decline lives here.
CALCULATE( SUM( fact_movement[expansion_rate] ), LASTDATE( dim_date[date] ) )
```

### Growth

```dax
MRR Last Year =
-- SAMEPERIODLASTYEAR is a time-intelligence function. It ONLY works because
-- dim_date is marked as the date table in Step 2. Skip that step and this
-- returns blank, with no error, forever.
CALCULATE(
    [Total MRR],
    SAMEPERIODLASTYEAR( dim_date[date] )
)
```

```dax
MRR YoY % =
DIVIDE(
    [Total MRR] - [MRR Last Year],
    [MRR Last Year]
)
```

### Formatting

Select each measure, then use the **Measure tools** ribbon:

| Measure | Format |
|---|---|
| Total MRR, Total ARR, Average MRR per Customer | Currency, 0 decimals |
| NRR, GRR, Churn/Contraction/Expansion Rate, MRR YoY % | Percentage, 1 decimal |
| Active Customers | Whole number |

---

## Step 6 — The pages

Four pages. Each answers one question. Resist the urge to add a fifth.

### Page 1 — "The Answer"

The CFO asked one question. This page answers it before they finish reading the
title.

**Title:** `Why NRR fell from 106% to 95%`

- **Four cards across the top:** `Total MRR`, `NRR`, `Active Customers`, `MRR YoY %`
- **Line chart:** `year_month` on the X axis, `NRR` and `GRR` as two lines.
  The gap between them *is* the expansion contribution — when the lines converge,
  expansion is dying. Add a reference line at 100%.
- **Stacked column:** `year_month` on X; `churned_mrr`, `contraction_mrr`,
  `expansion_mrr` from `fact_movement` as the values. This is the waterfall. It
  shows expansion (the positive bar) shrinking while the negative bars hold
  roughly steady.
- **A text box.** Write the finding in words:

  > Churn improved. NRR still collapsed. The cause is a 9.3-point collapse in
  > expansion revenue — 87% of the total decline — beginning with the July 2025
  > sales reorg. Contraction from SMB downgrades accounts for a further 2.7
  > points, starting when the competitor launched in October.

A dashboard that makes the reader work out the finding for themselves has failed.
Say it.

### Page 2 — "Who"

- **Slicers:** `segment`, `plan_tier`, `acquisition_channel`, `signup_cohort`
- **Matrix:** rows = `segment`, columns = `year_month`, values = `NRR`.
  Conditional-format the background red-to-green. The SMB row will be visibly red.
- **Bar chart:** `segment` on the axis, `Total MRR` as the value, `plan_tier` as
  the legend
- **Scatter:** `Churn Rate` on X, `Expansion Rate` on Y, one dot per segment,
  bubble size = `Total MRR`. Segments in the bottom-right are dying quietly.

### Page 3 — "The Forecast"

- **Line chart** from `fact_forecast`: `week` on X; plot `actual`, `forecast`,
  `lower`, and `upper`. Set `lower` and `upper` to a dashed line style.
- **A text box.** This one is not optional:

  > 13-week forecast: $16.2m (range $14.9m–$18.1m). Selected by walk-forward
  > backtest across 20 origins; the interval is derived from measured
  > out-of-sample error, not model assumptions.
  >
  > **Warning:** the model's bias has flipped from −$832k to +$730k as growth
  > decelerated. It is currently the best available and it is actively degrading.
  > Re-run the backtest monthly. This forecast cannot see the next structural
  > break, and the interval will not protect you from one.

### Page 4 — "Does it tie?"

The page that makes the other three believable.

- **Table** from `fact_reconciliation_ledger`: the raw-to-clean bridge. Every
  invoice accounted for.
- **Table** from `fact_reconciliation_monthly`: `month`, `contracted_mrr`,
  `less_not_invoiced`, `plus_invoiced_not_active`, `plus_refunds_base`,
  `billed_base`, **`residual`**
- **Card:** `Max Residual = MAX( fact_reconciliation_monthly[residual] )`.
  It reads **$0.00**.
- **A text box:**

  > Every month in this dashboard reconciles to the billing ledger to the cent.
  > The bridge is above. Unexplained residual across all 29 months: $0.00.

Most dashboards ask to be trusted. This one shows its working.

---

## Step 7 — Publish

**Home** → **Publish** → select a workspace.

Then take a screenshot of Page 1, put it at the top of your README, and link the
published report. **A hiring manager will look at the screenshot for four seconds
and decide whether to open the repo.** Make those four seconds count: the finding
should be legible without clicking anything.

---

## What to say about this in an interview

Not *"I built a Power BI dashboard."* Everyone says that.

> *"The data model is a star schema, so the measures are semi-additive — MRR is a
> stock, so summing it across months would have reported twelve times the real
> number. The retention definitions are not re-implemented in DAX; they come from
> a versioned metrics contract with 112 tests behind it, so the dashboard cannot
> drift from the source of truth. And there is a reconciliation page showing the
> whole thing ties to the billing ledger to the cent."*

That is a different conversation from the one everyone else is having.
