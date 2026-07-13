# Northwind Metrics — Revenue & Churn Diagnostic

![CI](https://github.com/vamkotss/northwind-revenue-diagnostic/actions/workflows/ci.yml/badge.svg)

## The business problem

Northwind Metrics is a B2B SaaS company (~$14M ARR, ~1,800 customers, three plan
tiers plus usage-based add-ons). Net revenue retention has fallen from 108% to 94%
over three quarters. The CFO wants to know why, and what to do about it.

The complication: nobody agrees on what "churn" means. Sales counts logo churn.
Finance counts revenue churn. Last quarter's board deck reconciles with neither —
nor with the billing ledger.

## What this repository delivers

1. A **governed metrics layer** — every metric defined and every edge case ruled
   on, signed off before any analysis runs.
2. A **reconciliation** of the analytics tables to the billing ledger, to the dollar.
3. A **decomposition** of the NRR decline by cohort, plan tier, and segment.
4. A **salvaged A/B test** — the pricing experiment was contaminated; this documents
   the detection and the defensible readout.
5. A **backtested 13-week revenue forecast** with honest error bounds.
6. An **executive memo** where every number traces back to a specific query.

## Status

In progress — Milestone 0 (scaffold + CI) complete.

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m pytest tests/ -v

## Repository layout

| Path | Purpose |
|---|---|
| `src/northwind/` | Application code |
| `data/raw/` | Generated source data — never edited by hand |
| `data/processed/` | Cleaned output — always reproducible from `raw` |
| `sql/` | Analysis queries |
| `tests/` | Automated tests, run by CI on every push |
| `docs/adr/` | Architecture Decision Records — why each choice was made |
| `docs/metrics/` | Metric definitions and edge-case rulings |
| `reports/` | Executive memo and charts |

## Stack

Python 3.13 · pandas · DuckDB · matplotlib · pytest · ruff · GitHub Actions · Power BI

## Author

Sri Vamsi Kota — [github.com/vamkotss](https://github.com/vamkotss)