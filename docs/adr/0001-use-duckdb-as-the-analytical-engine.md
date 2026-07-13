# ADR 0001 — Use DuckDB as the analytical engine

**Status:** Accepted
**Date:** 2026-07-13

## Context

Project 1 requires SQL over roughly a million rows of subscription, invoice, and
usage data. The analysis must be reproducible by anyone who clones this repository,
and it must run in a CI job on a fresh machine with no manual setup.

## Decision

Use DuckDB, an in-process analytical database, with data stored as Parquet files.

## Alternatives considered

**PostgreSQL.** The industry default, and a skill worth having. Rejected here
because it needs a running server, credentials, and a Docker Compose file before
anyone can execute a single query. That friction is real: it means a reviewer
cannot clone the repo and reproduce the numbers in under a minute, and CI would
have to spin up a database service. Postgres is used in Project 2, where the
orchestration and modelling work justifies the setup cost.

**SQLite.** Zero setup, same as DuckDB. Rejected because it is row-oriented and
optimised for transactions, not analysis. Window functions over cohorts — the
core of this project — are markedly slower, and its type system is loose enough
to hide exactly the data-quality defects this project is designed to surface.

**pandas alone.** Rejected on purpose. The primary hiring signal for a Data
Analyst role is SQL. Doing cohort retention with `groupby` chains would hide the
skill the portfolio exists to demonstrate.

## Consequences

**Positive.** Anyone can clone and reproduce every number with `pip install`.
CI runs the full analysis with no external services. Columnar execution makes
window functions over cohorts fast. DuckDB queries pandas DataFrames directly,
so the Python and SQL layers interoperate cleanly.

**Negative.** DuckDB is single-process — it does not demonstrate concurrency,
connection pooling, or server administration. Those skills are demonstrated in
Project 2 (Postgres + Airflow) instead. Some employers screen for "PostgreSQL"
by keyword; the README and this ADR name the trade-off explicitly so it reads
as a decision rather than an omission.