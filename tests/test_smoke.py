"""Smoke tests: prove the environment and package imports actually work.

A "smoke test" is the cheapest possible test. It does not check that your
logic is correct - it checks that the thing turns on without catching fire.
"""

import sys

import duckdb
import numpy as np
import pandas as pd
import yaml


def test_python_version():
    """Confirm we are on Python 3.11 or newer."""
    # sys.version_info is a tuple like (3, 13, 7). We compare the first parts.
    assert sys.version_info >= (3, 11), "Python 3.11+ required"


def test_core_libraries_are_importable():
    """Confirm each core library imported AND is usable.

    Reading each library's version proves the import worked and actually USES
    the module - a bare unused import is dead code, and ruff is right to flag it.
    """
    assert pd.__version__      # pandas: tables of data
    assert np.__version__      # numpy: fast math
    assert duckdb.__version__  # duckdb: our SQL engine
    assert yaml.__version__    # pyyaml: reads our metric definition files


def test_duckdb_can_run_sql():
    """Confirm DuckDB actually executes SQL, not just imports."""
    # Open an in-memory database. Nothing is written to disk.
    con = duckdb.connect()

    # fetchone() gets the first row; [0] pulls the first column from that row.
    result = con.execute("SELECT 1 + 1 AS answer").fetchone()[0]

    con.close()
    assert result == 2, "DuckDB failed basic arithmetic"


def test_duckdb_can_query_a_pandas_dataframe():
    """Confirm DuckDB can run SQL against a pandas DataFrame.

    This is the core workflow of Project 1: data lives in a DataFrame, and we
    query it with SQL. If this breaks, nothing else in the project works.

    We REGISTER the DataFrame explicitly rather than relying on DuckDB's magic
    variable lookup. Explicit is better: rename the variable and you get a clear
    error instead of a silent failure - and the linter can actually see it.
    """
    # Build a tiny two-row table in memory.
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "mrr":         [100.0, 250.0],   # MRR = Monthly Recurring Revenue
    })

    con = duckdb.connect()

    # Hand the DataFrame to DuckDB under the SQL table name "customers".
    # THIS is the line that makes the dependency explicit and visible.
    con.register("customers", df)

    # Now query it as if it were a normal database table.
    total = con.execute("SELECT SUM(mrr) AS total FROM customers").fetchone()[0]

    con.close()
    assert total == 350.0, "DuckDB could not read the registered DataFrame"
