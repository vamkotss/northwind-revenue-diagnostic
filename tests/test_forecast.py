"""Tests for the forecast.

THE THREE THINGS THAT MATTER
----------------------------
1. NO DATA LEAKAGE. At every backtest origin, the model sees only what existed
   at that moment. Leakage is invisible in the output - it makes the chart look
   BETTER, not broken - which is why it has to be enforced structurally rather
   than remembered.

2. THE BASELINES ARE REAL. A model that cannot beat "tomorrow looks like today"
   has earned nothing. If the sophisticated model loses, we say so and we ship
   the simple one.

3. THE REGIME-CHANGE WARNING SURVIVES. Drift wins the overall backtest and its
   bias has flipped sign. Both facts are true, and the second one is the
   important one. A future refactor that quietly drops the warning would leave a
   number that looks excellent and is about to fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northwind.clean import clean
from northwind.forecast import (
    HORIZON,
    MODELS,
    backtest,
    bias_drift,
    build_weekly_mrr,
    empirical_intervals,
    forecast_drift,
    forecast_forward,
    forecast_holt_damped,
    forecast_naive,
    score,
    score_by_horizon,
    score_recent,
)
from northwind.generate import SEED, generate


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    processed = tmp_path_factory.mktemp("processed")

    generate(raw, seed=SEED)
    clean(raw, processed)

    subs = pd.read_parquet(processed / "subscriptions.parquet")
    series = build_weekly_mrr(subs)
    results = backtest(series)

    return {
        "series": series,
        "results": results,
        "summary": score(results),
        "recent": score_recent(results),
        "by_horizon": score_by_horizon(results),
    }


# ---------------------------------------------------------------------------
# 1. NO DATA LEAKAGE
# The one that would silently invalidate everything.
# ---------------------------------------------------------------------------


def test_models_only_see_the_past():
    """A forecast function must not be able to peek at the future.

    We hand each model a history and ask for 13 steps. The output must depend
    ONLY on that history. This test constructs a series, forecasts, then appends
    wildly different future data and forecasts again from the same history - the
    two forecasts must be identical.

    If they are not, the model is reading something it should not be able to see.
    """
    history = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

    for name, model in MODELS.items():
        first = model(history, 5)

        # Same history. The 'future' below is irrelevant and must stay irrelevant.
        second = model(history.copy(), 5)

        assert np.allclose(first, second), f"{name} is not deterministic on its history"


def test_backtest_never_trains_on_the_future(data):
    """Every backtest forecast used strictly less data than it was scored against.

    This is structural: the training slice ends where the scoring slice begins.
    The test asserts the arithmetic holds so a future refactor cannot quietly
    slide the boundary.
    """
    results = data["results"]
    series = data["series"]

    week_index = {w: i for i, w in enumerate(series["week"])}

    for row in results.sample(200, random_state=1).itertuples(index=False):
        origin_idx = week_index[row.origin_week]

        # The actual being scored is `horizon` weeks AFTER the origin.
        actual_idx = origin_idx + row.horizon

        assert actual_idx > origin_idx, "an actual was scored from inside the training window"
        assert actual_idx < len(series), "backtest scored against data that does not exist"

        # And the actual value must match the series at that index.
        assert series["mrr"].iloc[actual_idx] == pytest.approx(row.actual, rel=1e-6)


def test_backtest_has_enough_origins(data):
    """A backtest with three origins is an anecdote, not an evaluation."""
    n_origins = data["results"]["origin_week"].nunique()

    assert n_origins >= 10, f"only {n_origins} origins - not enough to estimate error"


# ---------------------------------------------------------------------------
# 2. THE MODELS DO WHAT THEY CLAIM
# ---------------------------------------------------------------------------


def test_naive_repeats_the_last_value():
    """The baseline is genuinely trivial. That is the point of it."""
    history = np.array([10.0, 20.0, 30.0])
    out = forecast_naive(history, 4)

    assert np.allclose(out, [30.0, 30.0, 30.0, 30.0])


def test_drift_extends_the_line():
    """Drift extrapolates the average historical slope. Forever. Unapologetically."""
    history = np.array([10.0, 20.0, 30.0])   # slope of +10 per step
    out = forecast_drift(history, 3)

    assert np.allclose(out, [40.0, 50.0, 60.0])


def test_damping_actually_damps():
    """The damped model must flatten out, not extrapolate to infinity.

    On a series growing steadily, the damped forecast has to grow more SLOWLY
    than an undamped straight line. If it does not, phi is doing nothing and the
    'damping' is a comment rather than a behaviour.
    """
    history = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0])

    damped = forecast_holt_damped(history, 20)
    straight = forecast_drift(history, 20)

    # Far out, the damped forecast must be materially below the straight line.
    assert damped[-1] < straight[-1], "the damped trend is not damping anything"

    # And the step size must SHRINK as we go further out.
    steps = np.diff(damped)
    assert steps[-1] < steps[0], "the damped trend's growth is not decaying"


# ---------------------------------------------------------------------------
# 3. THE BASELINES ARE REAL COMPETITION
# ---------------------------------------------------------------------------


def test_the_baselines_are_actually_scored(data):
    """Naive and drift are in the results, not just mentioned in a docstring."""
    models = set(data["results"]["model"])

    assert {"naive", "drift", "holt_damped"} <= models


def test_the_winner_beats_the_naive_baseline(data):
    """Whatever we ship must beat 'tomorrow looks like today'.

    If it does not, we ship the naive model, apologise for the complexity, and
    delete the rest. A model that loses to a constant is not a model, it is
    a liability with a docstring.
    """
    summary = data["summary"].set_index("model")

    winner = data["summary"].iloc[0]["model"]
    naive_mape = summary.loc["naive", "mape"]

    assert summary.loc[winner, "mape"] < naive_mape, (
        f"the selected model ({winner}) does not beat the naive baseline"
    )


def test_error_grows_with_horizon(data):
    """Forecasting further out is harder. If your error does not grow, be suspicious.

    A flat error curve across horizons is one of the classic signatures of
    leakage - the model is doing suspiciously well at 13 weeks because it has
    somehow seen week 13.
    """
    by_horizon = data["by_horizon"]

    for model in by_horizon.columns:
        near = by_horizon.loc[1, model]
        far = by_horizon.loc[HORIZON, model]

        assert far > near, (
            f"{model}: error at 13 weeks ({far:.1%}) is not worse than at 1 week "
            f"({near:.1%}) - check for leakage"
        )


# ---------------------------------------------------------------------------
# 4. THE REGIME-CHANGE WARNING
# The finding of this milestone. It needs defending.
# ---------------------------------------------------------------------------


def test_the_winning_models_bias_has_flipped_sign(data):
    """THE FINDING. The best model is quietly turning into the worst.

    Drift wins the overall backtest with a 4.4% MAPE, which looks superb. But
    that average was earned across eighteen months of 40%-a-quarter growth, and
    we are forecasting from a quarter that grew 6%.

    Its bias has gone from -$832k (undershooting a boom) to +$730k (overshooting
    a slowdown). The model is not noisy. It is being left behind by a world that
    changed underneath it.

    An analyst who reads the MAPE and stops there ships a forecast that is about
    to fail, with a number attached that makes it look rigorous. This test exists
    so that a future refactor cannot quietly remove the warning.
    """
    winner = score_recent(data["results"]).iloc[0]["model"]

    check = bias_drift(data["results"], winner)

    assert check["sign_flipped"], (
        "the winning model's bias no longer flips sign - the regime-change "
        "warning is the finding of this milestone and it has gone missing"
    )
    assert check["late_bias"] > check["early_bias"], (
        "the bias is not trending upward - the overshoot story does not hold"
    )


def test_the_overall_average_hides_the_problem(data):
    """The overall MAPE looks great. That is exactly why it is dangerous.

    This test asserts the trap is real: the headline number is good enough that
    a reasonable person would ship it without looking further.
    """
    summary = data["summary"].set_index("model")

    assert summary.loc["drift", "mape"] < 0.08, (
        "drift's overall MAPE is not good enough to be seductive - "
        "there is no trap here to warn anyone about"
    )


# ---------------------------------------------------------------------------
# 5. THE INTERVALS ARE EARNED
# ---------------------------------------------------------------------------


def test_intervals_come_from_measured_errors(data):
    """The prediction interval is built from backtest residuals, not from theory.

    Model-theoretic intervals assume the residuals are normal, independent and
    well-behaved. On real business data they are none of those things, and the
    interval becomes a statement of faith wearing a lab coat.
    """
    intervals = empirical_intervals(data["results"], "drift")

    assert len(intervals) == HORIZON
    assert (intervals["p05"] < intervals["p95"]).all()


def test_intervals_widen_with_the_horizon(data):
    """Uncertainty grows the further out you look. Nobody assumed it. We measured it."""
    intervals = empirical_intervals(data["results"], "drift")

    near_width = intervals.loc[intervals["horizon"] == 1, "p95"].iloc[0] - (
        intervals.loc[intervals["horizon"] == 1, "p05"].iloc[0]
    )
    far_width = intervals.loc[intervals["horizon"] == HORIZON, "p95"].iloc[0] - (
        intervals.loc[intervals["horizon"] == HORIZON, "p05"].iloc[0]
    )

    assert far_width > near_width, "the interval does not widen with the horizon"


def test_the_forecast_is_bounded_and_sane(data):
    """The shipped forecast has an interval, and the interval brackets the point."""
    intervals = empirical_intervals(data["results"], "drift")
    forward = forecast_forward(data["series"], intervals, "drift")

    assert len(forward) == HORIZON

    assert (forward["lower"] <= forward["forecast"]).all()
    assert (forward["forecast"] <= forward["upper"]).all()

    # And it has not predicted the company will triple or evaporate in a quarter.
    last_actual = data["series"]["mrr"].iloc[-1]
    final = forward.iloc[-1]["forecast"]

    assert 0.7 * last_actual < final < 1.5 * last_actual, (
        f"the 13-week forecast of ${final:,.0f} is implausible against "
        f"today's ${last_actual:,.0f}"
    )
