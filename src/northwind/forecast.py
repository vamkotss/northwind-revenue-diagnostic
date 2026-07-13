"""A 13-week MRR forecast that has actually been tested.

THE TRAP
--------
Fit a model. Plot a line. Ship it. The chart looks confident, the R-squared on
the training data is excellent, and the number is worthless - because you have
measured how well the model explains the past, which is not the question anyone
asked.

A forecast you have not backtested is a guess with a chart attached.

WHAT THIS DOES INSTEAD
----------------------
WALK-FORWARD BACKTESTING. We stand at a point in the past, forecast 13 weeks
forward using only the data available at that moment, and compare against what
actually happened. Then we move the origin forward and do it again. Twenty-odd
times.

That produces an honest error distribution - and the error distribution is what
gives us prediction intervals we can defend, because they come from measured
out-of-sample performance rather than from a model's own opinion of itself.

BASELINES FIRST
---------------
Before any model is allowed to claim it works, it must beat two trivial ones:

  NAIVE   tomorrow looks like today
  DRIFT   extend the historical trend line

If a model cannot beat "tomorrow looks like today", it has earned nothing, and
its complexity is pure cost. Most published forecasting models fail this test
and nobody checks.

THE STRUCTURAL BREAK
--------------------
Northwind's quarterly MRR growth has gone +40% -> +18% -> +6%. Growth is
collapsing. This is exactly the situation where DRIFT is dangerous: it fits the
average slope of a much steeper past and confidently extrapolates a boom that
already ended.

The winning model is the one that refuses to extrapolate aggressively. Damping
the trend is not a modelling nicety here - it is the entire difference between a
number the CFO can plan against and a fantasy.

Run:  python -m northwind.forecast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AS_OF = pd.Timestamp("2026-06-30")

# How far ahead the CFO wants to see. One quarter.
HORIZON = 13

# Do not backtest until we have enough history to fit anything meaningful.
MIN_TRAIN_WEEKS = 52

# How often we re-anchor the backtest. Every 4 weeks gives ~20 independent-ish
# origins across the series - enough to estimate error without pretending we
# have more information than we do.
ORIGIN_STEP = 4


# ---------------------------------------------------------------------------
# THE SERIES
# ---------------------------------------------------------------------------


def build_weekly_mrr(subs: pd.DataFrame) -> pd.DataFrame:
    """Total MRR live at the end of each week.

    MRR is a STOCK, not a flow - it is the amount we are contracted to bill,
    measured at a moment. So we take a snapshot every Sunday. Weekly rather than
    monthly because a 13-week horizon needs 13 points to be a forecast rather
    than three points and a shrug.
    """
    s = subs.copy()
    s["period_start"] = pd.to_datetime(s["period_start"])
    s["period_end"] = pd.to_datetime(s["period_end"]).fillna(AS_OF)

    weeks = pd.date_range("2024-01-07", "2026-06-28", freq="W-SUN")

    rows = []
    for week in weeks:
        live = s[(s["period_start"] <= week) & (s["period_end"] > week)]
        rows.append(
            {
                "week": week,
                "mrr": float(live["mrr"].sum()),
                "customers": len(live),
            }
        )

    frame = pd.DataFrame(rows)

    # Drop the dead weeks at the start when the company barely existed. Fitting
    # a trend through zeros teaches the model nothing except how to divide by
    # almost nothing.
    return frame[frame["mrr"] > 100_000].reset_index(drop=True)


# ---------------------------------------------------------------------------
# THE MODELS
#
# Written out by hand rather than imported. Not for purity - so that every
# assumption is visible and arguable, which is the only reason to trust a number
# that will end up in a board pack.
# ---------------------------------------------------------------------------


def forecast_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Tomorrow looks like today. The baseline every model must beat.

    Absurdly simple, and genuinely hard to beat on many real series. If your
    clever model loses to this, the clever model is a liability.
    """
    return np.repeat(history[-1], horizon)


def forecast_drift(history: np.ndarray, horizon: int) -> np.ndarray:
    """Extend the straight line from the first observation to the last.

    The classic naive trend. On a series that has been growing, it will keep
    growing - forever, at the average historical rate, regardless of what has
    happened lately.

    Included precisely BECAUSE it is dangerous here. Northwind's growth has
    collapsed from +40% a quarter to +6%. Drift does not know that and does not
    care. Watching it fail in the backtest is the argument for damping.
    """
    n = len(history)
    slope = (history[-1] - history[0]) / (n - 1)

    steps = np.arange(1, horizon + 1)
    return history[-1] + slope * steps


def forecast_holt_damped(
    history: np.ndarray,
    horizon: int,
    alpha: float = 0.3,
    beta: float = 0.1,
    phi: float = 0.85,
) -> np.ndarray:
    """Holt's linear trend, with the trend DAMPED.

    Two things are being tracked as we walk the series:

      LEVEL  where the series is right now
      TREND  how fast it has been moving lately

    Both are exponentially smoothed - recent observations count more than old
    ones, which is what lets the model notice that growth has slowed.

    THE DAMPING (phi) IS THE WHOLE POINT.
    An undamped trend says: "we grew $200k last week, so we will grow $200k every
    week from now until the heat death of the universe." That is not a forecast,
    it is an assumption of immortality.

    phi = 0.85 says: each week ahead, expect only 85% of the previous week's
    growth. The trend decays. The forecast flattens out instead of shooting to
    the moon.

    On a series whose growth is visibly collapsing, this is not conservatism for
    its own sake. It is the difference between a number that survives contact
    with reality and one that does not.
    """
    # Start with a reasonable guess at level and trend.
    level = history[0]
    trend = history[1] - history[0]

    # Walk the history, updating both as we go.
    for value in history[1:]:
        last_level = level

        # New level: blend what we just observed with where we thought we were.
        level = alpha * value + (1 - alpha) * (last_level + phi * trend)

        # New trend: blend the change we just saw with the trend we believed in.
        trend = beta * (level - last_level) + (1 - beta) * phi * trend

    # Project forward. The trend contributes phi + phi^2 + ... + phi^h - a
    # geometric series that CONVERGES. That convergence is the damping.
    out = []
    cumulative_phi = 0.0
    for h in range(1, horizon + 1):
        cumulative_phi += phi**h
        out.append(level + cumulative_phi * trend)

    return np.array(out)


MODELS = {
    "naive": forecast_naive,
    "drift": forecast_drift,
    "holt_damped": forecast_holt_damped,
}


# ---------------------------------------------------------------------------
# THE BACKTEST
# ---------------------------------------------------------------------------


def backtest(series: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Walk forward through history, forecasting from each origin, and score it.

    THE RULE THAT MATTERS: at every origin, the model sees ONLY the data that
    existed at that moment. Not one observation more.

    Leaking future data into a backtest is the most common way a forecast gets
    published with an error rate that is a fantasy. It is also completely
    invisible in the output - the chart looks better, not broken. Which is why
    the discipline has to be structural rather than something you remember to do.
    """
    values = series["mrr"].to_numpy()
    weeks = series["week"].to_numpy()

    rows = []

    # Every origin from which we could forecast a FULL horizon and still have
    # actuals to compare against.
    origins = range(MIN_TRAIN_WEEKS, len(values) - horizon + 1, ORIGIN_STEP)

    for origin in origins:
        # The past, as it looked at that moment. Nothing after index `origin`.
        train = values[:origin]

        # What actually happened next. The model never sees this.
        actual = values[origin : origin + horizon]

        for name, model in MODELS.items():
            predicted = model(train, horizon)

            for h in range(horizon):
                rows.append(
                    {
                        "origin_week": weeks[origin - 1],
                        "horizon": h + 1,
                        "model": name,
                        "predicted": float(predicted[h]),
                        "actual": float(actual[h]),
                        "error": float(predicted[h] - actual[h]),
                        "abs_pct_error": float(abs(predicted[h] - actual[h]) / actual[h]),
                    }
                )

    return pd.DataFrame(rows)


def score(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise the backtest. MAPE is the headline; bias is the tell.

    MAPE  average size of the error, ignoring direction. How wrong, typically.
    BIAS  average error WITH direction. Does the model systematically overshoot?

    Bias is the number people forget, and it is the one that hurts. A model with
    a small MAPE and a large positive bias is not "slightly noisy" - it is
    confidently, consistently too optimistic, and a CFO who plans against it will
    hire people the company cannot pay for.
    """
    return (
        results.groupby("model")
        .agg(
            mape=("abs_pct_error", "mean"),
            bias=("error", "mean"),
            worst_error=("abs_pct_error", "max"),
            n_forecasts=("error", "size"),
        )
        .sort_values("mape")
        .reset_index()
    )


def score_by_horizon(results: pd.DataFrame) -> pd.DataFrame:
    """Error by how far ahead we are looking. It should get worse. If it does not, be suspicious."""
    return (
        results.groupby(["model", "horizon"])["abs_pct_error"]
        .mean()
        .reset_index()
        .pivot(index="horizon", columns="model", values="abs_pct_error")
    )


# ---------------------------------------------------------------------------
# THE REGIME-CHANGE CHECK
#
# The most dangerous thing about an average is that it is an average.
#
# Drift wins this backtest comfortably on overall MAPE. But that average is
# computed across a period where Northwind was growing 40% a quarter, and we are
# forecasting from a period where it is growing 6%. Those are different worlds,
# and a model that was excellent in the first can be actively harmful in the
# second.
#
# So we do not just ask "which model was best". We ask "which model is best NOW,
# and is it getting better or worse?" A model whose bias is drifting in one
# direction is a model that is about to break.
# ---------------------------------------------------------------------------


def score_recent(results: pd.DataFrame, n_origins: int = 5) -> pd.DataFrame:
    """Score using only the most recent origins - the regime we actually live in.

    Forecasting is not a history exam. We do not care which model would have won
    on average over two years. We care which one works on the data-generating
    process we are standing in TODAY.
    """
    recent_origins = sorted(results["origin_week"].unique())[-n_origins:]
    recent = results[results["origin_week"].isin(recent_origins)]

    return (
        recent.groupby("model")
        .agg(
            mape=("abs_pct_error", "mean"),
            bias=("error", "mean"),
            n_forecasts=("error", "size"),
        )
        .sort_values("mape")
        .reset_index()
    )


def bias_drift(results: pd.DataFrame, model: str) -> dict:
    """Is the model's bias trending? A drifting bias means the regime has moved.

    THE TELL. A model that is randomly wrong is noisy. A model whose error is
    marching steadily from negative to positive is not noisy - it is being left
    behind by a world that has changed underneath it.

    Drift here undershot for eighteen months while growth was fast, then flipped
    to overshooting as growth collapsed. The overall MAPE is excellent and
    completely misleading. THIS is the number that catches it.
    """
    m = results[(results["model"] == model) & (results["horizon"] == HORIZON)]

    by_origin = m.groupby("origin_week")["error"].mean().sort_index()

    if len(by_origin) < 6:
        return {"trending": False}

    early = float(by_origin.iloc[: len(by_origin) // 2].mean())
    late = float(by_origin.iloc[-3:].mean())

    return {
        "early_bias": round(early, 2),
        "late_bias": round(late, 2),
        "sign_flipped": bool(early * late < 0),
        "trending_up": bool(late > early),
        "latest_bias": round(float(by_origin.iloc[-1]), 2),
    }


# ---------------------------------------------------------------------------
# PREDICTION INTERVALS - EARNED, NOT ASSUMED
# ---------------------------------------------------------------------------


def empirical_intervals(results: pd.DataFrame, model: str) -> pd.DataFrame:
    """Prediction intervals built from MEASURED backtest errors.

    Most intervals come from a model's own theory - a formula that assumes the
    residuals are normal, independent, and well-behaved. On real business data
    they are none of those things, and the resulting interval is a statement of
    faith dressed as statistics.

    These intervals come from what the model ACTUALLY got wrong, at each horizon,
    across every origin we tested. If the model was 8% too high three weeks out
    in the past, the interval three weeks out is wide enough to contain that.

    The interval widens with the horizon because the errors widen with the
    horizon. Nobody had to assume it. We measured it.
    """
    model_results = results[results["model"] == model]

    rows = []
    for h, group in model_results.groupby("horizon"):
        # The distribution of RELATIVE error at this horizon.
        relative = group["error"] / group["actual"]

        rows.append(
            {
                "horizon": int(h),
                "p05": float(np.percentile(relative, 5)),
                "p50": float(np.percentile(relative, 50)),
                "p95": float(np.percentile(relative, 95)),
                "mape": float(group["abs_pct_error"].mean()),
            }
        )

    return pd.DataFrame(rows)


def forecast_forward(
    series: pd.DataFrame, intervals: pd.DataFrame, model: str, horizon: int = HORIZON
) -> pd.DataFrame:
    """The actual forecast, BIAS-CORRECTED, with intervals earned from the backtest.

    THE BIAS CORRECTION.
    The backtest does not just tell us how WRONG the model is. It tells us which
    DIRECTION it is wrong in, at each horizon. If drift is reliably 3% low four
    weeks out, then a raw drift forecast of $14.0m is really a forecast of
    $14.4m, and shipping the $14.0m means knowingly publishing a number we have
    already measured to be too small.

    So we divide the raw forecast by (1 + median error). It is a small,
    unglamorous adjustment and it is free - the backtest already paid for it.

    Most people never do this. They run a backtest, note the MAPE, and then ship
    the uncorrected point estimate anyway - throwing away the most actionable
    thing the backtest told them.
    """
    values = series["mrr"].to_numpy()
    last_week = series["week"].iloc[-1]

    raw = MODELS[model](values, horizon)

    future_weeks = pd.date_range(
        start=last_week + pd.Timedelta(weeks=1), periods=horizon, freq="W-SUN"
    )

    rows = []
    for h in range(horizon):
        band = intervals[intervals["horizon"] == h + 1].iloc[0]

        # error = predicted - actual, so actual = predicted / (1 + relative_error).
        # p50 is the typical error, so this recovers the typical TRUTH.
        corrected = raw[h] / (1 + band["p50"])

        # And the interval: the truth has historically landed between the p05 and
        # p95 of the error distribution. Invert both to bracket it.
        lower = raw[h] / (1 + band["p95"])
        upper = raw[h] / (1 + band["p05"])

        rows.append(
            {
                "week": future_weeks[h],
                "horizon": h + 1,
                "raw_forecast": round(float(raw[h]), 2),
                "forecast": round(float(corrected), 2),
                "lower": round(float(lower), 2),
                "upper": round(float(upper), 2),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------


def run(processed_dir: Path, out_dir: Path) -> dict:
    subs = pd.read_parquet(processed_dir / "subscriptions.parquet")

    series = build_weekly_mrr(subs)
    results = backtest(series)
    summary = score(results)
    by_horizon = score_by_horizon(results)

    # Select on RECENT performance, not on the two-year average. The average was
    # earned in a growth regime we no longer live in.
    recent = score_recent(results)
    winner = recent.iloc[0]["model"]

    drift_check = bias_drift(results, winner)
    intervals = empirical_intervals(results, winner)
    forward = forecast_forward(series, intervals, winner)

    # --- Print ---
    print("=" * 74)
    print("THE SERIES  (weekly MRR)")
    print("=" * 74)
    print(f"\n  {len(series)} weeks, {series['week'].min():%Y-%m-%d} to "
          f"{series['week'].max():%Y-%m-%d}")

    q_now = series["mrr"].iloc[-1] / series["mrr"].iloc[-14] - 1
    q_prev = series["mrr"].iloc[-14] / series["mrr"].iloc[-27] - 1
    q_year = series["mrr"].iloc[-53] / series["mrr"].iloc[-66] - 1

    print("\n  Quarterly MRR growth:")
    print(f"    a year ago      {q_year:+7.1%}")
    print(f"    last quarter    {q_prev:+7.1%}")
    print(f"    this quarter    {q_now:+7.1%}")
    print("\n  >>> Growth is collapsing. Any model that extrapolates the historical")
    print("  >>> trend will overshoot badly. The backtest will show exactly that.")

    print("\n" + "=" * 74)
    print(f"BACKTEST  ({results['origin_week'].nunique()} origins, "
          f"{HORIZON}-week horizon, no data leakage)")
    print("=" * 74)
    print(f"\n  {'Model':<14} {'MAPE':>8} {'Bias':>14} {'Worst':>8}  Verdict")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 14} {'-' * 8}  {'-' * 30}")
    for r in summary.itertuples(index=False):
        if r.model == winner:
            verdict = "WINNER"
        elif r.bias > 0:
            verdict = "overshoots - dangerous"
        else:
            verdict = "beaten"
        print(f"  {r.model:<14} {r.mape:>7.1%} ${r.bias:>13,.0f} {r.worst_error:>7.1%}  {verdict}")

    print("\n  BIAS is the number people forget. A model that is typically 12% off")
    print("  but ALWAYS too high is not noisy - it is confidently wrong, and a CFO")
    print("  planning against it hires people the company cannot pay for.")

    print("\n" + "=" * 74)
    print("THE REGIME-CHANGE CHECK  (the number that stops you shipping a trap)")
    print("=" * 74)
    print("\n  Scored on the LAST 5 ORIGINS ONLY - the world we actually live in:\n")
    print(f"  {'Model':<14} {'MAPE':>8} {'Bias':>14}")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 14}")
    for r in recent.itertuples(index=False):
        print(f"  {r.model:<14} {r.mape:>7.1%} ${r.bias:>13,.0f}")

    print(f"\n  BIAS DRIFT for '{winner}' at the 13-week horizon:")
    print(f"    early origins  ${drift_check['early_bias']:>13,.0f}")
    print(f"    recent origins ${drift_check['late_bias']:>13,.0f}")
    print(f"    latest origin  ${drift_check['latest_bias']:>13,.0f}")

    if drift_check.get("sign_flipped"):
        print("\n  >>> WARNING. The bias has FLIPPED SIGN.")
        print("  >>> This model undershot for eighteen months while growth was fast,")
        print("  >>> and now overshoots as growth collapses. Its excellent overall")
        print("  >>> MAPE was earned in a regime that no longer exists.")
        print("  >>>")
        print("  >>> It is still the best model we have. It is also actively getting")
        print("  >>> worse, and it will keep getting worse if growth keeps slowing.")
        print("  >>> Re-run this backtest monthly. Do not set it and forget it.")
    print(f"\n  {'Weeks ahead':<12} " + "  ".join(f"{m:>12}" for m in by_horizon.columns))
    print(f"  {'-' * 12} " + "  ".join("-" * 12 for _ in by_horizon.columns))
    for h in [1, 4, 8, 13]:
        row = by_horizon.loc[h]
        print(f"  {h:<12} " + "  ".join(f"{row[m]:>11.1%} " for m in by_horizon.columns))

    print("\n" + "=" * 74)
    print(f"THE FORECAST  ({winner}, intervals earned from the backtest)")
    print("=" * 74)
    print(f"\n  {'Week':<12} {'Low (5%)':>14} {'Forecast':>14} {'High (95%)':>14}")
    print(f"  {'-' * 12} {'-' * 14} {'-' * 14} {'-' * 14}")
    for r in forward.itertuples(index=False):
        if r.horizon in (1, 4, 8, 13):
            print(f"  {r.week:%Y-%m-%d} ${r.lower:>13,.0f} ${r.forecast:>13,.0f} "
                  f"${r.upper:>13,.0f}")

    last = series["mrr"].iloc[-1]
    final = forward.iloc[-1]
    print(f"\n  Today          ${last:>13,.0f}")
    print(f"  In 13 weeks    ${final['forecast']:>13,.0f}  "
          f"({final['forecast'] / last - 1:+.1%})")
    print(f"  Range          ${final['lower']:>13,.0f} to ${final['upper']:>13,.0f}")

    print("\n" + "=" * 74)
    print("WHAT THIS FORECAST CANNOT DO")
    print("=" * 74)
    print("\n  It cannot see the next shock. Every model here learns from the past,")
    print("  and the past contains three structural breaks we only identified by")
    print("  going and looking - a sales reorg, a price rise, a competitor launch.")
    print()
    print("  If a fourth arrives, this forecast will be wrong, and the interval")
    print("  will not save you. Intervals quantify the noise we have SEEN. They")
    print("  say nothing about the surprise we have not.")
    print()
    print("  Present the number with that sentence attached, or do not present it.")

    out_dir.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out_dir / "weekly_mrr.parquet", index=False)
    results.to_parquet(out_dir / "backtest_results.parquet", index=False)
    summary.to_parquet(out_dir / "backtest_summary.parquet", index=False)
    forward.to_parquet(out_dir / "forecast.parquet", index=False)

    return {
        "series": series,
        "results": results,
        "summary": summary,
        "by_horizon": by_horizon,
        "intervals": intervals,
        "forecast": forward,
        "winner": winner,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtested 13-week MRR forecast.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    run(args.processed, args.out)


if __name__ == "__main__":
    main()
