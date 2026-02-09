"""Tests for RevenueForecastSkill."""

import json
import pytest
import asyncio
from pathlib import Path

from singularity.skills.revenue_forecast import (
    RevenueForecastSkill,
    _moving_average,
    _moving_average_forecast,
    _exponential_smoothing,
    _ema_forecast,
    _linear_regression,
    _linreg_forecast,
    _mae,
    _rmse,
    _mape,
    _detect_trend,
    _confidence_interval,
    _load_state,
    _save_state,
    STATE_FILE,
    DATA_DIR,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Remove state file before/after each test."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    yield
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# --- Unit tests for math functions ---

def test_moving_average():
    assert _moving_average([1, 2, 3, 4, 5], 3) == pytest.approx(4.0)
    assert _moving_average([10], 3) == pytest.approx(10.0)
    assert _moving_average([], 3) == 0.0


def test_moving_average_forecast():
    result = _moving_average_forecast([1, 2, 3], 3, window=3)
    assert len(result) == 3
    assert result[0] == pytest.approx(2.0)  # avg of [1,2,3]


def test_exponential_smoothing():
    series = [10, 12, 14, 16, 18]
    smoothed = _exponential_smoothing(series, alpha=0.5)
    assert len(smoothed) == 5
    assert smoothed[0] == 10  # first value unchanged
    assert smoothed[1] == pytest.approx(11.0)  # 0.5*12 + 0.5*10


def test_ema_forecast():
    result = _ema_forecast([10, 20, 30], 5, alpha=0.3)
    assert len(result) == 5
    # All values should be the same (flat forecast)
    assert all(v == result[0] for v in result)


def test_linear_regression():
    # Perfect linear: y = 2 + 3*x
    series = [2, 5, 8, 11, 14]
    intercept, slope = _linear_regression(series)
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(2.0)


def test_linreg_forecast():
    series = [2, 5, 8, 11, 14]  # y = 2 + 3*x
    result = _linreg_forecast(series, 3)
    assert len(result) == 3
    assert result[0] == pytest.approx(17.0)
    assert result[1] == pytest.approx(20.0)


def test_mae():
    assert _mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert _mae([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


def test_rmse():
    assert _rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert _rmse([0, 0], [1, 1]) == pytest.approx(1.0)


def test_mape():
    assert _mape([10, 20], [10, 20]) == 0.0
    assert _mape([10, 20], [12, 18]) == pytest.approx(15.0)  # (20%+10%)/2


def test_detect_trend_growth():
    series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    trend = _detect_trend(series)
    assert trend["direction"] in ("growth", "strong_growth")
    assert trend["slope"] > 0


def test_detect_trend_decline():
    series = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    trend = _detect_trend(series)
    assert trend["direction"] in ("decline", "strong_decline")
    assert trend["slope"] < 0


def test_detect_trend_reversal():
    series = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    trend = _detect_trend(series)
    assert trend["reversal"] == "growth_to_decline"


def test_confidence_interval():
    result = _confidence_interval([10.0, 12.0], std=1.0, z=1.96)
    assert len(result) == 2
    assert result[0]["forecast"] == 10.0
    assert result[0]["lower"] < 10.0
    assert result[0]["upper"] > 10.0


# --- Skill integration tests ---

def test_record_and_history():
    skill = RevenueForecastSkill()
    r1 = run(skill.execute("record", {"revenue": 10.5, "source": "test"}))
    assert r1.success
    r2 = run(skill.execute("record", {"revenue": 20.0}))
    assert r2.success

    h = run(skill.execute("history", {}))
    assert h.success
    assert h.data["manual_records"] == 2


def test_forecast_with_recorded_data():
    skill = RevenueForecastSkill()
    for val in [10, 12, 14, 16, 18, 20]:
        run(skill.execute("record", {"revenue": val}))

    result = run(skill.execute("forecast", {"periods": 3, "model": "linear_regression"}))
    assert result.success
    assert result.data["periods"] == 3
    assert len(result.data["forecast"]) == 3
    # Linear trend should predict ~22, 24, 26
    assert result.data["forecast"][0]["forecast"] > 20


def test_forecast_insufficient_data():
    skill = RevenueForecastSkill()
    run(skill.execute("record", {"revenue": 10}))
    result = run(skill.execute("forecast", {}))
    assert not result.success
    assert "2 data points" in result.message


def test_trend_action():
    skill = RevenueForecastSkill()
    for val in [5, 10, 15, 20, 25]:
        run(skill.execute("record", {"revenue": val}))

    result = run(skill.execute("trend", {}))
    assert result.success
    assert result.data["slope"] > 0


def test_breakeven_profitable():
    skill = RevenueForecastSkill()
    for val in [5, 6, 7, 8, 9, 10]:
        run(skill.execute("record", {"revenue": val}))

    result = run(skill.execute("breakeven", {"cost_per_period": 5.0}))
    assert result.success
    assert result.data["is_profitable"]
    assert result.data["status"] == "already_profitable"


def test_breakeven_converging():
    skill = RevenueForecastSkill()
    for val in [0.1, 0.2, 0.3, 0.4, 0.5]:
        run(skill.execute("record", {"revenue": val}))

    result = run(skill.execute("breakeven", {"cost_per_period": 5.0}))
    assert result.success
    assert result.data["status"] == "converging"
    assert result.data["periods_to_breakeven"] > 0


def test_scenarios():
    skill = RevenueForecastSkill()
    for val in [10, 12, 11, 13, 14, 15]:
        run(skill.execute("record", {"revenue": val}))

    result = run(skill.execute("scenarios", {"periods": 5}))
    assert result.success
    assert "optimistic" in result.data
    assert "baseline" in result.data
    assert "pessimistic" in result.data
    assert result.data["optimistic"]["total"] >= result.data["baseline"]["total"]
    assert result.data["baseline"]["total"] >= result.data["pessimistic"]["total"]


def test_backtest():
    skill = RevenueForecastSkill()
    for val in [10, 12, 14, 16, 18, 20, 22]:
        run(skill.execute("record", {"revenue": val}))

    result = run(skill.execute("backtest", {"test_ratio": 0.3}))
    assert result.success
    assert "best_model" in result.data
    assert result.data["best_model"] in ("moving_average", "exponential_smoothing", "linear_regression")


def test_stats():
    skill = RevenueForecastSkill()
    result = run(skill.execute("stats", {}))
    assert result.success
    assert "stats" in result.data


def test_unknown_action():
    skill = RevenueForecastSkill()
    result = run(skill.execute("nonexistent", {}))
    assert not result.success
    assert "Unknown action" in result.message
