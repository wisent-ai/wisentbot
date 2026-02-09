#!/usr/bin/env python3
"""
RevenueForecastSkill - Time-series revenue forecasting with multiple models.

The agent tracks revenue through many subsystems (TaskPricing, HTTPBridge,
DatabaseBridge, Marketplace, etc.) and records snapshots in
RevenueAnalyticsDashboardSkill. But there is NO forward-looking capability:
the agent cannot predict future revenue, detect trend changes, estimate
when it will become profitable, or plan resource allocation.

This skill fills that critical gap:
1. Reads historical revenue snapshots from revenue_analytics_dashboard.json
2. Applies multiple forecasting models (moving average, exponential smoothing,
   linear regression) to project future revenue
3. Calculates confidence intervals for forecasts
4. Detects trend direction changes (growth → decline, etc.)
5. Estimates break-even timeline (when revenue > costs)
6. Provides scenario analysis (optimistic / baseline / pessimistic)
7. Compares model accuracy via backtesting to pick the best forecaster

Data source:
  - revenue_analytics_dashboard.json → snapshots[] (each with total_revenue, timestamp)
  - Optionally: metric_snapshots from revenue_observability_bridge.json

Actions:
  - forecast: Generate N-period revenue forecast with confidence intervals
  - trend: Detect current trend direction and strength
  - breakeven: Estimate time to break-even given cost rate
  - scenarios: Optimistic/baseline/pessimistic projections
  - backtest: Compare model accuracy on historical data
  - record: Manually record a revenue data point
  - history: View recorded revenue time series
  - stats: Forecasting accuracy and usage statistics

Pillar: Revenue (primary) + Goal Setting (data-driven planning)
"""

import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "revenue_forecast.json"
DASHBOARD_FILE = DATA_DIR / "revenue_analytics_dashboard.json"
BRIDGE_FILE = DATA_DIR / "revenue_observability_bridge.json"

MAX_HISTORY = 500
MAX_FORECASTS = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


def _load_state() -> Dict:
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {
        "time_series": [],         # [{timestamp, revenue, source}]
        "forecasts": [],           # past forecasts for accuracy tracking
        "backtest_results": {},    # model_name -> {mae, rmse, mape}
        "stats": {
            "total_forecasts": 0,
            "total_backtests": 0,
            "total_trend_checks": 0,
            "total_breakeven_checks": 0,
            "best_model": "exponential_smoothing",
        },
        "config": {
            "default_periods": 7,
            "default_model": "exponential_smoothing",
            "ema_alpha": 0.3,        # smoothing factor for EMA
            "confidence_level": 0.9,
            "cost_per_period": 1.0,  # USD per period for breakeven calc
        },
    }


def _save_state(state: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(state.get("time_series", [])) > MAX_HISTORY:
        state["time_series"] = state["time_series"][-MAX_HISTORY:]
    if len(state.get("forecasts", [])) > MAX_FORECASTS:
        state["forecasts"] = state["forecasts"][-MAX_FORECASTS:]
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except IOError:
        pass


def _load_dashboard_snapshots() -> List[Dict]:
    """Load revenue snapshots from the analytics dashboard."""
    try:
        if DASHBOARD_FILE.exists():
            data = json.loads(DASHBOARD_FILE.read_text())
            return data.get("snapshots", [])
    except (json.JSONDecodeError, IOError):
        pass
    return []


def _extract_revenue_series(snapshots: List[Dict]) -> List[float]:
    """Extract total_revenue values from dashboard snapshots."""
    series = []
    for snap in snapshots:
        rev = snap.get("total_revenue") or snap.get("revenue", {}).get("total", 0)
        try:
            series.append(float(rev))
        except (ValueError, TypeError):
            series.append(0.0)
    return series


# ---------------------------------------------------------------------------
# Forecasting models
# ---------------------------------------------------------------------------

def _moving_average(series: List[float], window: int = 3) -> float:
    """Simple moving average of last `window` points."""
    if not series:
        return 0.0
    w = min(window, len(series))
    return sum(series[-w:]) / w


def _moving_average_forecast(series: List[float], periods: int, window: int = 3) -> List[float]:
    """Forecast using rolling moving average."""
    extended = list(series)
    for _ in range(periods):
        val = _moving_average(extended, window)
        extended.append(val)
    return extended[len(series):]


def _exponential_smoothing(series: List[float], alpha: float = 0.3) -> List[float]:
    """Single exponential smoothing (SES). Returns smoothed series."""
    if not series:
        return []
    smoothed = [series[0]]
    for i in range(1, len(series)):
        s = alpha * series[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(s)
    return smoothed


def _ema_forecast(series: List[float], periods: int, alpha: float = 0.3) -> List[float]:
    """Forecast using exponential smoothing (flat forecast from last level)."""
    if not series:
        return [0.0] * periods
    smoothed = _exponential_smoothing(series, alpha)
    last_level = smoothed[-1]
    return [last_level] * periods


def _linear_regression(series: List[float]) -> Tuple[float, float]:
    """Fit y = a + b*x via OLS. Returns (intercept, slope)."""
    n = len(series)
    if n < 2:
        return (series[0] if series else 0.0, 0.0)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(series) / n
    ss_xy = sum((xs[i] - x_mean) * (series[i] - y_mean) for i in range(n))
    ss_xx = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if ss_xx == 0:
        return (y_mean, 0.0)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return (intercept, slope)


def _linreg_forecast(series: List[float], periods: int) -> List[float]:
    """Forecast using linear regression extrapolation."""
    if not series:
        return [0.0] * periods
    intercept, slope = _linear_regression(series)
    n = len(series)
    return [intercept + slope * (n + i) for i in range(periods)]


def _residual_std(series: List[float], fitted: List[float]) -> float:
    """Compute standard deviation of residuals."""
    if len(series) < 2:
        return 0.0
    residuals = [series[i] - fitted[i] for i in range(min(len(series), len(fitted)))]
    n = len(residuals)
    if n < 2:
        return 0.0
    mean_r = sum(residuals) / n
    var = sum((r - mean_r) ** 2 for r in residuals) / (n - 1)
    return math.sqrt(var) if var > 0 else 0.0


def _confidence_interval(forecast: List[float], std: float, z: float = 1.645) -> List[Dict]:
    """Add confidence intervals to forecast values. z=1.645 for 90%."""
    result = []
    for i, val in enumerate(forecast):
        # Wider intervals further into the future
        spread = std * z * math.sqrt(i + 1)
        result.append({
            "period": i + 1,
            "forecast": round(val, 4),
            "lower": round(val - spread, 4),
            "upper": round(val + spread, 4),
        })
    return result


# Z-scores for common confidence levels
Z_SCORES = {0.80: 1.282, 0.85: 1.440, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def _get_z(conf: float) -> float:
    """Get z-score for confidence level."""
    closest = min(Z_SCORES.keys(), key=lambda k: abs(k - conf))
    return Z_SCORES[closest]


def _forecast_with_model(
    series: List[float], periods: int, model: str, alpha: float = 0.3
) -> List[float]:
    """Dispatch to the right forecasting model."""
    if model == "moving_average":
        return _moving_average_forecast(series, periods)
    elif model == "linear_regression":
        return _linreg_forecast(series, periods)
    else:  # exponential_smoothing (default)
        return _ema_forecast(series, periods, alpha)


def _fitted_values(series: List[float], model: str, alpha: float = 0.3) -> List[float]:
    """Get in-sample fitted values for a model."""
    if model == "moving_average":
        window = 3
        fitted = []
        for i in range(len(series)):
            w = min(window, i + 1)
            fitted.append(sum(series[max(0, i - w + 1):i + 1]) / w)
        return fitted
    elif model == "linear_regression":
        intercept, slope = _linear_regression(series)
        return [intercept + slope * i for i in range(len(series))]
    else:
        return _exponential_smoothing(series, alpha)


def _mae(actual: List[float], predicted: List[float]) -> float:
    """Mean Absolute Error."""
    n = min(len(actual), len(predicted))
    if n == 0:
        return 0.0
    return sum(abs(actual[i] - predicted[i]) for i in range(n)) / n


def _rmse(actual: List[float], predicted: List[float]) -> float:
    """Root Mean Squared Error."""
    n = min(len(actual), len(predicted))
    if n == 0:
        return 0.0
    mse = sum((actual[i] - predicted[i]) ** 2 for i in range(n)) / n
    return math.sqrt(mse)


def _mape(actual: List[float], predicted: List[float]) -> float:
    """Mean Absolute Percentage Error. Skips zeros in actual."""
    n = min(len(actual), len(predicted))
    if n == 0:
        return 0.0
    errors = []
    for i in range(n):
        if actual[i] != 0:
            errors.append(abs((actual[i] - predicted[i]) / actual[i]))
    return (sum(errors) / len(errors) * 100) if errors else 0.0


def _detect_trend(series: List[float]) -> Dict:
    """Detect trend direction and strength from a series."""
    if len(series) < 2:
        return {"direction": "insufficient_data", "strength": 0, "slope": 0}

    _, slope = _linear_regression(series)

    # Normalize slope by mean to get relative strength
    mean_val = sum(series) / len(series) if series else 1
    if mean_val == 0:
        mean_val = 1
    rel_slope = slope / abs(mean_val)

    if rel_slope > 0.05:
        direction = "strong_growth"
    elif rel_slope > 0.01:
        direction = "growth"
    elif rel_slope > -0.01:
        direction = "stable"
    elif rel_slope > -0.05:
        direction = "decline"
    else:
        direction = "strong_decline"

    # Check for trend reversal in recent vs older data
    reversal = None
    if len(series) >= 6:
        mid = len(series) // 2
        _, old_slope = _linear_regression(series[:mid])
        _, new_slope = _linear_regression(series[mid:])
        if old_slope > 0.01 and new_slope < -0.01:
            reversal = "growth_to_decline"
        elif old_slope < -0.01 and new_slope > 0.01:
            reversal = "decline_to_growth"

    return {
        "direction": direction,
        "strength": round(abs(rel_slope), 4),
        "slope": round(slope, 6),
        "slope_per_period": round(slope, 4),
        "reversal": reversal,
        "data_points": len(series),
    }


class RevenueForecastSkill(Skill):
    """
    Time-series revenue forecasting with multiple models, confidence intervals,
    trend detection, break-even estimation, and scenario analysis.
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_forecast",
            name="Revenue Forecast",
            version="1.0.0",
            category="revenue",
            description="Time-series revenue forecasting with multiple models, trend detection, and break-even estimation",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="forecast",
                    description="Generate N-period revenue forecast with confidence intervals",
                    parameters={
                        "periods": {"type": "int", "required": False, "description": "Number of periods to forecast (default: 7)"},
                        "model": {"type": "str", "required": False, "description": "Model: moving_average, exponential_smoothing, linear_regression"},
                    },
                ),
                SkillAction(
                    name="trend",
                    description="Detect current revenue trend direction and strength",
                    parameters={
                        "window": {"type": "int", "required": False, "description": "Number of recent points to analyze (default: all)"},
                    },
                ),
                SkillAction(
                    name="breakeven",
                    description="Estimate time to break-even given cost rate",
                    parameters={
                        "cost_per_period": {"type": "float", "required": False, "description": "Cost per period in USD (default: from config)"},
                    },
                ),
                SkillAction(
                    name="scenarios",
                    description="Optimistic/baseline/pessimistic revenue projections",
                    parameters={
                        "periods": {"type": "int", "required": False, "description": "Number of periods (default: 12)"},
                    },
                ),
                SkillAction(
                    name="backtest",
                    description="Compare model accuracy on historical data",
                    parameters={
                        "test_ratio": {"type": "float", "required": False, "description": "Fraction of data for testing (default: 0.3)"},
                    },
                ),
                SkillAction(
                    name="record",
                    description="Manually record a revenue data point",
                    parameters={
                        "revenue": {"type": "float", "required": True, "description": "Revenue amount in USD"},
                        "source": {"type": "str", "required": False, "description": "Revenue source label"},
                    },
                ),
                SkillAction(
                    name="history",
                    description="View recorded revenue time series",
                    parameters={
                        "limit": {"type": "int", "required": False, "description": "Max entries to return (default: 50)"},
                    },
                ),
                SkillAction(
                    name="stats",
                    description="Forecasting accuracy and usage statistics",
                    parameters={},
                ),
            ],
        )

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        try:
            if action == "forecast":
                return self._forecast(params)
            elif action == "trend":
                return self._trend(params)
            elif action == "breakeven":
                return self._breakeven(params)
            elif action == "scenarios":
                return self._scenarios(params)
            elif action == "backtest":
                return self._backtest(params)
            elif action == "record":
                return self._record(params)
            elif action == "history":
                return self._history(params)
            elif action == "stats":
                return self._stats(params)
            else:
                return SkillResult(
                    success=False,
                    message=f"Unknown action: {action}. Available: forecast, trend, breakeven, scenarios, backtest, record, history, stats",
                )
        except Exception as e:
            return SkillResult(success=False, message=f"RevenueForecast error: {e}")

    def _get_series(self, state: Dict) -> List[float]:
        """Get revenue time series, combining dashboard snapshots and manual records."""
        # Priority: use manual time_series if populated, else dashboard snapshots
        manual = state.get("time_series", [])
        if manual:
            return [entry.get("revenue", 0.0) for entry in manual]

        # Fall back to dashboard snapshots
        snapshots = _load_dashboard_snapshots()
        if snapshots:
            return _extract_revenue_series(snapshots)

        return []

    def _forecast(self, params: Dict) -> SkillResult:
        state = _load_state()
        series = self._get_series(state)
        if len(series) < 2:
            return SkillResult(
                success=False,
                message=f"Need at least 2 data points for forecasting, have {len(series)}. Use 'record' to add data or wait for dashboard snapshots.",
            )

        periods = int(params.get("periods", state["config"]["default_periods"]))
        model = params.get("model", state["config"]["default_model"])
        alpha = state["config"]["ema_alpha"]
        conf = state["config"]["confidence_level"]

        # Generate forecast
        forecast_values = _forecast_with_model(series, periods, model, alpha)

        # Calculate confidence intervals
        fitted = _fitted_values(series, model, alpha)
        std = _residual_std(series, fitted)
        z = _get_z(conf)
        forecast_with_ci = _confidence_interval(forecast_values, std, z)

        # Summary stats
        current = series[-1] if series else 0
        forecast_end = forecast_values[-1] if forecast_values else 0
        total_forecast = sum(forecast_values)

        result = {
            "model": model,
            "periods": periods,
            "confidence_level": conf,
            "current_value": round(current, 4),
            "forecast_end_value": round(forecast_end, 4),
            "total_forecast_revenue": round(total_forecast, 4),
            "change_pct": round(((forecast_end - current) / current * 100) if current else 0, 2),
            "forecast": forecast_with_ci,
            "data_points_used": len(series),
            "residual_std": round(std, 4),
        }

        # Save forecast for accuracy tracking
        state["stats"]["total_forecasts"] += 1
        state["forecasts"].append({
            "timestamp": _now_iso(),
            "model": model,
            "periods": periods,
            "forecast_values": [round(v, 4) for v in forecast_values],
            "data_points": len(series),
        })
        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Forecast ({model}): current ${current:.4f} → ${forecast_end:.4f} over {periods} periods ({result['change_pct']:+.1f}%)",
            data=result,
        )

    def _trend(self, params: Dict) -> SkillResult:
        state = _load_state()
        series = self._get_series(state)
        if len(series) < 2:
            return SkillResult(
                success=False,
                message=f"Need at least 2 data points for trend analysis, have {len(series)}.",
            )

        window = params.get("window")
        if window:
            window = int(window)
            series = series[-window:]

        trend_info = _detect_trend(series)

        # Add recent momentum (last 3 points vs previous 3)
        if len(series) >= 6:
            recent = sum(series[-3:]) / 3
            earlier = sum(series[-6:-3]) / 3
            if earlier > 0:
                trend_info["recent_momentum_pct"] = round((recent - earlier) / earlier * 100, 2)
            else:
                trend_info["recent_momentum_pct"] = 0

        state["stats"]["total_trend_checks"] += 1
        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Trend: {trend_info['direction']} (strength {trend_info['strength']}, slope {trend_info['slope_per_period']:+.4f}/period)"
                    + (f" ⚠ Reversal: {trend_info['reversal']}" if trend_info.get("reversal") else ""),
            data=trend_info,
        )

    def _breakeven(self, params: Dict) -> SkillResult:
        state = _load_state()
        series = self._get_series(state)
        if len(series) < 2:
            return SkillResult(
                success=False,
                message="Need at least 2 data points for break-even estimation.",
            )

        cost_per_period = float(params.get("cost_per_period", state["config"]["cost_per_period"]))
        alpha = state["config"]["ema_alpha"]

        # Use exponential smoothing for break-even projection
        smoothed = _exponential_smoothing(series, alpha)
        current_level = smoothed[-1] if smoothed else 0
        _, slope = _linear_regression(series)

        result = {
            "current_revenue_per_period": round(current_level, 4),
            "cost_per_period": round(cost_per_period, 4),
            "current_gap": round(current_level - cost_per_period, 4),
            "is_profitable": current_level >= cost_per_period,
        }

        if current_level >= cost_per_period:
            result["status"] = "already_profitable"
            result["margin"] = round((current_level - cost_per_period) / cost_per_period * 100, 2)
            result["message"] = f"Already profitable! Revenue ${current_level:.4f} > costs ${cost_per_period:.4f} ({result['margin']:.1f}% margin)"
        elif slope <= 0:
            result["status"] = "not_converging"
            result["message"] = f"Revenue trend is flat/declining (slope {slope:+.6f}). Break-even unreachable at current trajectory."
        else:
            # Periods until revenue = cost
            gap = cost_per_period - current_level
            periods_to_breakeven = math.ceil(gap / slope) if slope > 0 else float("inf")
            result["status"] = "converging"
            result["periods_to_breakeven"] = periods_to_breakeven
            result["slope_per_period"] = round(slope, 6)
            # Project date
            result["message"] = f"Estimated {periods_to_breakeven} periods to break-even (slope {slope:+.6f}/period, gap ${gap:.4f})"

        state["stats"]["total_breakeven_checks"] += 1
        _save_state(state)

        return SkillResult(
            success=True,
            message=result.get("message", "Break-even analysis complete"),
            data=result,
        )

    def _scenarios(self, params: Dict) -> SkillResult:
        state = _load_state()
        series = self._get_series(state)
        if len(series) < 3:
            return SkillResult(
                success=False,
                message=f"Need at least 3 data points for scenario analysis, have {len(series)}.",
            )

        periods = int(params.get("periods", 12))
        alpha = state["config"]["ema_alpha"]

        # Baseline: exponential smoothing
        baseline = _ema_forecast(series, periods, alpha)
        # Linear regression for trend
        linreg = _linreg_forecast(series, periods)

        # Calculate std for scenario spread
        fitted = _fitted_values(series, "exponential_smoothing", alpha)
        std = _residual_std(series, fitted)
        if std == 0:
            std = abs(baseline[0]) * 0.1 if baseline and baseline[0] != 0 else 0.01

        # Optimistic: baseline + 1.5 std growth trend
        optimistic = [max(0, baseline[i] + 1.5 * std * math.sqrt(i + 1)) for i in range(periods)]
        # Pessimistic: baseline - 1.5 std
        pessimistic = [max(0, baseline[i] - 1.5 * std * math.sqrt(i + 1)) for i in range(periods)]

        scenarios = {
            "periods": periods,
            "optimistic": {
                "values": [round(v, 4) for v in optimistic],
                "total": round(sum(optimistic), 4),
                "end_value": round(optimistic[-1], 4),
            },
            "baseline": {
                "values": [round(v, 4) for v in baseline],
                "total": round(sum(baseline), 4),
                "end_value": round(baseline[-1], 4),
            },
            "pessimistic": {
                "values": [round(v, 4) for v in pessimistic],
                "total": round(sum(pessimistic), 4),
                "end_value": round(pessimistic[-1], 4),
            },
            "linear_trend": {
                "values": [round(v, 4) for v in linreg],
                "total": round(sum(linreg), 4),
                "end_value": round(linreg[-1], 4),
            },
            "spread_std": round(std, 4),
        }

        return SkillResult(
            success=True,
            message=f"Scenarios over {periods} periods: optimistic ${scenarios['optimistic']['total']:.2f}, baseline ${scenarios['baseline']['total']:.2f}, pessimistic ${scenarios['pessimistic']['total']:.2f}",
            data=scenarios,
        )

    def _backtest(self, params: Dict) -> SkillResult:
        state = _load_state()
        series = self._get_series(state)
        if len(series) < 5:
            return SkillResult(
                success=False,
                message=f"Need at least 5 data points for backtesting, have {len(series)}.",
            )

        test_ratio = float(params.get("test_ratio", 0.3))
        split_idx = max(2, int(len(series) * (1 - test_ratio)))
        train = series[:split_idx]
        test = series[split_idx:]
        if not test:
            return SkillResult(success=False, message="Not enough data for test split.")

        alpha = state["config"]["ema_alpha"]
        models = ["moving_average", "exponential_smoothing", "linear_regression"]
        results = {}

        for model in models:
            forecast = _forecast_with_model(train, len(test), model, alpha)
            results[model] = {
                "mae": round(_mae(test, forecast), 6),
                "rmse": round(_rmse(test, forecast), 6),
                "mape": round(_mape(test, forecast), 2),
                "forecast": [round(v, 4) for v in forecast],
                "actual": [round(v, 4) for v in test],
            }

        # Find best model by MAE
        best = min(results.keys(), key=lambda m: results[m]["mae"])

        state["backtest_results"] = results
        state["stats"]["total_backtests"] += 1
        state["stats"]["best_model"] = best
        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Backtest ({split_idx} train / {len(test)} test): Best model = {best} (MAE={results[best]['mae']:.6f}, RMSE={results[best]['rmse']:.6f})",
            data={"train_size": split_idx, "test_size": len(test), "results": results, "best_model": best},
        )

    def _record(self, params: Dict) -> SkillResult:
        if "revenue" not in params:
            return SkillResult(success=False, message="Missing required parameter: revenue")

        state = _load_state()
        entry = {
            "timestamp": _now_iso(),
            "revenue": float(params["revenue"]),
            "source": params.get("source", "manual"),
        }
        state["time_series"].append(entry)
        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Recorded revenue ${entry['revenue']:.4f} from {entry['source']} ({len(state['time_series'])} total points)",
            data={"entry": entry, "total_points": len(state["time_series"])},
        )

    def _history(self, params: Dict) -> SkillResult:
        state = _load_state()
        series = state.get("time_series", [])

        # Also check dashboard
        dashboard_count = len(_load_dashboard_snapshots())

        limit = int(params.get("limit", 50))
        recent = series[-limit:] if series else []

        total_revenue = sum(e.get("revenue", 0) for e in series)

        return SkillResult(
            success=True,
            message=f"Revenue history: {len(series)} manual records, {dashboard_count} dashboard snapshots. Total manual revenue: ${total_revenue:.4f}",
            data={
                "manual_records": len(series),
                "dashboard_snapshots": dashboard_count,
                "total_manual_revenue": round(total_revenue, 4),
                "recent": recent,
            },
        )

    def _stats(self, params: Dict) -> SkillResult:
        state = _load_state()
        stats = state.get("stats", {})
        backtest = state.get("backtest_results", {})
        config = state.get("config", {})

        return SkillResult(
            success=True,
            message=f"Forecast stats: {stats.get('total_forecasts', 0)} forecasts, {stats.get('total_backtests', 0)} backtests, best model: {stats.get('best_model', 'N/A')}",
            data={
                "stats": stats,
                "backtest_accuracy": {
                    model: {"mae": res.get("mae"), "rmse": res.get("rmse"), "mape": res.get("mape")}
                    for model, res in backtest.items()
                },
                "config": config,
                "data_points": {
                    "manual": len(state.get("time_series", [])),
                    "dashboard": len(_load_dashboard_snapshots()),
                },
            },
        )
