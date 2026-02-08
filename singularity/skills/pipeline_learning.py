#!/usr/bin/env python3
"""
PipelineLearningSkill - Auto-tune pipeline optimization using execution outcome data.

Currently, PipelinePlannerSkill has three static optimization strategies (cost, speed,
reliability) with hardcoded parameter adjustments. It records outcomes but never learns
from them. This skill closes the feedback loop:

1. Analyze pipeline execution outcomes to identify patterns
2. Learn per-tool performance profiles (avg duration, failure rate, cost)
3. Auto-recommend optimization strategies based on pipeline characteristics
4. Generate tuned step parameters from historical data (timeouts, retries, cost)
5. Detect bottleneck steps that consistently fail or exceed budgets
6. Track strategy effectiveness over time

This is the #1 priority from session 174 memory. It makes pipeline optimization
data-driven instead of static.

Pillar: Self-Improvement (act → measure → adapt feedback loop for pipelines)
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import Skill, SkillAction, SkillManifest, SkillResult


LEARNING_DATA_FILE = Path(__file__).parent.parent / "data" / "pipeline_learning.json"
PIPELINE_PLANS_FILE = Path(__file__).parent.parent / "data" / "pipeline_plans.json"


class PipelineLearningSkill(Skill):
    """Learns from pipeline execution outcomes to auto-tune optimization strategies."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="pipeline_learning",
            name="Pipeline Learning",
            version="1.0.0",
            category="planning",
            description="Auto-tune pipeline optimization from execution outcome data",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="ingest",
                    description="Ingest pipeline execution results into learning data",
                    parameters={
                        "pipeline_id": {"type": "string", "required": True, "description": "Pipeline or plan ID"},
                        "steps": {"type": "array", "required": True, "description": "List of step results with tool, success, duration_ms, cost, retries"},
                        "overall_success": {"type": "boolean", "required": True},
                        "strategy_used": {"type": "string", "required": False, "description": "Optimization strategy that was used"},
                        "pipeline_type": {"type": "string", "required": False, "description": "Category of pipeline (e.g. deploy, review, content)"},
                    },
                ),
                SkillAction(
                    name="tool_profile",
                    description="Get learned performance profile for a specific tool",
                    parameters={
                        "tool": {"type": "string", "required": True, "description": "Tool identifier (e.g. shell:run)"},
                    },
                ),
                SkillAction(
                    name="recommend",
                    description="Recommend optimization strategy and tuned parameters for a pipeline",
                    parameters={
                        "pipeline": {"type": "array", "required": True, "description": "Pipeline steps to optimize"},
                        "pipeline_type": {"type": "string", "required": False, "description": "Pipeline category"},
                        "goal": {"type": "string", "required": False, "description": "Optimization goal: auto, cost, speed, reliability"},
                    },
                ),
                SkillAction(
                    name="bottlenecks",
                    description="Identify tools and steps that consistently underperform",
                    parameters={
                        "min_executions": {"type": "integer", "required": False, "description": "Minimum executions to consider (default 3)"},
                    },
                ),
                SkillAction(
                    name="strategy_effectiveness",
                    description="Compare effectiveness of different optimization strategies",
                    parameters={},
                ),
                SkillAction(
                    name="tune_step",
                    description="Get data-driven parameter recommendations for a single step",
                    parameters={
                        "tool": {"type": "string", "required": True},
                        "current_timeout": {"type": "number", "required": False},
                        "current_retries": {"type": "integer", "required": False},
                        "current_max_cost": {"type": "number", "required": False},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Show learning data summary and coverage",
                    parameters={},
                ),
            ],
        )

    def _load(self) -> Dict:
        if LEARNING_DATA_FILE.exists():
            try:
                return json.loads(LEARNING_DATA_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "tool_stats": {},
            "strategy_stats": {},
            "pipeline_type_stats": {},
            "executions": [],
        }

    def _save(self, data: Dict) -> None:
        LEARNING_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        LEARNING_DATA_FILE.write_text(json.dumps(data, indent=2, default=str))

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "ingest": self._ingest,
            "tool_profile": self._tool_profile,
            "recommend": self._recommend,
            "bottlenecks": self._bottlenecks,
            "strategy_effectiveness": self._strategy_effectiveness,
            "tune_step": self._tune_step,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    def _ingest(self, params: Dict) -> SkillResult:
        """Ingest pipeline execution results into learning data."""
        pipeline_id = params.get("pipeline_id", "")
        steps = params.get("steps", [])
        overall_success = params.get("overall_success", False)
        strategy_used = params.get("strategy_used", "none")
        pipeline_type = params.get("pipeline_type", "unknown")

        if not pipeline_id:
            return SkillResult(success=False, message="pipeline_id is required")
        if not steps:
            return SkillResult(success=False, message="steps is required")

        data = self._load()

        # Update per-tool statistics
        for step in steps:
            tool = step.get("tool", "unknown")
            if tool not in data["tool_stats"]:
                data["tool_stats"][tool] = {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_duration_ms": 0.0,
                    "total_cost": 0.0,
                    "total_retries": 0,
                    "durations": [],
                    "costs": [],
                }
            ts = data["tool_stats"][tool]
            ts["executions"] += 1
            if step.get("success", False):
                ts["successes"] += 1
            else:
                ts["failures"] += 1
            duration = step.get("duration_ms", 0.0)
            cost = step.get("cost", 0.0)
            ts["total_duration_ms"] += duration
            ts["total_cost"] += cost
            ts["total_retries"] += step.get("retries", 0)
            # Keep last 50 data points for percentile calculations
            ts["durations"].append(duration)
            ts["durations"] = ts["durations"][-50:]
            ts["costs"].append(cost)
            ts["costs"] = ts["costs"][-50:]

        # Update strategy statistics
        if strategy_used not in data["strategy_stats"]:
            data["strategy_stats"][strategy_used] = {
                "executions": 0,
                "successes": 0,
                "total_cost": 0.0,
                "total_duration_ms": 0.0,
            }
        ss = data["strategy_stats"][strategy_used]
        ss["executions"] += 1
        if overall_success:
            ss["successes"] += 1
        ss["total_cost"] += sum(s.get("cost", 0) for s in steps)
        ss["total_duration_ms"] += sum(s.get("duration_ms", 0) for s in steps)

        # Update pipeline type statistics
        if pipeline_type not in data["pipeline_type_stats"]:
            data["pipeline_type_stats"][pipeline_type] = {
                "executions": 0,
                "successes": 0,
                "best_strategy": None,
                "strategy_results": {},
            }
        pts = data["pipeline_type_stats"][pipeline_type]
        pts["executions"] += 1
        if overall_success:
            pts["successes"] += 1
        if strategy_used not in pts["strategy_results"]:
            pts["strategy_results"][strategy_used] = {"runs": 0, "successes": 0}
        pts["strategy_results"][strategy_used]["runs"] += 1
        if overall_success:
            pts["strategy_results"][strategy_used]["successes"] += 1
        # Update best strategy for this pipeline type
        best = None
        best_rate = -1
        for strat, res in pts["strategy_results"].items():
            if res["runs"] >= 2:
                rate = res["successes"] / res["runs"]
                if rate > best_rate:
                    best_rate = rate
                    best = strat
        pts["best_strategy"] = best

        # Store execution record (keep last 200)
        data["executions"].append({
            "pipeline_id": pipeline_id,
            "pipeline_type": pipeline_type,
            "strategy": strategy_used,
            "success": overall_success,
            "step_count": len(steps),
            "steps_succeeded": sum(1 for s in steps if s.get("success")),
            "total_cost": sum(s.get("cost", 0) for s in steps),
            "total_duration_ms": sum(s.get("duration_ms", 0) for s in steps),
            "timestamp": datetime.now().isoformat(),
        })
        data["executions"] = data["executions"][-200:]

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Ingested {len(steps)} step results for pipeline {pipeline_id}",
            data={
                "pipeline_id": pipeline_id,
                "tools_updated": list(set(s.get("tool", "unknown") for s in steps)),
                "strategy_updated": strategy_used,
                "pipeline_type": pipeline_type,
            },
        )

    def _tool_profile(self, params: Dict) -> SkillResult:
        """Get learned performance profile for a specific tool."""
        tool = params.get("tool", "")
        if not tool:
            return SkillResult(success=False, message="tool is required")

        data = self._load()
        ts = data["tool_stats"].get(tool)
        if not ts:
            return SkillResult(
                success=False,
                message=f"No data for tool '{tool}'. Ingest pipeline results first.",
            )

        execs = ts["executions"]
        success_rate = ts["successes"] / execs if execs else 0
        avg_duration = ts["total_duration_ms"] / execs if execs else 0
        avg_cost = ts["total_cost"] / execs if execs else 0
        avg_retries = ts["total_retries"] / execs if execs else 0

        # Percentile calculations from recent data
        durations = sorted(ts.get("durations", []))
        costs = sorted(ts.get("costs", []))
        p50_duration = self._percentile(durations, 50)
        p95_duration = self._percentile(durations, 95)
        p50_cost = self._percentile(costs, 50)
        p95_cost = self._percentile(costs, 95)

        # Reliability classification
        if success_rate >= 0.95:
            reliability = "high"
        elif success_rate >= 0.75:
            reliability = "medium"
        else:
            reliability = "low"

        return SkillResult(
            success=True,
            message=f"Tool '{tool}': {execs} runs, {success_rate:.0%} success, avg {avg_duration:.0f}ms",
            data={
                "tool": tool,
                "executions": execs,
                "success_rate": round(success_rate, 3),
                "failure_rate": round(1 - success_rate, 3),
                "avg_duration_ms": round(avg_duration, 1),
                "p50_duration_ms": round(p50_duration, 1),
                "p95_duration_ms": round(p95_duration, 1),
                "avg_cost": round(avg_cost, 4),
                "p50_cost": round(p50_cost, 4),
                "p95_cost": round(p95_cost, 4),
                "avg_retries_per_run": round(avg_retries, 2),
                "reliability": reliability,
                "recommended_timeout_ms": round(p95_duration * 1.3, 0) if p95_duration else None,
                "recommended_max_cost": round(p95_cost * 1.2, 4) if p95_cost else None,
                "recommended_retries": 1 if success_rate < 0.9 else 0,
            },
        )

    def _recommend(self, params: Dict) -> SkillResult:
        """Recommend optimization strategy and tuned parameters for a pipeline."""
        pipeline = params.get("pipeline", [])
        if not pipeline:
            return SkillResult(success=False, message="pipeline is required")

        pipeline_type = params.get("pipeline_type", "unknown")
        goal = params.get("goal", "auto")
        data = self._load()

        # If goal is auto, pick the best strategy for this pipeline type
        if goal == "auto":
            goal = self._auto_select_strategy(data, pipeline_type, pipeline)

        # Tune each step using learned data
        tuned_pipeline = []
        adjustments = []
        for i, step in enumerate(pipeline):
            tool = step.get("tool", "unknown")
            tuned = dict(step)
            ts = data["tool_stats"].get(tool)

            if ts and ts["executions"] >= 3:
                # Data-driven tuning
                execs = ts["executions"]
                success_rate = ts["successes"] / execs
                durations = sorted(ts.get("durations", []))
                costs = sorted(ts.get("costs", []))
                p95_dur = self._percentile(durations, 95)
                p95_cost = self._percentile(costs, 95)

                if goal == "cost":
                    # Tight cost, short timeout, no retries
                    median_cost = self._percentile(costs, 50)
                    tuned["max_cost"] = round(median_cost * 1.1, 4) if median_cost else step.get("max_cost", 0.05)
                    median_dur = self._percentile(durations, 50)
                    tuned["timeout_seconds"] = round((median_dur * 1.2) / 1000, 1) if median_dur else step.get("timeout_seconds", 30)
                    tuned["retry_count"] = 0
                elif goal == "speed":
                    # Tight timeout based on actual data, no retries
                    p75_dur = self._percentile(durations, 75)
                    tuned["timeout_seconds"] = round((p75_dur * 1.1) / 1000, 1) if p75_dur else step.get("timeout_seconds", 15)
                    tuned["retry_count"] = 0
                    tuned["max_cost"] = round(p95_cost * 1.1, 4) if p95_cost else step.get("max_cost", 0.05)
                elif goal == "reliability":
                    # Generous timeout and retries based on failure rate
                    tuned["timeout_seconds"] = round((p95_dur * 1.5) / 1000, 1) if p95_dur else step.get("timeout_seconds", 45)
                    tuned["max_cost"] = round(p95_cost * 1.5, 4) if p95_cost else step.get("max_cost", 0.1)
                    tuned["retry_count"] = 2 if success_rate < 0.8 else (1 if success_rate < 0.95 else 0)

                changes = []
                for key in ["timeout_seconds", "max_cost", "retry_count"]:
                    if tuned.get(key) != step.get(key):
                        changes.append(f"{key}: {step.get(key)} → {tuned.get(key)}")
                if changes:
                    adjustments.append(f"Step {i} ({tool}): {', '.join(changes)}")
            else:
                # No data - use static defaults based on strategy
                if goal == "cost":
                    tuned["max_cost"] = min(step.get("max_cost", 0.05), 0.03)
                    tuned["retry_count"] = 0
                elif goal == "speed":
                    tuned["timeout_seconds"] = min(step.get("timeout_seconds", 30), 15)
                    tuned["retry_count"] = 0
                elif goal == "reliability":
                    tuned["retry_count"] = max(step.get("retry_count", 0), 1)
                    tuned["timeout_seconds"] = step.get("timeout_seconds", 30) * 1.5

            tuned_pipeline.append(tuned)

        # Estimate tuned pipeline cost/time
        est_cost = sum(s.get("max_cost", 0.05) for s in tuned_pipeline)
        est_time = sum(s.get("timeout_seconds", 30) for s in tuned_pipeline)

        return SkillResult(
            success=True,
            message=f"Recommended strategy '{goal}': {len(adjustments)} data-driven adjustments",
            data={
                "strategy": goal,
                "pipeline": tuned_pipeline,
                "adjustments": adjustments,
                "adjustments_count": len(adjustments),
                "estimated_max_cost": round(est_cost, 4),
                "estimated_max_duration_s": round(est_time, 1),
                "data_driven_steps": sum(1 for t in data["tool_stats"] if data["tool_stats"][t]["executions"] >= 3),
                "total_steps": len(pipeline),
            },
        )

    def _auto_select_strategy(self, data: Dict, pipeline_type: str, pipeline: List[Dict]) -> str:
        """Auto-select the best strategy based on learning data."""
        # Check if we have a known best strategy for this pipeline type
        pts = data.get("pipeline_type_stats", {}).get(pipeline_type)
        if pts and pts.get("best_strategy"):
            return pts["best_strategy"]

        # Check overall strategy effectiveness
        strategy_stats = data.get("strategy_stats", {})
        best_strat = None
        best_rate = -1
        for strat, ss in strategy_stats.items():
            if strat == "none":
                continue
            if ss["executions"] >= 3:
                rate = ss["successes"] / ss["executions"]
                if rate > best_rate:
                    best_rate = rate
                    best_strat = strat

        if best_strat and best_rate > 0.5:
            return best_strat

        # Heuristic fallback: check tools in pipeline for reliability concerns
        low_reliability_count = 0
        for step in pipeline:
            tool = step.get("tool", "unknown")
            ts = data.get("tool_stats", {}).get(tool)
            if ts and ts["executions"] >= 3:
                rate = ts["successes"] / ts["executions"]
                if rate < 0.8:
                    low_reliability_count += 1

        if low_reliability_count > 0:
            return "reliability"

        # Default to reliability for safety
        return "reliability"

    def _bottlenecks(self, params: Dict) -> SkillResult:
        """Identify tools that consistently underperform."""
        min_execs = params.get("min_executions", 3)
        data = self._load()

        bottlenecks = []
        for tool, ts in data.get("tool_stats", {}).items():
            if ts["executions"] < min_execs:
                continue
            execs = ts["executions"]
            success_rate = ts["successes"] / execs
            avg_duration = ts["total_duration_ms"] / execs
            avg_retries = ts["total_retries"] / execs

            issues = []
            if success_rate < 0.8:
                issues.append(f"high failure rate ({1-success_rate:.0%})")
            if avg_retries > 0.5:
                issues.append(f"frequent retries (avg {avg_retries:.1f})")
            if avg_duration > 60000:
                issues.append(f"slow execution (avg {avg_duration/1000:.1f}s)")

            if issues:
                bottlenecks.append({
                    "tool": tool,
                    "executions": execs,
                    "success_rate": round(success_rate, 3),
                    "avg_duration_ms": round(avg_duration, 1),
                    "avg_retries": round(avg_retries, 2),
                    "issues": issues,
                    "severity": "critical" if success_rate < 0.5 else ("high" if success_rate < 0.75 else "medium"),
                })

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2}
        bottlenecks.sort(key=lambda b: severity_order.get(b["severity"], 3))

        return SkillResult(
            success=True,
            message=f"Found {len(bottlenecks)} bottleneck tools" if bottlenecks else "No bottlenecks detected",
            data={
                "bottlenecks": bottlenecks,
                "total_tools_analyzed": sum(
                    1 for ts in data.get("tool_stats", {}).values()
                    if ts["executions"] >= min_execs
                ),
                "min_executions_threshold": min_execs,
            },
        )

    def _strategy_effectiveness(self, params: Dict) -> SkillResult:
        """Compare effectiveness of different optimization strategies."""
        data = self._load()
        strategies = []

        for strat, ss in data.get("strategy_stats", {}).items():
            execs = ss["executions"]
            if execs == 0:
                continue
            success_rate = ss["successes"] / execs
            avg_cost = ss["total_cost"] / execs
            avg_duration = ss["total_duration_ms"] / execs

            # Wilson score lower bound for ranking with small samples
            wilson = self._wilson_lower(ss["successes"], execs)

            strategies.append({
                "strategy": strat,
                "executions": execs,
                "success_rate": round(success_rate, 3),
                "wilson_score": round(wilson, 3),
                "avg_cost": round(avg_cost, 4),
                "avg_duration_ms": round(avg_duration, 1),
            })

        strategies.sort(key=lambda s: s["wilson_score"], reverse=True)

        # Per-pipeline-type breakdown
        type_breakdown = {}
        for ptype, pts in data.get("pipeline_type_stats", {}).items():
            if pts["executions"] > 0:
                type_breakdown[ptype] = {
                    "executions": pts["executions"],
                    "success_rate": round(pts["successes"] / pts["executions"], 3),
                    "best_strategy": pts.get("best_strategy"),
                    "strategies_tried": list(pts.get("strategy_results", {}).keys()),
                }

        return SkillResult(
            success=True,
            message=f"Analyzed {len(strategies)} strategies across {len(type_breakdown)} pipeline types",
            data={
                "strategies": strategies,
                "best_overall": strategies[0]["strategy"] if strategies else None,
                "pipeline_type_breakdown": type_breakdown,
            },
        )

    def _tune_step(self, params: Dict) -> SkillResult:
        """Get data-driven parameter recommendations for a single step."""
        tool = params.get("tool", "")
        if not tool:
            return SkillResult(success=False, message="tool is required")

        data = self._load()
        ts = data["tool_stats"].get(tool)
        if not ts or ts["executions"] < 2:
            return SkillResult(
                success=False,
                message=f"Insufficient data for '{tool}' (need >=2 executions, have {ts['executions'] if ts else 0})",
            )

        execs = ts["executions"]
        success_rate = ts["successes"] / execs
        durations = sorted(ts.get("durations", []))
        costs = sorted(ts.get("costs", []))

        current_timeout = params.get("current_timeout")
        current_retries = params.get("current_retries")
        current_max_cost = params.get("current_max_cost")

        # Calculate recommended values
        p50_dur = self._percentile(durations, 50)
        p95_dur = self._percentile(durations, 95)
        p50_cost = self._percentile(costs, 50)
        p95_cost = self._percentile(costs, 95)

        rec_timeout = round((p95_dur * 1.3) / 1000, 1) if p95_dur else None
        rec_max_cost = round(p95_cost * 1.2, 4) if p95_cost else None
        rec_retries = 2 if success_rate < 0.7 else (1 if success_rate < 0.9 else 0)

        suggestions = []
        if current_timeout and rec_timeout:
            if current_timeout > rec_timeout * 2:
                suggestions.append(f"Timeout too generous ({current_timeout}s vs recommended {rec_timeout}s)")
            elif current_timeout < rec_timeout * 0.5:
                suggestions.append(f"Timeout too tight ({current_timeout}s vs recommended {rec_timeout}s)")
        if current_max_cost and rec_max_cost:
            if current_max_cost < rec_max_cost:
                suggestions.append(f"Cost budget may be too low (${current_max_cost} vs recommended ${rec_max_cost})")
        if current_retries is not None and current_retries != rec_retries:
            suggestions.append(f"Retry count: {current_retries} → {rec_retries} (based on {success_rate:.0%} success rate)")

        return SkillResult(
            success=True,
            message=f"Tuning for '{tool}': {len(suggestions)} suggestions",
            data={
                "tool": tool,
                "data_points": execs,
                "success_rate": round(success_rate, 3),
                "recommended": {
                    "timeout_seconds": rec_timeout,
                    "max_cost": rec_max_cost,
                    "retry_count": rec_retries,
                },
                "current": {
                    "timeout_seconds": current_timeout,
                    "max_cost": current_max_cost,
                    "retry_count": current_retries,
                },
                "suggestions": suggestions,
                "confidence": "high" if execs >= 10 else ("medium" if execs >= 5 else "low"),
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """Show learning data summary and coverage."""
        data = self._load()

        tool_stats = data.get("tool_stats", {})
        total_tools = len(tool_stats)
        total_execs = sum(ts["executions"] for ts in tool_stats.values())
        tools_with_enough_data = sum(1 for ts in tool_stats.values() if ts["executions"] >= 3)

        strategy_stats = data.get("strategy_stats", {})
        total_strategies = len(strategy_stats)

        pipeline_types = data.get("pipeline_type_stats", {})
        types_with_best = sum(1 for pts in pipeline_types.values() if pts.get("best_strategy"))

        executions = data.get("executions", [])
        recent = executions[-10:] if executions else []
        recent_success = sum(1 for e in recent if e.get("success")) / len(recent) if recent else 0

        return SkillResult(
            success=True,
            message=f"Pipeline Learning: {total_tools} tools, {total_execs} total executions, {tools_with_enough_data} tools with enough data",
            data={
                "total_tools_tracked": total_tools,
                "total_executions": total_execs,
                "tools_with_sufficient_data": tools_with_enough_data,
                "strategies_tracked": total_strategies,
                "pipeline_types_tracked": len(pipeline_types),
                "pipeline_types_with_best_strategy": types_with_best,
                "recent_10_success_rate": round(recent_success, 3),
                "total_execution_records": len(executions),
            },
        )

    @staticmethod
    def _percentile(sorted_data: List[float], pct: float) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        d = k - f
        return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])

    @staticmethod
    def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
        """Wilson score confidence interval lower bound."""
        if total == 0:
            return 0.0
        p = successes / total
        denominator = 1 + z * z / total
        center = p + z * z / (2 * total)
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        return (center - spread) / denominator
