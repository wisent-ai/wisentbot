#!/usr/bin/env python3
"""
OutcomeTracker Skill - Automatic action performance tracking and learning.

Closes the self-improvement feedback loop by:
- Recording every action outcome (success/failure/cost/duration)
- Computing per-skill and per-action performance metrics
- Identifying failure patterns and cost inefficiencies
- Generating performance reports with actionable recommendations
- Persisting metrics across sessions for long-term learning

This is the "measure outcome → adapt" component of the self-improvement cycle.
Without it, the agent has no way to know which actions work and which don't.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from .base import Skill, SkillManifest, SkillAction, SkillResult


OUTCOMES_FILE = Path(__file__).parent.parent / "data" / "outcomes.json"


class OutcomeTracker(Skill):
    """
    Tracks action outcomes to enable data-driven self-improvement.

    Records every action the agent takes along with its outcome,
    then provides analytics to identify what works, what doesn't,
    and what costs too much.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()
        self._session_start = datetime.now().isoformat()
        self._session_outcomes: List[Dict] = []

    def _ensure_data(self):
        OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not OUTCOMES_FILE.exists():
            self._save({
                "outcomes": [],
                "sessions": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_actions": 0,
                    "total_successes": 0,
                    "total_failures": 0,
                }
            })

    def _load(self) -> Dict:
        try:
            with open(OUTCOMES_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "outcomes": [],
                "sessions": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_actions": 0,
                    "total_successes": 0,
                    "total_failures": 0,
                }
            }

    def _save(self, data: Dict):
        with open(OUTCOMES_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="outcomes",
            name="Outcome Tracker",
            version="1.0.0",
            category="meta",
            description="Track action outcomes for data-driven self-improvement",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="record",
                    description="Record an action outcome (called automatically by agent loop)",
                    parameters={
                        "tool": {
                            "type": "string",
                            "required": True,
                            "description": "Tool/action that was executed (e.g. 'github:create_issue')"
                        },
                        "success": {
                            "type": "boolean",
                            "required": True,
                            "description": "Whether the action succeeded"
                        },
                        "cost": {
                            "type": "number",
                            "required": False,
                            "description": "Cost of the action in USD"
                        },
                        "duration_ms": {
                            "type": "number",
                            "required": False,
                            "description": "Execution time in milliseconds"
                        },
                        "error": {
                            "type": "string",
                            "required": False,
                            "description": "Error message if action failed"
                        },
                        "context": {
                            "type": "string",
                            "required": False,
                            "description": "Additional context about the action"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="report",
                    description="Get a performance report with per-skill success rates and recommendations",
                    parameters={
                        "skill_filter": {
                            "type": "string",
                            "required": False,
                            "description": "Filter report to a specific skill ID"
                        },
                        "last_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Only consider last N outcomes (default: all)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="failures",
                    description="List recent failures with error patterns",
                    parameters={
                        "last_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent failures to show (default: 10)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="trends",
                    description="Show performance trends over recent sessions",
                    parameters={
                        "last_n_sessions": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of sessions to analyze (default: 5)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recommendations",
                    description="Get actionable recommendations to improve performance",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="cost_analysis",
                    description="Analyze cost efficiency across skills and actions",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="session_summary",
                    description="Get summary of the current session's performance",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="end_session",
                    description="End current tracking session and persist summary",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "record": self._record,
            "report": self._report,
            "failures": self._failures,
            "trends": self._trends,
            "recommendations": self._recommendations,
            "cost_analysis": self._cost_analysis,
            "session_summary": self._session_summary,
            "end_session": self._end_session,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # === Record ===

    async def _record(self, params: Dict) -> SkillResult:
        """Record an action outcome."""
        tool = params.get("tool", "").strip()
        success = params.get("success", False)
        cost = params.get("cost", 0.0)
        duration_ms = params.get("duration_ms", 0.0)
        error = params.get("error", "")
        context = params.get("context", "")

        if not tool:
            return SkillResult(success=False, message="Tool name required")

        # Parse skill:action format
        parts = tool.split(":")
        skill_id = parts[0] if parts else tool
        action_name = parts[1] if len(parts) > 1 else "unknown"

        outcome = {
            "tool": tool,
            "skill_id": skill_id,
            "action": action_name,
            "success": bool(success),
            "cost": float(cost),
            "duration_ms": float(duration_ms),
            "error": str(error) if error else "",
            "context": str(context) if context else "",
            "timestamp": datetime.now().isoformat(),
            "session": self._session_start,
        }

        # Store in session memory
        self._session_outcomes.append(outcome)

        # Persist to disk
        data = self._load()
        data["outcomes"].append(outcome)
        data["metadata"]["total_actions"] = data["metadata"].get("total_actions", 0) + 1
        if success:
            data["metadata"]["total_successes"] = data["metadata"].get("total_successes", 0) + 1
        else:
            data["metadata"]["total_failures"] = data["metadata"].get("total_failures", 0) + 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recorded {'success' if success else 'FAILURE'} for {tool}",
            data={"outcome": outcome}
        )

    # === Record from agent loop (convenience method, not an action) ===

    def record_sync(self, tool: str, success: bool, cost: float = 0.0,
                    duration_ms: float = 0.0, error: str = "", context: str = ""):
        """Synchronous recording for use in agent loop without awaiting."""
        parts = tool.split(":")
        skill_id = parts[0] if parts else tool
        action_name = parts[1] if len(parts) > 1 else "unknown"

        outcome = {
            "tool": tool,
            "skill_id": skill_id,
            "action": action_name,
            "success": bool(success),
            "cost": float(cost),
            "duration_ms": float(duration_ms),
            "error": str(error) if error else "",
            "context": str(context) if context else "",
            "timestamp": datetime.now().isoformat(),
            "session": self._session_start,
        }

        self._session_outcomes.append(outcome)

        try:
            data = self._load()
            data["outcomes"].append(outcome)
            data["metadata"]["total_actions"] = data["metadata"].get("total_actions", 0) + 1
            if success:
                data["metadata"]["total_successes"] = data["metadata"].get("total_successes", 0) + 1
            else:
                data["metadata"]["total_failures"] = data["metadata"].get("total_failures", 0) + 1
            self._save(data)
        except Exception:
            pass  # Don't let tracking failures break the agent

    # === Report ===

    async def _report(self, params: Dict) -> SkillResult:
        """Generate a performance report."""
        skill_filter = params.get("skill_filter", "")
        last_n = params.get("last_n", 0)

        data = self._load()
        outcomes = data.get("outcomes", [])

        if skill_filter:
            outcomes = [o for o in outcomes if o.get("skill_id") == skill_filter]

        if last_n and last_n > 0:
            outcomes = outcomes[-last_n:]

        if not outcomes:
            return SkillResult(
                success=True,
                message="No outcomes recorded yet",
                data={"total": 0}
            )

        # Compute per-skill metrics
        skill_metrics = self._compute_skill_metrics(outcomes)

        # Compute overall metrics
        total = len(outcomes)
        successes = sum(1 for o in outcomes if o.get("success"))
        failures = total - successes
        total_cost = sum(o.get("cost", 0) for o in outcomes)
        avg_duration = sum(o.get("duration_ms", 0) for o in outcomes) / total if total else 0

        report = {
            "total_actions": total,
            "successes": successes,
            "failures": failures,
            "success_rate": round(successes / total * 100, 1) if total else 0,
            "total_cost_usd": round(total_cost, 4),
            "avg_duration_ms": round(avg_duration, 1),
            "per_skill": skill_metrics,
        }

        return SkillResult(
            success=True,
            message=f"Performance report: {successes}/{total} actions succeeded ({report['success_rate']}%)",
            data=report
        )

    # === Failures ===

    async def _failures(self, params: Dict) -> SkillResult:
        """List recent failures with error patterns."""
        last_n = params.get("last_n", 10)

        data = self._load()
        outcomes = data.get("outcomes", [])

        failures = [o for o in outcomes if not o.get("success")]
        recent_failures = failures[-last_n:] if last_n > 0 else failures

        # Identify error patterns
        error_counts = defaultdict(int)
        for f in failures:
            error = f.get("error", "unknown")
            # Truncate long errors to find patterns
            error_key = error[:100] if error else "no_error_message"
            error_counts[error_key] += 1

        # Most common errors
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Most failing skills
        skill_fail_counts = defaultdict(int)
        for f in failures:
            skill_fail_counts[f.get("skill_id", "unknown")] += 1
        worst_skills = sorted(skill_fail_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return SkillResult(
            success=True,
            message=f"{len(failures)} total failures, showing last {len(recent_failures)}",
            data={
                "recent_failures": recent_failures,
                "total_failures": len(failures),
                "common_errors": [{"error": e, "count": c} for e, c in common_errors],
                "worst_skills": [{"skill": s, "failures": c} for s, c in worst_skills],
            }
        )

    # === Trends ===

    async def _trends(self, params: Dict) -> SkillResult:
        """Show performance trends across sessions."""
        last_n_sessions = params.get("last_n_sessions", 5)

        data = self._load()
        outcomes = data.get("outcomes", [])

        if not outcomes:
            return SkillResult(
                success=True,
                message="No outcomes to analyze trends",
                data={"sessions": []}
            )

        # Group by session
        sessions = defaultdict(list)
        for o in outcomes:
            session_key = o.get("session", "unknown")
            sessions[session_key].append(o)

        # Sort sessions by first timestamp
        sorted_sessions = sorted(
            sessions.items(),
            key=lambda x: x[1][0].get("timestamp", "") if x[1] else ""
        )

        # Take last N sessions
        recent_sessions = sorted_sessions[-last_n_sessions:]

        trends = []
        for session_key, session_outcomes in recent_sessions:
            total = len(session_outcomes)
            successes = sum(1 for o in session_outcomes if o.get("success"))
            total_cost = sum(o.get("cost", 0) for o in session_outcomes)

            trends.append({
                "session": session_key,
                "actions": total,
                "success_rate": round(successes / total * 100, 1) if total else 0,
                "total_cost": round(total_cost, 4),
                "skills_used": list(set(o.get("skill_id", "") for o in session_outcomes)),
            })

        # Determine if improving
        improving = None
        if len(trends) >= 2:
            recent_rate = trends[-1]["success_rate"]
            earlier_rate = trends[0]["success_rate"]
            if recent_rate > earlier_rate:
                improving = "improving"
            elif recent_rate < earlier_rate:
                improving = "declining"
            else:
                improving = "stable"

        return SkillResult(
            success=True,
            message=f"Trends across {len(trends)} sessions: {improving or 'insufficient data'}",
            data={
                "sessions": trends,
                "trend": improving,
                "total_sessions_tracked": len(sessions),
            }
        )

    # === Recommendations ===

    async def _recommendations(self, params: Dict) -> SkillResult:
        """Generate actionable recommendations based on outcome data."""
        data = self._load()
        outcomes = data.get("outcomes", [])

        if len(outcomes) < 5:
            return SkillResult(
                success=True,
                message="Need at least 5 recorded outcomes for recommendations",
                data={"recommendations": [], "reason": "insufficient_data"}
            )

        recommendations = []
        skill_metrics = self._compute_skill_metrics(outcomes)

        for skill_id, metrics in skill_metrics.items():
            success_rate = metrics["success_rate"]
            total = metrics["total"]

            # Flag skills with low success rate (need at least 3 attempts)
            if total >= 3 and success_rate < 50:
                recommendations.append({
                    "type": "low_success_rate",
                    "priority": "high",
                    "skill": skill_id,
                    "message": f"Skill '{skill_id}' has only {success_rate}% success rate over {total} attempts. Consider: checking credentials, reviewing error patterns, or avoiding this skill.",
                    "data": metrics,
                })

            # Flag expensive skills
            avg_cost = metrics.get("avg_cost", 0)
            if avg_cost > 0.01 and total >= 3:
                recommendations.append({
                    "type": "high_cost",
                    "priority": "medium",
                    "skill": skill_id,
                    "message": f"Skill '{skill_id}' costs avg ${avg_cost:.4f}/action. Consider batching operations or using cheaper alternatives.",
                    "data": metrics,
                })

            # Flag slow skills
            avg_duration = metrics.get("avg_duration_ms", 0)
            if avg_duration > 10000 and total >= 3:  # >10s
                recommendations.append({
                    "type": "slow_execution",
                    "priority": "low",
                    "skill": skill_id,
                    "message": f"Skill '{skill_id}' takes avg {avg_duration:.0f}ms. Consider if faster alternatives exist.",
                    "data": metrics,
                })

        # Check for repeated failures of same action
        action_failures = defaultdict(int)
        for o in outcomes[-50:]:  # Last 50 outcomes
            if not o.get("success"):
                action_failures[o.get("tool", "")] += 1

        for action, count in action_failures.items():
            if count >= 3:
                recommendations.append({
                    "type": "repeated_failure",
                    "priority": "high",
                    "skill": action,
                    "message": f"Action '{action}' has failed {count} times recently. Stop retrying and investigate the root cause.",
                })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))

        return SkillResult(
            success=True,
            message=f"Generated {len(recommendations)} recommendation(s)",
            data={"recommendations": recommendations}
        )

    # === Cost Analysis ===

    async def _cost_analysis(self, params: Dict) -> SkillResult:
        """Analyze cost efficiency across skills."""
        data = self._load()
        outcomes = data.get("outcomes", [])

        if not outcomes:
            return SkillResult(
                success=True,
                message="No outcomes to analyze costs",
                data={"total_cost": 0}
            )

        total_cost = sum(o.get("cost", 0) for o in outcomes)
        successful_cost = sum(o.get("cost", 0) for o in outcomes if o.get("success"))
        wasted_cost = sum(o.get("cost", 0) for o in outcomes if not o.get("success"))

        # Per-skill cost breakdown
        skill_costs = defaultdict(lambda: {"total_cost": 0, "actions": 0, "successes": 0})
        for o in outcomes:
            sid = o.get("skill_id", "unknown")
            skill_costs[sid]["total_cost"] += o.get("cost", 0)
            skill_costs[sid]["actions"] += 1
            if o.get("success"):
                skill_costs[sid]["successes"] += 1

        cost_breakdown = []
        for sid, info in skill_costs.items():
            cost_breakdown.append({
                "skill": sid,
                "total_cost": round(info["total_cost"], 4),
                "actions": info["actions"],
                "cost_per_action": round(info["total_cost"] / info["actions"], 4) if info["actions"] else 0,
                "cost_per_success": round(
                    info["total_cost"] / info["successes"], 4
                ) if info["successes"] else None,
                "waste_ratio": round(
                    1 - (info["successes"] / info["actions"]), 2
                ) if info["actions"] else 0,
            })

        cost_breakdown.sort(key=lambda x: x["total_cost"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Total cost: ${total_cost:.4f} (${wasted_cost:.4f} wasted on failures)",
            data={
                "total_cost": round(total_cost, 4),
                "successful_cost": round(successful_cost, 4),
                "wasted_cost": round(wasted_cost, 4),
                "waste_percentage": round(wasted_cost / total_cost * 100, 1) if total_cost else 0,
                "per_skill": cost_breakdown,
            }
        )

    # === Session Summary ===

    async def _session_summary(self, params: Dict) -> SkillResult:
        """Get summary of current session."""
        outcomes = self._session_outcomes

        if not outcomes:
            return SkillResult(
                success=True,
                message="No actions recorded in current session",
                data={"session_start": self._session_start, "total": 0}
            )

        total = len(outcomes)
        successes = sum(1 for o in outcomes if o.get("success"))
        total_cost = sum(o.get("cost", 0) for o in outcomes)
        skills_used = list(set(o.get("skill_id", "") for o in outcomes))

        return SkillResult(
            success=True,
            message=f"Session: {successes}/{total} actions succeeded, ${total_cost:.4f} spent",
            data={
                "session_start": self._session_start,
                "total_actions": total,
                "successes": successes,
                "failures": total - successes,
                "success_rate": round(successes / total * 100, 1) if total else 0,
                "total_cost": round(total_cost, 4),
                "skills_used": skills_used,
                "actions": [
                    {
                        "tool": o.get("tool"),
                        "success": o.get("success"),
                        "cost": o.get("cost", 0),
                    }
                    for o in outcomes
                ],
            }
        )

    # === End Session ===

    async def _end_session(self, params: Dict) -> SkillResult:
        """End current session and persist summary."""
        outcomes = self._session_outcomes

        total = len(outcomes)
        successes = sum(1 for o in outcomes if o.get("success"))
        total_cost = sum(o.get("cost", 0) for o in outcomes)

        session_summary = {
            "session_start": self._session_start,
            "session_end": datetime.now().isoformat(),
            "total_actions": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / total * 100, 1) if total else 0,
            "total_cost": round(total_cost, 4),
            "skills_used": list(set(o.get("skill_id", "") for o in outcomes)),
        }

        # Persist session summary
        data = self._load()
        if "sessions" not in data:
            data["sessions"] = []
        data["sessions"].append(session_summary)
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Session ended: {successes}/{total} actions, ${total_cost:.4f} cost",
            data=session_summary
        )

    # === Helpers ===

    def _compute_skill_metrics(self, outcomes: List[Dict]) -> Dict:
        """Compute per-skill performance metrics."""
        skill_data = defaultdict(lambda: {
            "total": 0, "successes": 0, "failures": 0,
            "total_cost": 0, "total_duration_ms": 0,
            "actions": defaultdict(lambda: {"total": 0, "successes": 0}),
        })

        for o in outcomes:
            sid = o.get("skill_id", "unknown")
            s = skill_data[sid]
            s["total"] += 1
            if o.get("success"):
                s["successes"] += 1
            else:
                s["failures"] += 1
            s["total_cost"] += o.get("cost", 0)
            s["total_duration_ms"] += o.get("duration_ms", 0)

            action_name = o.get("action", "unknown")
            s["actions"][action_name]["total"] += 1
            if o.get("success"):
                s["actions"][action_name]["successes"] += 1

        result = {}
        for sid, s in skill_data.items():
            total = s["total"]
            action_breakdown = {}
            for aname, adata in s["actions"].items():
                action_breakdown[aname] = {
                    "total": adata["total"],
                    "successes": adata["successes"],
                    "success_rate": round(adata["successes"] / adata["total"] * 100, 1) if adata["total"] else 0,
                }

            result[sid] = {
                "total": total,
                "successes": s["successes"],
                "failures": s["failures"],
                "success_rate": round(s["successes"] / total * 100, 1) if total else 0,
                "total_cost": round(s["total_cost"], 4),
                "avg_cost": round(s["total_cost"] / total, 4) if total else 0,
                "avg_duration_ms": round(s["total_duration_ms"] / total, 1) if total else 0,
                "actions": action_breakdown,
            }

        return result
