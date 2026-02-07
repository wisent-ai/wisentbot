"""
PerformanceTrackerSkill - Track and analyze agent performance metrics.

Records action outcomes, computes success rates, identifies patterns,
and provides data-driven recommendations for self-improvement.

This skill serves the Self-Improvement pillar by giving the agent
quantitative data about its own behavior and effectiveness.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillManifest, SkillAction, SkillResult


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class ActionRecord:
    """A single recorded action outcome."""
    skill_id: str
    action_name: str
    success: bool
    timestamp: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    error_message: str = ""
    cycle: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PerformanceTrackerSkill(Skill):
    """
    Tracks agent performance metrics and provides analytical reports.

    Capabilities:
    - Record action outcomes (success/failure, cost, duration)
    - Compute success rates per skill and action
    - Identify failure patterns and costly operations
    - Provide recommendations for behavior optimization
    - Persist metrics to disk for cross-session analysis
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._records: List[ActionRecord] = []
        self._session_start = datetime.now().isoformat()
        self._persist_path: Optional[Path] = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="performance_tracker",
            name="Performance Tracker",
            version="1.0.0",
            category="self-improvement",
            description="Track action outcomes, analyze performance, and recommend improvements",
            actions=[
                SkillAction(
                    name="record",
                    description="Record an action outcome for tracking",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill that was executed"},
                        "action_name": {"type": "string", "required": True, "description": "Action that was executed"},
                        "success": {"type": "boolean", "required": True, "description": "Whether the action succeeded"},
                        "duration_seconds": {"type": "number", "required": False, "description": "How long the action took"},
                        "cost_usd": {"type": "number", "required": False, "description": "Cost of the action in USD"},
                        "error_message": {"type": "string", "required": False, "description": "Error message if failed"},
                        "cycle": {"type": "integer", "required": False, "description": "Agent cycle number"},
                    },
                ),
                SkillAction(
                    name="report",
                    description="Get a comprehensive performance summary report",
                    parameters={
                        "last_n": {"type": "integer", "required": False, "description": "Only analyze last N records (default: all)"},
                    },
                ),
                SkillAction(
                    name="skill_stats",
                    description="Get detailed performance stats for a specific skill",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill ID to analyze"},
                    },
                ),
                SkillAction(
                    name="recommendations",
                    description="Get data-driven recommendations for improving performance",
                    parameters={},
                ),
                SkillAction(
                    name="failure_patterns",
                    description="Analyze failure patterns to identify recurring issues",
                    parameters={
                        "min_failures": {"type": "integer", "required": False, "description": "Minimum failures to report (default: 2)"},
                    },
                ),
                SkillAction(
                    name="reset",
                    description="Clear all performance data",
                    parameters={
                        "confirm": {"type": "boolean", "required": True, "description": "Must be true to confirm reset"},
                    },
                ),
            ],
            required_credentials=[],
        )

    def set_persist_path(self, path: str):
        """Set the path for persisting performance data."""
        self._persist_path = Path(path)
        self._load()

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "record": self._record,
            "report": self._report,
            "skill_stats": self._skill_stats,
            "recommendations": self._recommendations,
            "failure_patterns": self._failure_patterns,
            "reset": self._reset,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")

        return await handler(params)

    async def _record(self, params: Dict) -> SkillResult:
        """Record an action outcome."""
        skill_id = params.get("skill_id", "")
        action_name = params.get("action_name", "")

        if not skill_id or not action_name:
            return SkillResult(success=False, message="skill_id and action_name are required")

        record = ActionRecord(
            skill_id=skill_id,
            action_name=action_name,
            success=bool(params.get("success", False)),
            duration_seconds=float(params.get("duration_seconds", 0)),
            cost_usd=float(params.get("cost_usd", 0)),
            error_message=str(params.get("error_message", "")),
            cycle=int(params.get("cycle", 0)),
        )

        self._records.append(record)
        self._save()

        return SkillResult(
            success=True,
            message=f"Recorded {'success' if record.success else 'failure'} for {skill_id}:{action_name}",
            data={"total_records": len(self._records)},
        )

    def record_sync(self, skill_id: str, action_name: str, success: bool,
                    duration_seconds: float = 0, cost_usd: float = 0,
                    error_message: str = "", cycle: int = 0):
        """Synchronous record method for use in the agent's run loop."""
        record = ActionRecord(
            skill_id=skill_id,
            action_name=action_name,
            success=success,
            duration_seconds=duration_seconds,
            cost_usd=cost_usd,
            error_message=error_message,
            cycle=cycle,
        )
        self._records.append(record)
        self._save()

    async def _report(self, params: Dict) -> SkillResult:
        """Generate comprehensive performance report."""
        last_n = params.get("last_n")
        records = self._records[-last_n:] if last_n else self._records

        if not records:
            return SkillResult(
                success=True,
                message="No performance data recorded yet",
                data={"total_records": 0},
            )

        # Overall stats
        total = len(records)
        successes = sum(1 for r in records if r.success)
        failures = total - successes
        success_rate = successes / total if total > 0 else 0

        total_cost = sum(r.cost_usd for r in records)
        total_duration = sum(r.duration_seconds for r in records)
        avg_duration = total_duration / total if total > 0 else 0

        # Per-skill breakdown
        skill_data = defaultdict(lambda: {"total": 0, "successes": 0, "cost": 0.0, "duration": 0.0})
        for r in records:
            sd = skill_data[r.skill_id]
            sd["total"] += 1
            sd["successes"] += 1 if r.success else 0
            sd["cost"] += r.cost_usd
            sd["duration"] += r.duration_seconds

        skill_summary = {}
        for sid, sd in skill_data.items():
            skill_summary[sid] = {
                "total_actions": sd["total"],
                "success_rate": sd["successes"] / sd["total"] if sd["total"] > 0 else 0,
                "total_cost": round(sd["cost"], 6),
                "avg_duration": round(sd["duration"] / sd["total"], 3) if sd["total"] > 0 else 0,
            }

        # Most/least used
        sorted_by_usage = sorted(skill_summary.items(), key=lambda x: x[1]["total_actions"], reverse=True)
        most_used = sorted_by_usage[:3] if sorted_by_usage else []
        least_reliable = sorted(
            [(k, v) for k, v in skill_summary.items() if v["total_actions"] >= 2],
            key=lambda x: x[1]["success_rate"]
        )[:3]

        report = {
            "total_records": total,
            "overall_success_rate": round(success_rate, 4),
            "total_successes": successes,
            "total_failures": failures,
            "total_cost_usd": round(total_cost, 6),
            "total_duration_seconds": round(total_duration, 3),
            "avg_duration_seconds": round(avg_duration, 3),
            "skills_used": len(skill_data),
            "skill_breakdown": skill_summary,
            "most_used_skills": [{"skill": k, **v} for k, v in most_used],
            "least_reliable_skills": [{"skill": k, **v} for k, v in least_reliable],
            "session_start": self._session_start,
        }

        return SkillResult(
            success=True,
            message=f"Performance report: {total} actions, {success_rate:.1%} success rate, ${total_cost:.6f} total cost",
            data=report,
        )

    async def _skill_stats(self, params: Dict) -> SkillResult:
        """Get detailed stats for a specific skill."""
        target_skill = params.get("skill_id", "")
        if not target_skill:
            return SkillResult(success=False, message="skill_id is required")

        records = [r for r in self._records if r.skill_id == target_skill]
        if not records:
            return SkillResult(
                success=True,
                message=f"No records found for skill: {target_skill}",
                data={"skill_id": target_skill, "total_records": 0},
            )

        total = len(records)
        successes = sum(1 for r in records if r.success)

        # Per-action breakdown
        action_data = defaultdict(lambda: {"total": 0, "successes": 0, "failures": 0, "errors": []})
        for r in records:
            ad = action_data[r.action_name]
            ad["total"] += 1
            if r.success:
                ad["successes"] += 1
            else:
                ad["failures"] += 1
                if r.error_message:
                    ad["errors"].append(r.error_message)

        action_breakdown = {}
        for aname, ad in action_data.items():
            action_breakdown[aname] = {
                "total": ad["total"],
                "success_rate": ad["successes"] / ad["total"] if ad["total"] > 0 else 0,
                "failures": ad["failures"],
                "recent_errors": ad["errors"][-3:],  # Last 3 errors
            }

        # Trend: split records into first half and second half
        mid = total // 2
        if mid > 0:
            first_half_rate = sum(1 for r in records[:mid] if r.success) / mid
            second_half_rate = sum(1 for r in records[mid:] if r.success) / (total - mid)
            trend = "improving" if second_half_rate > first_half_rate + 0.05 else (
                "declining" if second_half_rate < first_half_rate - 0.05 else "stable"
            )
        else:
            trend = "insufficient_data"

        stats = {
            "skill_id": target_skill,
            "total_actions": total,
            "success_rate": round(successes / total, 4),
            "total_cost": round(sum(r.cost_usd for r in records), 6),
            "avg_duration": round(sum(r.duration_seconds for r in records) / total, 3),
            "action_breakdown": action_breakdown,
            "trend": trend,
        }

        return SkillResult(
            success=True,
            message=f"Stats for {target_skill}: {total} actions, {successes/total:.1%} success rate, trend: {trend}",
            data=stats,
        )

    async def _recommendations(self, params: Dict) -> SkillResult:
        """Generate data-driven recommendations."""
        if len(self._records) < 5:
            return SkillResult(
                success=True,
                message="Need at least 5 recorded actions to generate recommendations",
                data={"recommendations": [], "reason": "insufficient_data"},
            )

        recommendations = []

        # Analyze per-skill success rates
        skill_stats = defaultdict(lambda: {"total": 0, "successes": 0, "cost": 0.0})
        for r in self._records:
            ss = skill_stats[r.skill_id]
            ss["total"] += 1
            ss["successes"] += 1 if r.success else 0
            ss["cost"] += r.cost_usd

        for sid, ss in skill_stats.items():
            rate = ss["successes"] / ss["total"] if ss["total"] > 0 else 0

            # Low success rate warning
            if ss["total"] >= 3 and rate < 0.5:
                recommendations.append({
                    "type": "warning",
                    "priority": "high",
                    "skill": sid,
                    "message": f"Skill '{sid}' has a {rate:.0%} success rate over {ss['total']} actions. Consider debugging or avoiding this skill.",
                })

            # High cost alert
            if ss["cost"] > 0 and ss["total"] >= 2:
                avg_cost = ss["cost"] / ss["total"]
                if avg_cost > 0.05:
                    recommendations.append({
                        "type": "cost_alert",
                        "priority": "medium",
                        "skill": sid,
                        "message": f"Skill '{sid}' costs ${avg_cost:.4f}/action. Consider cheaper alternatives.",
                    })

        # Check for error patterns
        error_skills = defaultdict(list)
        for r in self._records:
            if not r.success and r.error_message:
                error_skills[r.skill_id].append(r.error_message)

        for sid, errors in error_skills.items():
            if len(errors) >= 3:
                # Check for repeated errors
                error_counts = defaultdict(int)
                for e in errors:
                    # Normalize error messages (first 50 chars)
                    key = e[:50]
                    error_counts[key] += 1

                for err_key, count in error_counts.items():
                    if count >= 2:
                        recommendations.append({
                            "type": "recurring_error",
                            "priority": "high",
                            "skill": sid,
                            "message": f"Recurring error in '{sid}' ({count}x): {err_key}...",
                        })

        # Overall trend
        total = len(self._records)
        mid = total // 2
        if mid >= 3:
            first_rate = sum(1 for r in self._records[:mid] if r.success) / mid
            second_rate = sum(1 for r in self._records[mid:] if r.success) / (total - mid)

            if second_rate < first_rate - 0.1:
                recommendations.append({
                    "type": "trend",
                    "priority": "high",
                    "message": f"Overall success rate is declining: {first_rate:.0%} → {second_rate:.0%}. Review recent failures.",
                })
            elif second_rate > first_rate + 0.1:
                recommendations.append({
                    "type": "trend",
                    "priority": "low",
                    "message": f"Overall success rate is improving: {first_rate:.0%} → {second_rate:.0%}. Current approach is working.",
                })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 2))

        return SkillResult(
            success=True,
            message=f"Generated {len(recommendations)} recommendations based on {total} recorded actions",
            data={"recommendations": recommendations, "based_on_records": total},
        )

    async def _failure_patterns(self, params: Dict) -> SkillResult:
        """Analyze failure patterns."""
        min_failures = int(params.get("min_failures", 2))

        failures = [r for r in self._records if not r.success]
        if not failures:
            return SkillResult(
                success=True,
                message="No failures recorded",
                data={"total_failures": 0, "patterns": []},
            )

        # Group by skill:action
        failure_groups = defaultdict(lambda: {"count": 0, "errors": [], "timestamps": []})
        for r in failures:
            key = f"{r.skill_id}:{r.action_name}"
            fg = failure_groups[key]
            fg["count"] += 1
            if r.error_message:
                fg["errors"].append(r.error_message)
            fg["timestamps"].append(r.timestamp)

        patterns = []
        for key, fg in failure_groups.items():
            if fg["count"] >= min_failures:
                # Find common error substrings
                common_errors = defaultdict(int)
                for e in fg["errors"]:
                    common_errors[e[:80]] += 1

                most_common = sorted(common_errors.items(), key=lambda x: -x[1])[:3]

                patterns.append({
                    "action": key,
                    "failure_count": fg["count"],
                    "common_errors": [{"message": e, "count": c} for e, c in most_common],
                    "first_seen": fg["timestamps"][0] if fg["timestamps"] else "",
                    "last_seen": fg["timestamps"][-1] if fg["timestamps"] else "",
                })

        patterns.sort(key=lambda p: -p["failure_count"])

        return SkillResult(
            success=True,
            message=f"Found {len(patterns)} failure patterns from {len(failures)} total failures",
            data={"total_failures": len(failures), "patterns": patterns},
        )

    async def _reset(self, params: Dict) -> SkillResult:
        """Reset all performance data."""
        if not params.get("confirm"):
            return SkillResult(success=False, message="Must set confirm=true to reset data")

        count = len(self._records)
        self._records = []
        self._session_start = datetime.now().isoformat()
        self._save()

        return SkillResult(
            success=True,
            message=f"Reset performance data. Cleared {count} records.",
            data={"cleared_records": count},
        )

    def _save(self):
        """Persist records to disk."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "session_start": self._session_start,
                "records": [asdict(r) for r in self._records[-1000:]],  # Keep last 1000
            }
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        """Load records from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)
            self._session_start = data.get("session_start", self._session_start)
            for rec_data in data.get("records", []):
                self._records.append(ActionRecord(**rec_data))
        except Exception:
            pass
