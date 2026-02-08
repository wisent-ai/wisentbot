#!/usr/bin/env python3
"""
SkillExecutionInstrumenter - Automatic metrics + event bridge integration for every skill execution.

This is the missing integration layer that wires ObservabilitySkill and SkillEventBridgeSkill
into the agent's skill execution path. When installed, it automatically:

1. METRICS: Emits observability metrics for every skill execution:
   - skill.execution.count (counter) - total executions per skill/action/status
   - skill.execution.latency (histogram) - execution time per skill/action
   - skill.execution.errors (counter) - error count per skill/action
   - skill.execution.active (gauge) - currently executing skills

2. BRIDGE EVENTS: Calls SkillEventBridge.emit_bridge_events() after each execution,
   enabling reactive cross-skill automation (e.g., health scan -> auto-incident).

3. ALERT CHECKING: Periodically triggers ObservabilitySkill.check_alerts() to evaluate
   threshold rules against accumulated metrics.

This transforms the agent from executing skills in isolation to a fully instrumented
system where every action is measured, events flow between skills, and alerts fire
when metrics breach thresholds.

Pillar: Self-Improvement (closes the act->measure->adapt loop with real metrics)

Actions:
- instrument: Wrap a skill execution with metrics + bridge events (called by agent)
- configure: Set which metrics to emit, alert check frequency
- stats: View instrumentation statistics (total instrumented, metrics emitted, events bridged)
- recent: View recent instrumented executions with latency/status
- top_skills: Rank skills by execution count, latency, error rate
- health: Check instrumentation health (are observability + bridge skills available?)
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


DATA_FILE = Path(__file__).parent.parent / "data" / "execution_instrumenter.json"

# Default config
DEFAULT_CONFIG = {
    "emit_count_metric": True,
    "emit_latency_metric": True,
    "emit_error_metric": True,
    "emit_active_gauge": True,
    "emit_bridge_events": True,
    "alert_check_interval": 50,  # Check alerts every N executions
    "max_recent_entries": 200,
    "excluded_skills": ["execution_instrumenter"],  # Don't instrument ourselves
}


class SkillExecutionInstrumenter(Skill):
    """Automatic metrics + event bridge integration for skill executions."""

    def __init__(self, credentials: Optional[Dict] = None):
        super().__init__(credentials or {})
        self._execution_count = 0

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="execution_instrumenter",
            name="Skill Execution Instrumenter",
            description="Automatic observability metrics and event bridge integration for every skill execution",
            version="1.0.0",
            category="infrastructure",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="instrument",
                    description="Instrument a skill execution with metrics and bridge events. Called by agent execution layer.",
                    parameters={
                        "skill_id": "str - The skill that was executed",
                        "action": "str - The action that was executed",
                        "success": "bool - Whether the execution succeeded",
                        "latency_ms": "float - Execution time in milliseconds",
                        "result_data": "dict - The result data from the execution (optional, for bridge events)",
                        "error": "str - Error message if failed (optional)",
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="configure",
                    description="Configure instrumentation settings",
                    parameters={
                        "emit_count_metric": "bool - Emit execution count metrics (default: true)",
                        "emit_latency_metric": "bool - Emit latency histogram metrics (default: true)",
                        "emit_error_metric": "bool - Emit error count metrics (default: true)",
                        "emit_bridge_events": "bool - Emit bridge events after execution (default: true)",
                        "alert_check_interval": "int - Check alerts every N executions (default: 50)",
                        "excluded_skills": "list - Skills to exclude from instrumentation",
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="stats",
                    description="View instrumentation statistics",
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="recent",
                    description="View recent instrumented executions",
                    parameters={
                        "limit": "int - Number of recent entries to show (default: 20)",
                        "skill_filter": "str - Filter by skill ID (optional)",
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="top_skills",
                    description="Rank skills by execution count, latency, or error rate",
                    parameters={
                        "sort_by": "str - Sort by: count, latency, errors, error_rate (default: count)",
                        "limit": "int - Number of top skills to show (default: 10)",
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="health",
                    description="Check instrumentation health - are observability and bridge skills available?",
                    parameters={},
                    estimated_cost=0.0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "instrument": self._instrument,
            "configure": self._configure,
            "stats": self._stats,
            "recent": self._recent,
            "top_skills": self._top_skills,
            "health": self._health,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Action: instrument ─────────────────────────────────────────

    async def _instrument(self, params: Dict) -> SkillResult:
        """Instrument a skill execution with metrics and bridge events."""
        skill_id = params.get("skill_id", "")
        action = params.get("action", "")
        success = params.get("success", True)
        latency_ms = params.get("latency_ms", 0.0)
        result_data = params.get("result_data", {})
        error = params.get("error", "")

        if not skill_id or not action:
            return SkillResult(success=False, message="skill_id and action are required")

        store = self._load()
        config = store.get("config", DEFAULT_CONFIG.copy())

        # Skip excluded skills
        if skill_id in config.get("excluded_skills", []):
            return SkillResult(
                success=True,
                message=f"Skipped instrumentation for excluded skill: {skill_id}",
                data={"skipped": True},
            )

        status_label = "success" if success else "error"
        metrics_emitted = []
        bridge_events = []
        alerts_checked = False

        # ── Emit metrics via ObservabilitySkill ─────────────────

        if self.context:
            # 1. Execution count (counter)
            if config.get("emit_count_metric", True):
                try:
                    await self.context.call_skill("observability", "emit", {
                        "name": "skill.execution.count",
                        "value": 1,
                        "metric_type": "counter",
                        "labels": {
                            "skill_id": skill_id,
                            "action": action,
                            "status": status_label,
                        },
                    })
                    metrics_emitted.append("skill.execution.count")
                except Exception:
                    pass  # Observability skill not available

            # 2. Latency histogram
            if config.get("emit_latency_metric", True) and latency_ms > 0:
                try:
                    await self.context.call_skill("observability", "emit", {
                        "name": "skill.execution.latency_ms",
                        "value": latency_ms,
                        "metric_type": "histogram",
                        "labels": {
                            "skill_id": skill_id,
                            "action": action,
                        },
                    })
                    metrics_emitted.append("skill.execution.latency_ms")
                except Exception:
                    pass

            # 3. Error count (counter - only on failure)
            if config.get("emit_error_metric", True) and not success:
                try:
                    await self.context.call_skill("observability", "emit", {
                        "name": "skill.execution.errors",
                        "value": 1,
                        "metric_type": "counter",
                        "labels": {
                            "skill_id": skill_id,
                            "action": action,
                            "error_type": error[:100] if error else "unknown",
                        },
                    })
                    metrics_emitted.append("skill.execution.errors")
                except Exception:
                    pass

            # ── Emit bridge events via SkillEventBridgeSkill ────

            if config.get("emit_bridge_events", True) and result_data:
                try:
                    bridge_result = await self.context.call_skill(
                        "skill_event_bridge", "trigger", {
                            "topic": f"skill.{skill_id}.{action}",
                            "data": {
                                "skill_id": skill_id,
                                "action": action,
                                "success": success,
                                "latency_ms": latency_ms,
                                **(result_data if isinstance(result_data, dict) else {}),
                            },
                        }
                    )
                    if bridge_result and hasattr(bridge_result, "data"):
                        bridge_events.append(f"skill.{skill_id}.{action}")
                except Exception:
                    pass

                # Also call emit_bridge_events for structured bridge matching
                try:
                    bridge_skill = self.context.registry.get("skill_event_bridge") if self.context else None
                    if bridge_skill and hasattr(bridge_skill, "emit_bridge_events"):
                        emitted = await bridge_skill.emit_bridge_events(
                            skill_id=skill_id,
                            action=action,
                            result_data=result_data if isinstance(result_data, dict) else {},
                        )
                        bridge_events.extend([e.get("topic", "") for e in (emitted or [])])
                except Exception:
                    pass

            # ── Periodic alert checking ─────────────────────────

            self._execution_count += 1
            interval = config.get("alert_check_interval", 50)
            if interval > 0 and self._execution_count % interval == 0:
                try:
                    await self.context.call_skill("observability", "check_alerts", {})
                    alerts_checked = True
                except Exception:
                    pass

        # ── Record execution in local store ─────────────────────

        entry = {
            "skill_id": skill_id,
            "action": action,
            "success": success,
            "latency_ms": round(latency_ms, 2),
            "status": status_label,
            "metrics_emitted": metrics_emitted,
            "bridge_events": bridge_events,
            "alerts_checked": alerts_checked,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if error:
            entry["error"] = error[:200]

        # Update recent executions
        recent = store.get("recent", [])
        recent.append(entry)
        max_recent = config.get("max_recent_entries", 200)
        if len(recent) > max_recent:
            recent = recent[-max_recent:]
        store["recent"] = recent

        # Update per-skill stats
        skill_stats = store.get("skill_stats", {})
        key = f"{skill_id}.{action}"
        if key not in skill_stats:
            skill_stats[key] = {
                "skill_id": skill_id,
                "action": action,
                "total_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_latency_ms": 0.0,
                "min_latency_ms": float("inf"),
                "max_latency_ms": 0.0,
                "last_executed": None,
            }
        stats = skill_stats[key]
        stats["total_count"] += 1
        if success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1
        if latency_ms > 0:
            stats["total_latency_ms"] += latency_ms
            stats["min_latency_ms"] = min(stats["min_latency_ms"], latency_ms)
            stats["max_latency_ms"] = max(stats["max_latency_ms"], latency_ms)
        stats["last_executed"] = entry["timestamp"]
        skill_stats[key] = stats
        store["skill_stats"] = skill_stats

        # Update global stats
        global_stats = store.get("global_stats", {
            "total_instrumented": 0,
            "total_metrics_emitted": 0,
            "total_bridge_events": 0,
            "total_alerts_checked": 0,
        })
        global_stats["total_instrumented"] += 1
        global_stats["total_metrics_emitted"] += len(metrics_emitted)
        global_stats["total_bridge_events"] += len(bridge_events)
        if alerts_checked:
            global_stats["total_alerts_checked"] += 1
        store["global_stats"] = global_stats

        self._save(store)

        return SkillResult(
            success=True,
            message=f"Instrumented {skill_id}:{action} ({status_label}). "
                    f"{len(metrics_emitted)} metric(s), {len(bridge_events)} bridge event(s)."
                    f"{' Alerts checked.' if alerts_checked else ''}",
            data=entry,
        )

    # ── Action: configure ──────────────────────────────────────────

    async def _configure(self, params: Dict) -> SkillResult:
        """Update instrumentation config."""
        store = self._load()
        config = store.get("config", DEFAULT_CONFIG.copy())

        updated = []
        for key in DEFAULT_CONFIG:
            if key in params:
                config[key] = params[key]
                updated.append(key)

        store["config"] = config
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} config setting(s)." if updated else "No changes.",
            data={"config": config, "updated": updated},
        )

    # ── Action: stats ──────────────────────────────────────────────

    async def _stats(self, params: Dict) -> SkillResult:
        """View instrumentation statistics."""
        store = self._load()
        global_stats = store.get("global_stats", {
            "total_instrumented": 0,
            "total_metrics_emitted": 0,
            "total_bridge_events": 0,
            "total_alerts_checked": 0,
        })
        config = store.get("config", DEFAULT_CONFIG.copy())
        skill_stats = store.get("skill_stats", {})

        unique_skills = set()
        unique_actions = set()
        for key, ss in skill_stats.items():
            unique_skills.add(ss["skill_id"])
            unique_actions.add(key)

        return SkillResult(
            success=True,
            message=f"Instrumented {global_stats['total_instrumented']} execution(s) across "
                    f"{len(unique_skills)} skill(s), {len(unique_actions)} action(s). "
                    f"{global_stats['total_metrics_emitted']} metrics emitted, "
                    f"{global_stats['total_bridge_events']} bridge events.",
            data={
                "global_stats": global_stats,
                "unique_skills": len(unique_skills),
                "unique_actions": len(unique_actions),
                "config": config,
                "session_execution_count": self._execution_count,
            },
        )

    # ── Action: recent ─────────────────────────────────────────────

    async def _recent(self, params: Dict) -> SkillResult:
        """View recent instrumented executions."""
        limit = params.get("limit", 20)
        skill_filter = params.get("skill_filter")

        store = self._load()
        recent = store.get("recent", [])

        if skill_filter:
            recent = [e for e in recent if e.get("skill_id") == skill_filter]

        entries = recent[-limit:]

        return SkillResult(
            success=True,
            message=f"{len(entries)} recent execution(s)" +
                    (f" for {skill_filter}" if skill_filter else "") + ".",
            data={
                "executions": entries,
                "total_recorded": len(recent),
            },
        )

    # ── Action: top_skills ─────────────────────────────────────────

    async def _top_skills(self, params: Dict) -> SkillResult:
        """Rank skills by various metrics."""
        sort_by = params.get("sort_by", "count")
        limit = params.get("limit", 10)

        store = self._load()
        skill_stats = store.get("skill_stats", {})

        rankings = []
        for key, ss in skill_stats.items():
            total = ss.get("total_count", 0)
            errors = ss.get("error_count", 0)
            total_lat = ss.get("total_latency_ms", 0)
            avg_latency = total_lat / total if total > 0 else 0
            error_rate = errors / total if total > 0 else 0
            min_lat = ss.get("min_latency_ms", 0)
            if min_lat == float("inf"):
                min_lat = 0

            rankings.append({
                "skill_action": key,
                "skill_id": ss["skill_id"],
                "action": ss["action"],
                "total_count": total,
                "success_count": ss.get("success_count", 0),
                "error_count": errors,
                "error_rate": round(error_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min_lat, 2),
                "max_latency_ms": round(ss.get("max_latency_ms", 0), 2),
                "last_executed": ss.get("last_executed"),
            })

        sort_keys = {
            "count": lambda r: r["total_count"],
            "latency": lambda r: r["avg_latency_ms"],
            "errors": lambda r: r["error_count"],
            "error_rate": lambda r: r["error_rate"],
        }
        sort_fn = sort_keys.get(sort_by, sort_keys["count"])
        rankings.sort(key=sort_fn, reverse=True)

        return SkillResult(
            success=True,
            message=f"Top {min(limit, len(rankings))} skill actions by {sort_by}.",
            data={
                "rankings": rankings[:limit],
                "sort_by": sort_by,
                "total_tracked": len(rankings),
            },
        )

    # ── Action: health ─────────────────────────────────────────────

    async def _health(self, params: Dict) -> SkillResult:
        """Check whether the required skills are available for instrumentation."""
        observability_ok = False
        bridge_ok = False
        details = {}

        if self.context:
            # Check ObservabilitySkill
            obs_skill = self.context.registry.get("observability") if self.context.registry else None
            observability_ok = obs_skill is not None
            details["observability"] = "available" if observability_ok else "not installed"

            # Check SkillEventBridgeSkill
            bridge_skill = self.context.registry.get("skill_event_bridge") if self.context.registry else None
            bridge_ok = bridge_skill is not None
            details["skill_event_bridge"] = "available" if bridge_ok else "not installed"
        else:
            details["context"] = "no skill context - metrics and bridge events disabled"

        all_ok = observability_ok and bridge_ok
        store = self._load()

        return SkillResult(
            success=True,
            message=f"Instrumentation {'fully operational' if all_ok else 'degraded'}. "
                    f"Observability: {'OK' if observability_ok else 'MISSING'}. "
                    f"EventBridge: {'OK' if bridge_ok else 'MISSING'}.",
            data={
                "healthy": all_ok,
                "observability_available": observability_ok,
                "bridge_available": bridge_ok,
                "details": details,
                "config": store.get("config", DEFAULT_CONFIG.copy()),
                "session_execution_count": self._execution_count,
            },
        )

    # ── Persistence ────────────────────────────────────────────────

    def _load(self) -> Dict:
        if DATA_FILE.exists():
            try:
                return json.loads(DATA_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "config": DEFAULT_CONFIG.copy(),
            "recent": [],
            "skill_stats": {},
            "global_stats": {
                "total_instrumented": 0,
                "total_metrics_emitted": 0,
                "total_bridge_events": 0,
                "total_alerts_checked": 0,
            },
        }

    def _save(self, store: Dict):
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text(json.dumps(store, indent=2, default=str))
