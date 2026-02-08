#!/usr/bin/env python3
"""
GoalProgressEventBridgeSkill - Emit EventBus events when goals transition states.

When GoalManagerSkill creates, progresses, completes, or abandons goals, this
bridge emits structured EventBus events so downstream skills can react:

Events emitted:
- goal.created: A new goal was created
- goal.milestone_completed: A milestone within a goal was completed
- goal.completed: A goal was fully completed
- goal.abandoned: A goal was abandoned
- goal.progress_stalled: A goal hasn't progressed within its deadline window
- goal.pillar_shift: The distribution of active goals across pillars changed significantly

This enables reactive automation:
- StrategySkill can reprioritize when goals are completed or abandoned
- RevenueGoalAutoSetter can react when revenue goals complete/stall
- ExperimentSkill can correlate experiment outcomes with goal progress
- AutonomousLoop can adjust focus based on goal state transitions
- AlertIncidentBridge can flag stalled critical goals

Architecture:
  GoalManager acts → GoalProgressEventBridge detects changes →
  EventBus emits events → downstream skills react

Pillar: Goal Setting (primary) + Self-Improvement (reactive goal management)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
BRIDGE_STATE_FILE = DATA_DIR / "goal_progress_events.json"
MAX_EVENT_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class GoalProgressEventBridgeSkill(Skill):
    """
    Bridge between GoalManagerSkill and EventBus.

    Monitors goal state changes and emits structured events so the rest
    of the agent ecosystem can react to goal transitions.

    Actions:
    - monitor: Check goal state for changes since last call and emit events
    - configure: Update event emission settings
    - status: View bridge health and emission statistics
    - history: View recent emitted events
    - emit_test: Emit a test event to verify EventBus integration
    - stall_check: Detect goals that haven't progressed and may be stalled
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._load_state()

    def _load_state(self):
        """Load persisted bridge state from disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if BRIDGE_STATE_FILE.exists():
            try:
                with open(BRIDGE_STATE_FILE) as f:
                    data = json.load(f)
                self._last_snapshot = data.get("last_snapshot", {})
                self._event_history = data.get("event_history", [])[-MAX_EVENT_HISTORY:]
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._last_snapshot: Dict = {
            "goal_ids": [],
            "completed_ids": [],
            "abandoned_ids": [],
            "milestone_states": {},
            "pillar_distribution": {},
            "last_monitor_ts": None,
        }
        self._event_history: List[Dict] = []
        self._config = self._default_config()
        self._stats = self._default_stats()

    def _default_config(self) -> Dict:
        return {
            "emit_on_created": True,
            "emit_on_milestone_completed": True,
            "emit_on_completed": True,
            "emit_on_abandoned": True,
            "emit_on_stalled": True,
            "emit_on_pillar_shift": True,
            "stall_threshold_hours": 24,
            "pillar_shift_threshold": 0.2,
            "event_source": "goal_progress_event_bridge",
            "priority_created": "normal",
            "priority_milestone_completed": "normal",
            "priority_completed": "high",
            "priority_abandoned": "high",
            "priority_stalled": "high",
            "priority_pillar_shift": "normal",
        }

    def _default_stats(self) -> Dict:
        return {
            "events_emitted": 0,
            "events_failed": 0,
            "monitors_run": 0,
            "stall_checks_run": 0,
            "goals_created_detected": 0,
            "milestones_completed_detected": 0,
            "goals_completed_detected": 0,
            "goals_abandoned_detected": 0,
            "stalls_detected": 0,
            "pillar_shifts_detected": 0,
            "last_monitor_time": None,
            "last_stall_check_time": None,
        }

    def _save_state(self):
        """Persist bridge state to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "last_snapshot": self._last_snapshot,
            "event_history": self._event_history[-MAX_EVENT_HISTORY:],
            "config": self._config,
            "stats": self._stats,
            "last_updated": _now_iso(),
        }
        with open(BRIDGE_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="goal_progress_events",
            name="Goal Progress Event Bridge",
            version="1.0.0",
            category="meta",
            description=(
                "Emit EventBus events when goals transition states "
                "(created, progressing, completed, abandoned, stalled). "
                "Enables reactive automation based on goal lifecycle."
            ),
            actions=[
                SkillAction(
                    name="monitor",
                    description=(
                        "Check goal state for changes since last monitor "
                        "call and emit events for new goals, completed "
                        "milestones, completed/abandoned goals, and pillar shifts."
                    ),
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update event emission settings and thresholds",
                    parameters={
                        "emit_on_created": {
                            "type": "bool", "required": False,
                            "description": "Emit when a new goal is created",
                        },
                        "emit_on_milestone_completed": {
                            "type": "bool", "required": False,
                            "description": "Emit when a milestone is completed",
                        },
                        "emit_on_completed": {
                            "type": "bool", "required": False,
                            "description": "Emit when a goal is completed",
                        },
                        "emit_on_abandoned": {
                            "type": "bool", "required": False,
                            "description": "Emit when a goal is abandoned",
                        },
                        "emit_on_stalled": {
                            "type": "bool", "required": False,
                            "description": "Emit when a goal appears stalled",
                        },
                        "emit_on_pillar_shift": {
                            "type": "bool", "required": False,
                            "description": "Emit when pillar distribution shifts significantly",
                        },
                        "stall_threshold_hours": {
                            "type": "float", "required": False,
                            "description": "Hours without progress before a goal is considered stalled",
                        },
                        "pillar_shift_threshold": {
                            "type": "float", "required": False,
                            "description": "Min fraction change in pillar distribution to trigger event (0-1)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="View bridge health, emission statistics, and tracked goals",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View recent emitted events",
                    parameters={
                        "limit": {
                            "type": "int", "required": False,
                            "description": "Max events to return (default 20)",
                        },
                        "topic_filter": {
                            "type": "str", "required": False,
                            "description": "Filter by event topic prefix",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="emit_test",
                    description="Emit a test event to verify EventBus integration",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stall_check",
                    description=(
                        "Detect goals that haven't progressed within their "
                        "threshold and emit stall events."
                    ),
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "monitor": self._monitor,
            "configure": self._configure,
            "status": self._status,
            "history": self._history,
            "emit_test": self._emit_test,
            "stall_check": self._stall_check,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    async def _monitor(self, params: Dict) -> SkillResult:
        """Check goal state for changes and emit events."""
        self._stats["monitors_run"] += 1
        self._stats["last_monitor_time"] = _now_iso()

        events_emitted = 0
        events_detail = []

        # Get current goals state
        goals_data = await self._get_goals_state()
        if goals_data is None:
            self._save_state()
            return SkillResult(
                success=True,
                message="Goals skill not available, no events emitted",
                data={"events_emitted": 0},
            )

        active_goals = goals_data.get("goals", [])
        completed_goals = goals_data.get("completed_goals", [])

        current_active_ids = {g["id"] for g in active_goals}
        current_completed = {g["id"] for g in completed_goals if g.get("status") == "completed"}
        current_abandoned = {g["id"] for g in completed_goals if g.get("status") == "abandoned"}

        prev_active_ids = set(self._last_snapshot.get("goal_ids", []))
        prev_completed = set(self._last_snapshot.get("completed_ids", []))
        prev_abandoned = set(self._last_snapshot.get("abandoned_ids", []))

        # Detect new goals (in active but not in previous active or completed/abandoned)
        all_prev = prev_active_ids | prev_completed | prev_abandoned
        new_goal_ids = current_active_ids - all_prev
        if new_goal_ids and self._config["emit_on_created"]:
            for gid in new_goal_ids:
                goal = next((g for g in active_goals if g["id"] == gid), None)
                if goal:
                    self._stats["goals_created_detected"] += 1
                    emitted = await self._emit_event(
                        "goal.created",
                        {
                            "goal_id": goal["id"],
                            "title": goal.get("title", ""),
                            "pillar": goal.get("pillar", "other"),
                            "priority": goal.get("priority", "medium"),
                            "milestones_count": len(goal.get("milestones", [])),
                            "deadline": goal.get("deadline"),
                            "timestamp": _now_iso(),
                        },
                        self._config["priority_created"],
                    )
                    if emitted:
                        events_emitted += 1
                        events_detail.append(f"created:{goal.get('title', gid)}")

        # Detect newly completed goals
        new_completed = current_completed - prev_completed
        if new_completed and self._config["emit_on_completed"]:
            for gid in new_completed:
                goal = next((g for g in completed_goals if g["id"] == gid), None)
                if goal:
                    self._stats["goals_completed_detected"] += 1
                    emitted = await self._emit_event(
                        "goal.completed",
                        {
                            "goal_id": goal["id"],
                            "title": goal.get("title", ""),
                            "pillar": goal.get("pillar", "other"),
                            "priority": goal.get("priority", "medium"),
                            "outcome": goal.get("outcome", ""),
                            "duration_hours": goal.get("duration_hours"),
                            "milestones_total": len(goal.get("milestones", [])),
                            "timestamp": _now_iso(),
                        },
                        self._config["priority_completed"],
                    )
                    if emitted:
                        events_emitted += 1
                        events_detail.append(f"completed:{goal.get('title', gid)}")

        # Detect newly abandoned goals
        new_abandoned = current_abandoned - prev_abandoned
        if new_abandoned and self._config["emit_on_abandoned"]:
            for gid in new_abandoned:
                goal = next((g for g in completed_goals if g["id"] == gid), None)
                if goal:
                    self._stats["goals_abandoned_detected"] += 1
                    emitted = await self._emit_event(
                        "goal.abandoned",
                        {
                            "goal_id": goal["id"],
                            "title": goal.get("title", ""),
                            "pillar": goal.get("pillar", "other"),
                            "priority": goal.get("priority", "medium"),
                            "reason": goal.get("abandon_reason", ""),
                            "timestamp": _now_iso(),
                        },
                        self._config["priority_abandoned"],
                    )
                    if emitted:
                        events_emitted += 1
                        events_detail.append(f"abandoned:{goal.get('title', gid)}")

        # Detect milestone completions
        prev_milestones = self._last_snapshot.get("milestone_states", {})
        if self._config["emit_on_milestone_completed"]:
            for goal in active_goals:
                gid = goal["id"]
                milestones = goal.get("milestones", [])
                prev_completed_ms = set(prev_milestones.get(gid, []))
                for ms in milestones:
                    ms_idx = ms.get("index", 0)
                    if ms.get("completed") and ms_idx not in prev_completed_ms:
                        self._stats["milestones_completed_detected"] += 1
                        done = sum(1 for m in milestones if m.get("completed"))
                        total = len(milestones)
                        emitted = await self._emit_event(
                            "goal.milestone_completed",
                            {
                                "goal_id": gid,
                                "goal_title": goal.get("title", ""),
                                "milestone_index": ms_idx,
                                "milestone_title": ms.get("title", ""),
                                "progress": f"{done}/{total}",
                                "pillar": goal.get("pillar", "other"),
                                "timestamp": _now_iso(),
                            },
                            self._config["priority_milestone_completed"],
                        )
                        if emitted:
                            events_emitted += 1
                            events_detail.append(f"milestone:{ms.get('title', ms_idx)}")

        # Detect pillar distribution shifts
        current_pillar_dist = self._calc_pillar_distribution(active_goals)
        prev_pillar_dist = self._last_snapshot.get("pillar_distribution", {})
        if prev_pillar_dist and self._config["emit_on_pillar_shift"]:
            shift = self._calc_pillar_shift(prev_pillar_dist, current_pillar_dist)
            if shift > self._config["pillar_shift_threshold"]:
                self._stats["pillar_shifts_detected"] += 1
                emitted = await self._emit_event(
                    "goal.pillar_shift",
                    {
                        "previous_distribution": prev_pillar_dist,
                        "current_distribution": current_pillar_dist,
                        "max_shift": round(shift, 3),
                        "active_goals_count": len(active_goals),
                        "timestamp": _now_iso(),
                    },
                    self._config["priority_pillar_shift"],
                )
                if emitted:
                    events_emitted += 1
                    events_detail.append(f"pillar_shift:{shift:.2f}")

        # Update snapshot
        self._last_snapshot = {
            "goal_ids": list(current_active_ids),
            "completed_ids": list(current_completed),
            "abandoned_ids": list(current_abandoned),
            "milestone_states": {
                g["id"]: [
                    m["index"] for m in g.get("milestones", []) if m.get("completed")
                ]
                for g in active_goals
            },
            "pillar_distribution": current_pillar_dist,
            "last_monitor_ts": _now_iso(),
        }

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Monitor complete: {events_emitted} events emitted, {len(active_goals)} active goals tracked",
            data={
                "events_emitted": events_emitted,
                "events_detail": events_detail,
                "active_goals": len(active_goals),
                "completed_goals": len(current_completed),
                "abandoned_goals": len(current_abandoned),
            },
        )

    async def _stall_check(self, params: Dict) -> SkillResult:
        """Detect goals that haven't progressed within threshold."""
        self._stats["stall_checks_run"] += 1
        self._stats["last_stall_check_time"] = _now_iso()

        goals_data = await self._get_goals_state()
        if goals_data is None:
            self._save_state()
            return SkillResult(
                success=True,
                message="Goals skill not available",
                data={"stalled_goals": 0},
            )

        active_goals = goals_data.get("goals", [])
        threshold_hours = self._config["stall_threshold_hours"]
        now = datetime.utcnow()
        stalled = []
        events_emitted = 0

        for goal in active_goals:
            # Find most recent activity timestamp
            last_activity = goal.get("created_at", "")
            for note in goal.get("progress_notes", []):
                ts = note.get("timestamp", "")
                if ts > last_activity:
                    last_activity = ts
            for ms in goal.get("milestones", []):
                ts = ms.get("completed_at", "") or ""
                if ts > last_activity:
                    last_activity = ts

            # Parse and check
            try:
                last_dt = datetime.fromisoformat(last_activity.replace("Z", ""))
                hours_idle = (now - last_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_idle = 0

            if hours_idle > threshold_hours:
                stalled.append({
                    "goal_id": goal["id"],
                    "title": goal.get("title", ""),
                    "pillar": goal.get("pillar", "other"),
                    "priority": goal.get("priority", "medium"),
                    "hours_idle": round(hours_idle, 1),
                    "deadline": goal.get("deadline"),
                })

        # Emit stall events
        if stalled and self._config["emit_on_stalled"]:
            for sg in stalled:
                self._stats["stalls_detected"] += 1
                emitted = await self._emit_event(
                    "goal.progress_stalled",
                    {
                        "goal_id": sg["goal_id"],
                        "title": sg["title"],
                        "pillar": sg["pillar"],
                        "priority": sg["priority"],
                        "hours_idle": sg["hours_idle"],
                        "deadline": sg["deadline"],
                        "threshold_hours": threshold_hours,
                        "timestamp": _now_iso(),
                    },
                    self._config["priority_stalled"],
                )
                if emitted:
                    events_emitted += 1

        self._save_state()

        return SkillResult(
            success=True,
            message=(
                f"Stall check: {len(stalled)} stalled goals "
                f"(>{threshold_hours}h idle), {events_emitted} events emitted"
            ),
            data={
                "stalled_goals": len(stalled),
                "stalled": stalled,
                "events_emitted": events_emitted,
                "threshold_hours": threshold_hours,
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update event emission configuration."""
        updated = []

        for key in (
            "emit_on_created",
            "emit_on_milestone_completed",
            "emit_on_completed",
            "emit_on_abandoned",
            "emit_on_stalled",
            "emit_on_pillar_shift",
        ):
            if key in params:
                self._config[key] = bool(params[key])
                updated.append(f"{key}={params[key]}")

        if "stall_threshold_hours" in params:
            val = float(params["stall_threshold_hours"])
            self._config["stall_threshold_hours"] = max(0.5, val)
            updated.append(f"stall_threshold_hours={self._config['stall_threshold_hours']}")

        if "pillar_shift_threshold" in params:
            val = float(params["pillar_shift_threshold"])
            self._config["pillar_shift_threshold"] = max(0.0, min(1.0, val))
            updated.append(f"pillar_shift_threshold={self._config['pillar_shift_threshold']}")

        for key in (
            "priority_created",
            "priority_milestone_completed",
            "priority_completed",
            "priority_abandoned",
            "priority_stalled",
            "priority_pillar_shift",
        ):
            if key in params:
                self._config[key] = str(params[key])
                updated.append(f"{key}={params[key]}")

        if "event_source" in params:
            self._config["event_source"] = str(params["event_source"])
            updated.append(f"event_source={params['event_source']}")

        if not updated:
            return SkillResult(
                success=False,
                message="No valid configuration parameters provided",
            )

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated: {', '.join(updated)}",
            data={"config": self._config},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """View bridge health and statistics."""
        lines = ["=== Goal Progress Event Bridge Status ==="]
        lines.append(f"Events emitted: {self._stats['events_emitted']}")
        lines.append(f"Events failed: {self._stats['events_failed']}")
        lines.append(f"Monitors run: {self._stats['monitors_run']}")
        lines.append(f"Stall checks run: {self._stats['stall_checks_run']}")
        lines.append(f"Goals created detected: {self._stats['goals_created_detected']}")
        lines.append(f"Milestones completed detected: {self._stats['milestones_completed_detected']}")
        lines.append(f"Goals completed detected: {self._stats['goals_completed_detected']}")
        lines.append(f"Goals abandoned detected: {self._stats['goals_abandoned_detected']}")
        lines.append(f"Stalls detected: {self._stats['stalls_detected']}")
        lines.append(f"Pillar shifts detected: {self._stats['pillar_shifts_detected']}")
        lines.append(f"Last monitor: {self._stats['last_monitor_time'] or 'never'}")
        lines.append(f"Last stall check: {self._stats['last_stall_check_time'] or 'never'}")
        lines.append(f"Tracked active goals: {len(self._last_snapshot.get('goal_ids', []))}")
        lines.append(f"Pillar distribution: {self._last_snapshot.get('pillar_distribution', {})}")

        return SkillResult(
            success=True,
            message="\n".join(lines),
            data={
                "stats": self._stats,
                "config": self._config,
                "snapshot": self._last_snapshot,
                "event_history_count": len(self._event_history),
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View recent emitted events."""
        limit = int(params.get("limit", 20))
        topic_filter = params.get("topic_filter", "")

        entries = self._event_history
        if topic_filter:
            entries = [e for e in entries if e.get("topic", "").startswith(topic_filter)]

        recent = entries[-limit:]

        if not recent:
            return SkillResult(
                success=True,
                message="No events in history",
                data={"events": [], "total": 0},
            )

        lines = [f"=== Event History (last {len(recent)}) ==="]
        for entry in reversed(recent):
            ts = entry.get("timestamp", "?")
            topic = entry.get("topic", "?")
            success = "ok" if entry.get("emitted") else "FAILED"
            lines.append(f"  [{ts}] {topic} ({success})")

        return SkillResult(
            success=True,
            message="\n".join(lines),
            data={"events": recent, "total": len(entries)},
        )

    async def _emit_test(self, params: Dict) -> SkillResult:
        """Emit a test event to verify EventBus integration."""
        emitted = await self._emit_event(
            "goal.test",
            {
                "message": "Test event from GoalProgressEventBridge",
                "timestamp": _now_iso(),
            },
            "normal",
        )

        return SkillResult(
            success=True,
            message=f"Test event {'emitted successfully' if emitted else 'failed (EventBus not available)'}",
            data={"emitted": emitted},
        )

    # --- Internal helpers ---

    def _calc_pillar_distribution(self, goals: List[Dict]) -> Dict[str, float]:
        """Calculate fraction of active goals per pillar."""
        if not goals:
            return {}
        counts: Dict[str, int] = {}
        for g in goals:
            pillar = g.get("pillar", "other")
            counts[pillar] = counts.get(pillar, 0) + 1
        total = len(goals)
        return {p: round(c / total, 3) for p, c in counts.items()}

    def _calc_pillar_shift(self, prev: Dict[str, float], current: Dict[str, float]) -> float:
        """Calculate max absolute shift in pillar distribution."""
        all_pillars = set(list(prev.keys()) + list(current.keys()))
        if not all_pillars:
            return 0.0
        max_shift = 0.0
        for p in all_pillars:
            shift = abs(current.get(p, 0.0) - prev.get(p, 0.0))
            if shift > max_shift:
                max_shift = shift
        return max_shift

    async def _get_goals_state(self) -> Optional[Dict]:
        """Get goal manager state via skill context."""
        try:
            if self.context:
                result = await self.context.call_skill(
                    "goals", "list", {"status": "all"}
                )
                if result and result.success:
                    return result.data
        except Exception:
            pass
        # Fallback: try reading goals file directly
        try:
            goals_file = DATA_DIR / "goals.json"
            if goals_file.exists():
                with open(goals_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    async def _emit_event(self, topic: str, data: Dict, priority: str = "normal") -> bool:
        """Emit an event via the skill registry's EventSkill."""
        event_record = {
            "topic": topic,
            "data": data,
            "priority": priority,
            "timestamp": _now_iso(),
            "emitted": False,
        }

        try:
            if hasattr(self, "_skill_registry") and self._skill_registry:
                result = await self._skill_registry.execute_skill(
                    "event", "publish",
                    {
                        "topic": topic,
                        "data": data,
                        "source": self._config.get("event_source", "goal_progress_event_bridge"),
                        "priority": priority,
                    },
                )
                emitted = result.success if result else False
            elif self.context:
                result = await self.context.call_skill(
                    "event", "publish",
                    {
                        "topic": topic,
                        "data": data,
                        "source": self._config.get("event_source", "goal_progress_event_bridge"),
                        "priority": priority,
                    },
                )
                emitted = result.success if result else False
            else:
                emitted = False
        except Exception:
            emitted = False

        event_record["emitted"] = emitted
        self._event_history.append(event_record)
        if len(self._event_history) > MAX_EVENT_HISTORY:
            self._event_history = self._event_history[-MAX_EVENT_HISTORY:]

        if emitted:
            self._stats["events_emitted"] += 1
        else:
            self._stats["events_failed"] += 1

        return emitted
