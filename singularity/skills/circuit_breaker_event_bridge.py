#!/usr/bin/env python3
"""
CircuitBreakerEventBridgeSkill - Emit EventBus events on circuit breaker state changes.

The CircuitBreakerSkill (PR #232) protects against runaway failures, and the
AutonomousLoop (PR #233) wires every skill execution through it. But circuit
state changes happen silently - no other skill knows when a circuit opens or
closes. This bridge fixes that.

When wired, it:
1. **Polls** the circuit breaker for state changes since last check
2. **Emits** structured events to EventBus on every state transition:
   - circuit_breaker.opened → skill X failed too much, circuit opened
   - circuit_breaker.half_open → cooldown elapsed, testing recovery
   - circuit_breaker.closed → skill X recovered
   - circuit_breaker.forced_open → manual block
   - circuit_breaker.forced_closed → manual override
   - circuit_breaker.budget_critical → budget protection activated
3. **Tracks** transition history for reporting
4. **Configures** which transitions emit events and at what priority

This enables reactive automation:
- AlertIncidentBridge can auto-create incidents when circuits open
- ServiceMonitor can degrade gracefully when dependencies break
- AgentReflection can auto-reflect on why a skill keeps failing
- StrategySkill can reprioritize when key capabilities go offline

Pillar: Self-Improvement (reactive safety automation)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

BRIDGE_DATA_FILE = Path(__file__).parent.parent / "data" / "cb_event_bridge.json"
MAX_HISTORY = 500
MAX_TRANSITION_LOG = 300


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


class CircuitBreakerEventBridgeSkill(Skill):
    """
    Bridge between CircuitBreakerSkill and EventBus.

    Polls circuit breaker state, detects transitions, and emits
    structured events so downstream skills can react automatically.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._load_state()

    def _load_state(self):
        """Load persisted bridge state."""
        BRIDGE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if BRIDGE_DATA_FILE.exists():
            try:
                with open(BRIDGE_DATA_FILE) as f:
                    data = json.load(f)
                self._known_states = data.get("known_states", {})
                self._transition_log = data.get("transition_log", [])
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._known_states: Dict[str, str] = {}  # skill_id -> last known state
        self._transition_log: List[Dict] = []
        self._config = self._default_config()
        self._stats = self._default_stats()

    def _default_config(self) -> Dict:
        return {
            "enabled": True,
            "poll_on_sync": True,
            # Which transitions to emit events for (all by default)
            "emit_on_opened": True,
            "emit_on_half_open": True,
            "emit_on_closed": True,
            "emit_on_forced": True,
            "emit_on_budget_critical": True,
            # Priority mapping for different transitions
            "priority_opened": "high",
            "priority_half_open": "normal",
            "priority_closed": "normal",
            "priority_forced_open": "critical",
            "priority_forced_closed": "high",
            "priority_budget_critical": "critical",
            # Source tag for emitted events
            "event_source": "circuit_breaker_event_bridge",
        }

    def _default_stats(self) -> Dict:
        return {
            "total_polls": 0,
            "total_transitions_detected": 0,
            "total_events_emitted": 0,
            "transitions_by_type": {},
            "last_poll_at": None,
            "last_event_at": None,
        }

    def _save_state(self):
        """Persist bridge state."""
        BRIDGE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "known_states": self._known_states,
            "transition_log": self._transition_log[-MAX_TRANSITION_LOG:],
            "config": self._config,
            "stats": self._stats,
        }
        with open(BRIDGE_DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="circuit_breaker_event_bridge",
            name="Circuit Breaker Event Bridge",
            version="1.0.0",
            category="infrastructure",
            description=(
                "Emits EventBus events when circuit breaker states change. "
                "Enables reactive automation: auto-create incidents on circuit open, "
                "auto-reflect on failures, auto-degrade services."
            ),
            actions=[
                SkillAction(
                    name="sync",
                    description=(
                        "Poll circuit breaker dashboard and emit events for any "
                        "state changes since last sync. Run this periodically or "
                        "after skill executions."
                    ),
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="configure",
                    description="Configure which transitions emit events and their priorities.",
                    parameters={
                        "emit_on_opened": {
                            "type": "boolean",
                            "required": False,
                            "description": "Emit events when circuits open",
                        },
                        "emit_on_half_open": {
                            "type": "boolean",
                            "required": False,
                            "description": "Emit events when circuits enter half-open",
                        },
                        "emit_on_closed": {
                            "type": "boolean",
                            "required": False,
                            "description": "Emit events when circuits close (recover)",
                        },
                        "emit_on_forced": {
                            "type": "boolean",
                            "required": False,
                            "description": "Emit events for manual force-open/close",
                        },
                        "emit_on_budget_critical": {
                            "type": "boolean",
                            "required": False,
                            "description": "Emit events when budget protection activates",
                        },
                        "enabled": {
                            "type": "boolean",
                            "required": False,
                            "description": "Enable/disable the bridge entirely",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="status",
                    description="View bridge health, known circuit states, and emission stats.",
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="history",
                    description="View recent state transitions detected by the bridge.",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max entries to return (default 20)",
                        },
                        "skill_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by skill ID",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="reset",
                    description="Clear known states and transition history. Next sync will re-baseline.",
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="emit_test",
                    description=(
                        "Emit a test event to verify EventBus integration. "
                        "Publishes circuit_breaker.test with a sample payload."
                    ),
                    parameters={},
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if action == "sync":
            return await self._sync(params)
        elif action == "configure":
            return self._configure(params)
        elif action == "status":
            return self._status()
        elif action == "history":
            return self._history(params)
        elif action == "reset":
            return self._reset()
        elif action == "emit_test":
            return await self._emit_test()
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    async def _sync(self, params: Dict) -> SkillResult:
        """Poll circuit breaker and emit events for state changes."""
        if not self._config.get("enabled", True):
            return SkillResult(
                success=True,
                message="Bridge disabled, skipping sync",
                data={"enabled": False},
            )

        # Get circuit breaker dashboard from the skill registry
        cb_data = await self._get_circuit_dashboard()
        if cb_data is None:
            return SkillResult(
                success=False,
                message="Could not read circuit breaker dashboard. Is CircuitBreakerSkill registered?",
            )

        self._stats["total_polls"] = self._stats.get("total_polls", 0) + 1
        self._stats["last_poll_at"] = _now_iso()

        transitions = []
        events_emitted = 0

        circuits = cb_data.get("circuits", {})
        for skill_id, circuit_info in circuits.items():
            current_state = circuit_info.get("state", "closed")
            previous_state = self._known_states.get(skill_id)

            if previous_state is None:
                # First time seeing this circuit - baseline it
                self._known_states[skill_id] = current_state
                continue

            if current_state != previous_state:
                # State changed!
                transition = {
                    "skill_id": skill_id,
                    "from_state": previous_state,
                    "to_state": current_state,
                    "detected_at": _now_iso(),
                    "circuit_info": {
                        "failure_rate": circuit_info.get("failure_rate", 0),
                        "total_requests": circuit_info.get("total_requests", 0),
                        "opened_count": circuit_info.get("opened_count", 0),
                    },
                }
                transitions.append(transition)
                self._known_states[skill_id] = current_state

                # Log transition
                self._transition_log.append(transition)
                if len(self._transition_log) > MAX_TRANSITION_LOG:
                    self._transition_log = self._transition_log[-MAX_TRANSITION_LOG:]

                # Track stats
                self._stats["total_transitions_detected"] = (
                    self._stats.get("total_transitions_detected", 0) + 1
                )
                transition_type = f"{previous_state}->{current_state}"
                by_type = self._stats.get("transitions_by_type", {})
                by_type[transition_type] = by_type.get(transition_type, 0) + 1
                self._stats["transitions_by_type"] = by_type

                # Emit event if configured
                emitted = await self._emit_transition_event(
                    skill_id, previous_state, current_state, circuit_info
                )
                if emitted:
                    events_emitted += 1

        # Check for budget critical condition
        budget_critical = cb_data.get("budget_critical", False)
        if budget_critical and self._config.get("emit_on_budget_critical", True):
            await self._emit_event(
                "circuit_breaker.budget_critical",
                {
                    "message": "Budget protection activated - non-essential skills blocked",
                    "circuits_affected": [
                        sid for sid, info in circuits.items()
                        if info.get("state") == "open"
                    ],
                },
                self._config.get("priority_budget_critical", "critical"),
            )
            events_emitted += 1

        self._stats["total_events_emitted"] = (
            self._stats.get("total_events_emitted", 0) + events_emitted
        )
        if events_emitted > 0:
            self._stats["last_event_at"] = _now_iso()

        self._save_state()

        return SkillResult(
            success=True,
            message=(
                f"Synced {len(circuits)} circuits: "
                f"{len(transitions)} transition(s), {events_emitted} event(s) emitted"
            ),
            data={
                "circuits_checked": len(circuits),
                "transitions": transitions,
                "events_emitted": events_emitted,
            },
        )

    async def _emit_transition_event(
        self,
        skill_id: str,
        from_state: str,
        to_state: str,
        circuit_info: Dict,
    ) -> bool:
        """Emit an EventBus event for a circuit state transition."""
        # Determine event topic based on target state
        topic_map = {
            "open": ("circuit_breaker.opened", "emit_on_opened", "priority_opened"),
            "half_open": ("circuit_breaker.half_open", "emit_on_half_open", "priority_half_open"),
            "closed": ("circuit_breaker.closed", "emit_on_closed", "priority_closed"),
            "forced_open": ("circuit_breaker.forced_open", "emit_on_forced", "priority_forced_open"),
            "forced_closed": ("circuit_breaker.forced_closed", "emit_on_forced", "priority_forced_closed"),
        }

        entry = topic_map.get(to_state)
        if entry is None:
            return False

        topic, config_key, priority_key = entry
        if not self._config.get(config_key, True):
            return False

        priority = self._config.get(priority_key, "normal")

        data = {
            "skill_id": skill_id,
            "from_state": from_state,
            "to_state": to_state,
            "failure_rate": circuit_info.get("failure_rate", 0),
            "total_requests": circuit_info.get("total_requests", 0),
            "opened_count": circuit_info.get("opened_count", 0),
            "timestamp": _now_iso(),
        }

        return await self._emit_event(topic, data, priority)

    async def _emit_event(self, topic: str, data: Dict, priority: str = "normal") -> bool:
        """Emit an event via the skill registry's EventSkill."""
        try:
            if hasattr(self, "_skill_registry") and self._skill_registry:
                result = await self._skill_registry.execute_skill(
                    "event", "publish",
                    {
                        "topic": topic,
                        "data": data,
                        "source": self._config.get("event_source", "circuit_breaker_event_bridge"),
                        "priority": priority,
                    },
                )
                return result.success if result else False
            return False
        except Exception:
            return False

    async def _get_circuit_dashboard(self) -> Optional[Dict]:
        """Get the circuit breaker dashboard data."""
        try:
            if hasattr(self, "_skill_registry") and self._skill_registry:
                result = await self._skill_registry.execute_skill(
                    "circuit_breaker", "dashboard", {}
                )
                if result and result.success:
                    return result.data
            return None
        except Exception:
            return None

    def _configure(self, params: Dict) -> SkillResult:
        """Update bridge configuration."""
        updated = []
        for key in [
            "enabled", "emit_on_opened", "emit_on_half_open", "emit_on_closed",
            "emit_on_forced", "emit_on_budget_critical",
            "priority_opened", "priority_half_open", "priority_closed",
            "priority_forced_open", "priority_forced_closed",
            "priority_budget_critical", "event_source",
        ]:
            if key in params:
                self._config[key] = params[key]
                updated.append(key)

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} config key(s): {', '.join(updated)}" if updated else "No changes",
            data={"config": self._config, "updated": updated},
        )

    def _status(self) -> SkillResult:
        """Return bridge health and status."""
        open_circuits = [
            sid for sid, state in self._known_states.items()
            if state in ("open", "forced_open")
        ]
        half_open_circuits = [
            sid for sid, state in self._known_states.items()
            if state == "half_open"
        ]

        return SkillResult(
            success=True,
            message=(
                f"Bridge {'enabled' if self._config.get('enabled') else 'disabled'} | "
                f"{len(self._known_states)} circuits tracked | "
                f"{len(open_circuits)} open | "
                f"{self._stats.get('total_events_emitted', 0)} events emitted"
            ),
            data={
                "enabled": self._config.get("enabled", True),
                "circuits_tracked": len(self._known_states),
                "known_states": self._known_states,
                "open_circuits": open_circuits,
                "half_open_circuits": half_open_circuits,
                "stats": self._stats,
                "config": self._config,
                "recent_transitions": self._transition_log[-5:],
            },
        )

    def _history(self, params: Dict) -> SkillResult:
        """Return recent transition history."""
        limit = params.get("limit", 20)
        skill_filter = params.get("skill_id")

        entries = self._transition_log
        if skill_filter:
            entries = [e for e in entries if e.get("skill_id") == skill_filter]

        entries = entries[-limit:]

        return SkillResult(
            success=True,
            message=f"Found {len(entries)} transition(s)",
            data={"transitions": entries, "count": len(entries)},
        )

    def _reset(self) -> SkillResult:
        """Reset all known states and history."""
        circuits_cleared = len(self._known_states)
        transitions_cleared = len(self._transition_log)
        self._known_states = {}
        self._transition_log = []
        self._stats = self._default_stats()
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Reset: cleared {circuits_cleared} known states, {transitions_cleared} transitions",
            data={
                "circuits_cleared": circuits_cleared,
                "transitions_cleared": transitions_cleared,
            },
        )

    async def _emit_test(self) -> SkillResult:
        """Emit a test event to verify EventBus integration."""
        emitted = await self._emit_event(
            "circuit_breaker.test",
            {
                "message": "Test event from CircuitBreakerEventBridge",
                "timestamp": _now_iso(),
            },
            "normal",
        )

        return SkillResult(
            success=True,
            message=f"Test event {'emitted' if emitted else 'failed (EventBus not available)'}",
            data={"emitted": emitted},
        )
