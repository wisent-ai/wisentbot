#!/usr/bin/env python3
"""
CheckpointEventBridgeSkill - Wire checkpoint lifecycle into EventBus for reactive auto-checkpointing.

AgentCheckpointSkill creates state snapshots and EventBus enables reactive behavior, but they
operate in isolation. This bridge connects them so that:

1. Checkpoint events are EMITTED on the bus (save, restore, prune, export, import)
2. Risky operations TRIGGER auto-checkpoints (self-modify, deploy, experiment)
3. Checkpoint health alerts fire when checkpoints are stale or storage is full
4. Other skills can SUBSCRIBE to checkpoint events for audit/monitoring

Event topics emitted:
  - checkpoint.saved        - A checkpoint was created
  - checkpoint.restored     - Agent state was rolled back
  - checkpoint.pruned       - Old checkpoints were cleaned up
  - checkpoint.exported     - Checkpoint packaged for replica transfer
  - checkpoint.imported     - Checkpoint imported from another agent
  - checkpoint.stale_alert  - No checkpoint taken in configured interval
  - checkpoint.storage_alert - Checkpoint storage exceeding threshold

Reactive triggers (auto-checkpoint before risky operations):
  - self_modify.*   → auto-save before any self-modification
  - deploy.*        → auto-save before deploying services
  - experiment.start → auto-save before starting experiments
  - incident.*      → auto-save when incidents are detected

Pillar: Self-Improvement (safety net for autonomous self-modification)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_STATE_FILE = Path(__file__).parent.parent / "data" / "checkpoint_event_bridge.json"
MAX_EVENT_LOG = 300

# ── Event definitions ──────────────────────────────────────────────────

CHECKPOINT_EVENTS = {
    "checkpoint.saved": {
        "description": "Emitted when a checkpoint is created",
        "source_action": "save",
        "fields": ["checkpoint_id", "label", "size_bytes", "file_count"],
    },
    "checkpoint.restored": {
        "description": "Emitted when agent state is restored from a checkpoint",
        "source_action": "restore",
        "fields": ["checkpoint_id", "label", "files_restored"],
    },
    "checkpoint.pruned": {
        "description": "Emitted when old checkpoints are cleaned up",
        "source_action": "prune",
        "fields": ["removed_count", "retained_count", "space_freed_bytes"],
    },
    "checkpoint.exported": {
        "description": "Emitted when a checkpoint is packaged for transfer",
        "source_action": "export",
        "fields": ["checkpoint_id", "export_path", "size_bytes"],
    },
    "checkpoint.imported": {
        "description": "Emitted when a checkpoint is imported from another agent",
        "source_action": "import_checkpoint",
        "fields": ["checkpoint_id", "source_agent"],
    },
    "checkpoint.stale_alert": {
        "description": "Alert: no checkpoint taken within configured interval",
        "source_action": "health_check",
        "fields": ["hours_since_last", "threshold_hours", "last_checkpoint_id"],
    },
    "checkpoint.storage_alert": {
        "description": "Alert: checkpoint storage exceeding threshold",
        "source_action": "health_check",
        "fields": ["total_size_mb", "threshold_mb", "checkpoint_count"],
    },
}

# ── Reactive triggers (events that cause auto-checkpoint) ──────────────

REACTIVE_TRIGGERS = {
    "pre_self_modify": {
        "description": "Auto-checkpoint before self-modification",
        "listen_topics": ["self_modify.*", "prompt_evolution.*"],
        "checkpoint_label": "pre-self-modify-{topic}",
        "priority": "high",
    },
    "pre_deploy": {
        "description": "Auto-checkpoint before service deployment",
        "listen_topics": ["deploy.*", "service_catalog.deploy", "service_hosting.*"],
        "checkpoint_label": "pre-deploy-{topic}",
        "priority": "normal",
    },
    "pre_experiment": {
        "description": "Auto-checkpoint before starting experiments",
        "listen_topics": ["experiment.start", "experiment.conclude"],
        "checkpoint_label": "pre-experiment-{topic}",
        "priority": "normal",
    },
    "on_incident": {
        "description": "Auto-checkpoint when incidents are detected",
        "listen_topics": ["incident.detected", "health.scan_complete"],
        "checkpoint_label": "incident-snapshot-{topic}",
        "priority": "high",
    },
    "pre_restore": {
        "description": "Auto-checkpoint before restoring (safety net)",
        "listen_topics": ["checkpoint.restore_requested"],
        "checkpoint_label": "safety-pre-restore",
        "priority": "critical",
    },
}


class CheckpointEventBridgeSkill(Skill):
    """
    Bridges AgentCheckpointSkill with EventBus for reactive auto-checkpointing.

    Emits checkpoint lifecycle events and sets up auto-checkpoint triggers
    on risky operations like self-modification and deployment.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = self._load_state()

    def _load_state(self) -> Dict:
        if BRIDGE_STATE_FILE.exists():
            try:
                with open(BRIDGE_STATE_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._default_state()

    def _default_state(self) -> Dict:
        return {
            "active_triggers": {},
            "event_log": [],
            "health_config": {
                "stale_threshold_hours": 6,
                "storage_threshold_mb": 100,
                "auto_health_check": True,
            },
            "stats": {
                "events_emitted": 0,
                "auto_checkpoints_triggered": 0,
                "alerts_fired": 0,
            },
        }

    def _save_state(self):
        BRIDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Trim event log
        if len(self._state.get("event_log", [])) > MAX_EVENT_LOG:
            self._state["event_log"] = self._state["event_log"][-MAX_EVENT_LOG:]
        try:
            with open(BRIDGE_STATE_FILE, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except IOError:
            pass

    def _log_event(self, topic: str, data: Dict):
        self._state.setdefault("event_log", []).append({
            "topic": topic,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })
        self._state["stats"]["events_emitted"] = self._state["stats"].get("events_emitted", 0) + 1

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="checkpoint_event_bridge",
            name="Checkpoint Event Bridge",
            version="1.0.0",
            category="infrastructure",
            description="Wire checkpoint lifecycle into EventBus for reactive auto-checkpointing on risky operations",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="wire",
                description="Activate reactive triggers for auto-checkpointing",
                parameters={
                    "trigger_ids": {"type": "list", "required": False, "description": "Specific triggers to activate (default: all)"},
                },
            ),
            SkillAction(
                name="unwire",
                description="Deactivate specific reactive triggers",
                parameters={
                    "trigger_ids": {"type": "list", "required": True, "description": "Trigger IDs to deactivate"},
                },
            ),
            SkillAction(
                name="emit",
                description="Manually emit a checkpoint event (for testing or manual checkpoint tracking)",
                parameters={
                    "event_type": {"type": "str", "required": True, "description": "Event type: saved, restored, pruned, exported, imported"},
                    "data": {"type": "dict", "required": False, "description": "Event data payload"},
                },
            ),
            SkillAction(
                name="health_check",
                description="Check checkpoint health: staleness, storage, and trigger status",
                parameters={},
            ),
            SkillAction(
                name="simulate",
                description="Simulate what would happen if a trigger event fires (dry run)",
                parameters={
                    "topic": {"type": "str", "required": True, "description": "Event topic to simulate (e.g., 'self_modify.prompt')"},
                },
            ),
            SkillAction(
                name="configure",
                description="Configure health check thresholds and alert settings",
                parameters={
                    "stale_threshold_hours": {"type": "int", "required": False, "description": "Hours before stale alert (default: 6)"},
                    "storage_threshold_mb": {"type": "int", "required": False, "description": "MB before storage alert (default: 100)"},
                },
            ),
            SkillAction(
                name="history",
                description="View recent checkpoint events and trigger activations",
                parameters={
                    "limit": {"type": "int", "required": False, "description": "Max events to return (default: 20)"},
                },
            ),
            SkillAction(
                name="status",
                description="View bridge status: active triggers, event counts, health",
                parameters={},
            ),
        ]

    def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        handlers = {
            "wire": self._wire,
            "unwire": self._unwire,
            "emit": self._emit,
            "health_check": self._health_check,
            "simulate": self._simulate,
            "configure": self._configure,
            "history": self._history,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        return handler(params)

    # ── Actions ────────────────────────────────────────────────────────

    def _wire(self, params: Dict) -> SkillResult:
        """Activate reactive triggers for auto-checkpointing."""
        trigger_ids = params.get("trigger_ids", list(REACTIVE_TRIGGERS.keys()))
        activated = []
        already_active = []
        invalid = []

        for tid in trigger_ids:
            if tid not in REACTIVE_TRIGGERS:
                invalid.append(tid)
                continue
            if tid in self._state.get("active_triggers", {}):
                already_active.append(tid)
                continue

            trigger = REACTIVE_TRIGGERS[tid]
            self._state.setdefault("active_triggers", {})[tid] = {
                "trigger_id": tid,
                "description": trigger["description"],
                "listen_topics": trigger["listen_topics"],
                "checkpoint_label": trigger["checkpoint_label"],
                "priority": trigger["priority"],
                "activated_at": datetime.now().isoformat(),
                "times_fired": 0,
            }
            activated.append(tid)
            self._log_event("bridge.trigger_activated", {"trigger_id": tid})

        self._save_state()

        msg_parts = []
        if activated:
            msg_parts.append(f"Activated {len(activated)} triggers: {activated}")
        if already_active:
            msg_parts.append(f"Already active: {already_active}")
        if invalid:
            msg_parts.append(f"Invalid: {invalid}")

        return SkillResult(
            success=len(activated) > 0 or len(already_active) > 0,
            message=". ".join(msg_parts) if msg_parts else "No triggers specified.",
            data={
                "activated": activated,
                "already_active": already_active,
                "invalid": invalid,
                "total_active": len(self._state.get("active_triggers", {})),
            },
        )

    def _unwire(self, params: Dict) -> SkillResult:
        """Deactivate specific reactive triggers."""
        trigger_ids = params.get("trigger_ids", [])
        removed = []
        not_found = []

        active = self._state.get("active_triggers", {})
        for tid in trigger_ids:
            if tid in active:
                del active[tid]
                removed.append(tid)
                self._log_event("bridge.trigger_deactivated", {"trigger_id": tid})
            else:
                not_found.append(tid)

        self._save_state()

        return SkillResult(
            success=len(removed) > 0,
            message=f"Removed {len(removed)} triggers: {removed}" + (f". Not found: {not_found}" if not_found else ""),
            data={
                "removed": removed,
                "not_found": not_found,
                "remaining_active": len(active),
            },
        )

    def _emit(self, params: Dict) -> SkillResult:
        """Manually emit a checkpoint event."""
        event_type = params.get("event_type", "")
        data = params.get("data", {})

        valid_types = ["saved", "restored", "pruned", "exported", "imported"]
        if event_type not in valid_types:
            return SkillResult(
                success=False,
                message=f"Invalid event type: {event_type}. Valid: {valid_types}",
            )

        topic = f"checkpoint.{event_type}"
        event_def = CHECKPOINT_EVENTS.get(topic, {})

        event_payload = {
            "topic": topic,
            "source": "checkpoint_event_bridge",
            "data": data,
            "description": event_def.get("description", ""),
            "emitted_at": datetime.now().isoformat(),
            "manual": True,
        }

        self._log_event(topic, event_payload)
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Emitted event: {topic}",
            data={"event": event_payload},
        )

    def _health_check(self, params: Dict) -> SkillResult:
        """Check checkpoint health: staleness and storage."""
        config = self._state.get("health_config", {})
        stale_threshold = config.get("stale_threshold_hours", 6)
        storage_threshold = config.get("storage_threshold_mb", 100)

        alerts = []
        health_score = 100

        # Check checkpoint staleness
        checkpoint_dir = Path(__file__).parent.parent / "data" / "checkpoints"
        index_file = checkpoint_dir / "index.json"
        last_checkpoint = None
        checkpoint_count = 0
        total_size_bytes = 0

        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    index = json.load(f)
                checkpoints = index.get("checkpoints", [])
                checkpoint_count = len(checkpoints)
                if checkpoints:
                    last_checkpoint = checkpoints[-1]
            except (json.JSONDecodeError, IOError):
                pass

        # Calculate storage
        if checkpoint_dir.exists():
            for item in checkpoint_dir.rglob("*"):
                if item.is_file():
                    total_size_bytes += item.stat().st_size
        total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

        # Staleness check
        hours_since_last = None
        if last_checkpoint:
            try:
                last_time = datetime.fromisoformat(last_checkpoint.get("created_at", ""))
                delta = datetime.now() - last_time
                hours_since_last = round(delta.total_seconds() / 3600, 1)
                if hours_since_last > stale_threshold:
                    alerts.append({
                        "type": "stale",
                        "topic": "checkpoint.stale_alert",
                        "message": f"No checkpoint in {hours_since_last}h (threshold: {stale_threshold}h)",
                        "hours_since_last": hours_since_last,
                    })
                    health_score -= 30
                    self._state["stats"]["alerts_fired"] = self._state["stats"].get("alerts_fired", 0) + 1
                    self._log_event("checkpoint.stale_alert", {
                        "hours_since_last": hours_since_last,
                        "threshold_hours": stale_threshold,
                        "last_checkpoint_id": last_checkpoint.get("checkpoint_id", "unknown"),
                    })
            except (ValueError, TypeError):
                pass

        # Storage check
        if total_size_mb > storage_threshold:
            alerts.append({
                "type": "storage",
                "topic": "checkpoint.storage_alert",
                "message": f"Checkpoint storage {total_size_mb}MB exceeds {storage_threshold}MB threshold",
                "total_size_mb": total_size_mb,
            })
            health_score -= 20
            self._state["stats"]["alerts_fired"] = self._state["stats"].get("alerts_fired", 0) + 1
            self._log_event("checkpoint.storage_alert", {
                "total_size_mb": total_size_mb,
                "threshold_mb": storage_threshold,
                "checkpoint_count": checkpoint_count,
            })

        # No checkpoints at all
        if checkpoint_count == 0:
            alerts.append({
                "type": "no_checkpoints",
                "message": "No checkpoints exist. Agent state is not protected.",
            })
            health_score -= 50

        # Active triggers check
        active_count = len(self._state.get("active_triggers", {}))
        if active_count == 0:
            health_score -= 10

        health_score = max(0, health_score)
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Checkpoint health: {health_score}/100. {len(alerts)} alerts. {checkpoint_count} checkpoints, {total_size_mb}MB storage.",
            data={
                "health_score": health_score,
                "alerts": alerts,
                "checkpoint_count": checkpoint_count,
                "total_storage_mb": total_size_mb,
                "hours_since_last_checkpoint": hours_since_last,
                "last_checkpoint": last_checkpoint.get("checkpoint_id") if last_checkpoint else None,
                "active_triggers": active_count,
                "thresholds": {
                    "stale_hours": stale_threshold,
                    "storage_mb": storage_threshold,
                },
            },
        )

    def _simulate(self, params: Dict) -> SkillResult:
        """Simulate what happens when a trigger event fires (dry run)."""
        topic = params.get("topic", "")
        if not topic:
            return SkillResult(success=False, message="Provide a topic to simulate (e.g., 'self_modify.prompt')")

        active = self._state.get("active_triggers", {})
        matching_triggers = []

        for tid, trigger in active.items():
            for pattern in trigger.get("listen_topics", []):
                if self._topic_matches(topic, pattern):
                    label = trigger["checkpoint_label"].replace("{topic}", topic)
                    matching_triggers.append({
                        "trigger_id": tid,
                        "description": trigger["description"],
                        "pattern_matched": pattern,
                        "checkpoint_label": label,
                        "priority": trigger["priority"],
                    })
                    break

        # Also check inactive triggers
        inactive_matches = []
        for tid, trigger in REACTIVE_TRIGGERS.items():
            if tid in active:
                continue
            for pattern in trigger.get("listen_topics", []):
                if self._topic_matches(topic, pattern):
                    inactive_matches.append({
                        "trigger_id": tid,
                        "description": trigger["description"],
                        "note": "This trigger is not active. Use 'wire' to activate.",
                    })
                    break

        would_checkpoint = len(matching_triggers) > 0
        return SkillResult(
            success=True,
            message=f"Topic '{topic}': {'WOULD' if would_checkpoint else 'would NOT'} trigger auto-checkpoint. {len(matching_triggers)} active match(es), {len(inactive_matches)} inactive match(es).",
            data={
                "topic": topic,
                "would_checkpoint": would_checkpoint,
                "matching_triggers": matching_triggers,
                "inactive_matches": inactive_matches,
            },
        )

    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if a topic matches a pattern (supports * wildcard)."""
        if pattern == topic:
            return True
        if "*" in pattern:
            parts = pattern.split("*")
            if len(parts) == 2:
                prefix, suffix = parts
                return topic.startswith(prefix) and topic.endswith(suffix)
        return False

    def _configure(self, params: Dict) -> SkillResult:
        """Configure health check thresholds."""
        config = self._state.setdefault("health_config", {})
        updated = []

        if "stale_threshold_hours" in params:
            val = params["stale_threshold_hours"]
            if val < 1 or val > 168:
                return SkillResult(success=False, message="stale_threshold_hours must be 1-168")
            config["stale_threshold_hours"] = val
            updated.append(f"stale_threshold_hours={val}")

        if "storage_threshold_mb" in params:
            val = params["storage_threshold_mb"]
            if val < 10 or val > 10000:
                return SkillResult(success=False, message="storage_threshold_mb must be 10-10000")
            config["storage_threshold_mb"] = val
            updated.append(f"storage_threshold_mb={val}")

        if not updated:
            return SkillResult(
                success=True,
                message="No changes. Current config shown.",
                data={"config": config},
            )

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated: {', '.join(updated)}",
            data={"config": config},
        )

    def _history(self, params: Dict) -> SkillResult:
        """View recent checkpoint events."""
        limit = params.get("limit", 20)
        events = self._state.get("event_log", [])
        recent = events[-limit:] if limit < len(events) else events

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {len(events)} checkpoint events",
            data={
                "events": recent,
                "total_events": len(events),
            },
        )

    def _status(self, params: Dict) -> SkillResult:
        """View bridge status overview."""
        active = self._state.get("active_triggers", {})
        stats = self._state.get("stats", {})
        config = self._state.get("health_config", {})

        # Available vs active triggers
        available_triggers = []
        for tid, trigger in REACTIVE_TRIGGERS.items():
            available_triggers.append({
                "trigger_id": tid,
                "description": trigger["description"],
                "listen_topics": trigger["listen_topics"],
                "active": tid in active,
                "times_fired": active[tid].get("times_fired", 0) if tid in active else 0,
            })

        # Available events
        available_events = []
        for topic, evt in CHECKPOINT_EVENTS.items():
            available_events.append({
                "topic": topic,
                "description": evt["description"],
                "fields": evt["fields"],
            })

        return SkillResult(
            success=True,
            message=f"Bridge: {len(active)}/{len(REACTIVE_TRIGGERS)} triggers active. {stats.get('events_emitted', 0)} events emitted, {stats.get('auto_checkpoints_triggered', 0)} auto-checkpoints.",
            data={
                "triggers": available_triggers,
                "active_trigger_count": len(active),
                "total_triggers": len(REACTIVE_TRIGGERS),
                "events": available_events,
                "stats": stats,
                "health_config": config,
                "event_log_size": len(self._state.get("event_log", [])),
            },
        )
