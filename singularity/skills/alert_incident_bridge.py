#!/usr/bin/env python3
"""
AlertIncidentBridgeSkill - Bridges ObservabilitySkill alerts to IncidentResponseSkill.

When ObservabilitySkill's check_alerts detects threshold breaches, this skill
automatically creates incidents via IncidentResponseSkill, maps alert severity
to incident severity, and auto-resolves incidents when alerts clear.

This closes a critical gap: observability alerts currently fire into the void.
With this bridge, alert.fired -> auto-creates incident, alert.resolved -> auto-resolves.

Bridges: ObservabilitySkill <-> IncidentResponseSkill via SkillEventBridge patterns.

Pillar: Self-Improvement (autonomous incident detection from metrics)
Also: Revenue (keeps production services running by auto-responding to degradation)

Actions:
- monitor: Run a full alert check cycle and bridge any fired/resolved alerts to incidents
- configure: Set alert-to-severity mappings, auto-resolve behavior, and thresholds
- status: View bridge status, active alert-incident mappings, and statistics
- mappings: View/edit alert name to incident severity mappings
- history: View recent alert-to-incident bridge events
- link: Manually link an existing alert to an existing incident
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_STATE_FILE = Path(__file__).parent.parent / "data" / "alert_incident_bridge.json"
MAX_HISTORY = 200

# Default mapping from alert severity to incident severity
DEFAULT_SEVERITY_MAP = {
    "critical": "sev1",
    "high": "sev2",
    "warning": "sev3",
    "info": "sev4",
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class AlertIncidentBridgeSkill(Skill):
    """
    Bridges ObservabilitySkill alerts to IncidentResponseSkill for autonomous
    incident management triggered by metric threshold breaches.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="alert_incident_bridge",
            name="Alert-Incident Bridge",
            version="1.0.0",
            category="meta",
            description="Auto-creates and resolves incidents from observability alerts",
            actions=[
                SkillAction(
                    name="monitor",
                    description="Run alert check cycle, auto-create/resolve incidents from fired/resolved alerts",
                    parameters={
                        "dry_run": {"type": "boolean", "required": False, "description": "Preview actions without executing (default: False)"},
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Configure bridge behavior: severity mapping, auto-resolve, dedup window",
                    parameters={
                        "severity_map": {"type": "object", "required": False, "description": "Map alert severity -> incident severity"},
                        "auto_resolve": {"type": "boolean", "required": False, "description": "Auto-resolve incidents when alerts clear (default: True)"},
                        "dedup_window_minutes": {"type": "number", "required": False, "description": "Don't create duplicate incidents within this window (default: 30)"},
                        "auto_triage": {"type": "boolean", "required": False, "description": "Auto-triage created incidents (default: True)"},
                        "default_assignee": {"type": "string", "required": False, "description": "Default agent to assign incidents to"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="View bridge status, active alert-incident links, and statistics",
                    parameters={},
                ),
                SkillAction(
                    name="mappings",
                    description="View current alert-to-incident severity mappings and overrides",
                    parameters={
                        "alert_name": {"type": "string", "required": False, "description": "Set/view mapping for a specific alert name"},
                        "incident_severity": {"type": "string", "required": False, "description": "Override severity for this alert name (sev1-sev4)"},
                    },
                ),
                SkillAction(
                    name="history",
                    description="View recent alert-to-incident bridge events",
                    parameters={
                        "limit": {"type": "number", "required": False, "description": "Max events to return (default: 20)"},
                        "event_type": {"type": "string", "required": False, "description": "Filter: 'created', 'resolved', 'deduped', 'all' (default: all)"},
                    },
                ),
                SkillAction(
                    name="link",
                    description="Manually link an alert to an existing incident",
                    parameters={
                        "alert_name": {"type": "string", "required": True, "description": "Name of the alert"},
                        "incident_id": {"type": "string", "required": True, "description": "ID of the incident to link to"},
                    },
                ),
            ],
            required_credentials=[],
        )

    # ── Persistence ───────────────────────────────────────────────

    def _default_state(self) -> Dict:
        return {
            "config": {
                "severity_map": dict(DEFAULT_SEVERITY_MAP),
                "alert_overrides": {},  # Per-alert-name severity overrides
                "auto_resolve": True,
                "auto_triage": True,
                "dedup_window_minutes": 30,
                "default_assignee": None,
            },
            "active_links": {},  # alert_name -> {incident_id, created_at, alert_severity, ...}
            "history": [],
            "stats": {
                "incidents_created": 0,
                "incidents_resolved": 0,
                "incidents_deduped": 0,
                "monitor_cycles": 0,
                "last_monitor": None,
            },
            "metadata": {
                "created_at": _now_iso(),
                "version": "1.0.0",
            },
        }

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        BRIDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if BRIDGE_STATE_FILE.exists():
            try:
                with open(BRIDGE_STATE_FILE, "r") as f:
                    self._store = json.load(f)
                    return self._store
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        self._store = self._default_state()
        self._save(self._store)
        return self._store

    def _save(self, data: Dict):
        self._store = data
        if len(data.get("history", [])) > MAX_HISTORY:
            data["history"] = data["history"][-MAX_HISTORY:]
        BRIDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BRIDGE_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Execute Dispatch ──────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "monitor": self._monitor,
            "configure": self._configure,
            "status": self._status,
            "mappings": self._mappings,
            "history": self._history,
            "link": self._link,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Helpers ───────────────────────────────────────────────────

    def _map_severity(self, store: Dict, alert_name: str, alert_severity: str) -> str:
        """Map an alert severity to an incident severity, with per-alert overrides."""
        overrides = store["config"].get("alert_overrides", {})
        if alert_name in overrides:
            return overrides[alert_name]
        sev_map = store["config"]["severity_map"]
        return sev_map.get(alert_severity, "sev3")

    def _is_dedup(self, store: Dict, alert_name: str) -> bool:
        """Check if an alert already has an active incident within the dedup window."""
        link = store["active_links"].get(alert_name)
        if not link:
            return False
        # Check if within dedup window
        created_at = link.get("created_at", "")
        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            now = datetime.utcnow()
            # Make created_dt naive for comparison
            if created_dt.tzinfo is not None:
                created_dt = created_dt.replace(tzinfo=None)
            dedup_mins = store["config"]["dedup_window_minutes"]
            from datetime import timedelta
            if (now - created_dt).total_seconds() < dedup_mins * 60:
                return True
        except (ValueError, TypeError):
            pass
        return False

    async def _call_skill(self, skill_id: str, action: str, params: Dict) -> Optional[Dict]:
        """Call another skill through SkillContext if available."""
        if not self.context:
            return None
        try:
            result = await self.context.invoke_skill(skill_id, action, params)
            if hasattr(result, "data"):
                return result.data if result.success else None
            return result
        except Exception:
            return None

    # ── Action: monitor ──────────────────────────────────────────

    async def _monitor(self, params: Dict) -> SkillResult:
        """Run a full alert check cycle, bridge fired/resolved alerts to incidents."""
        dry_run = params.get("dry_run", False)
        store = self._load()

        # Step 1: Check alerts via ObservabilitySkill
        alert_result = await self._call_skill("observability", "check_alerts", {})

        if alert_result is None:
            return SkillResult(
                success=False,
                message="Cannot reach ObservabilitySkill - ensure it is loaded",
                data={"error": "observability skill unavailable"},
            )

        fired_alerts = alert_result.get("fired", [])
        resolved_alerts = alert_result.get("resolved", [])

        created_incidents = []
        resolved_incidents = []
        deduped = []
        errors = []

        # Step 2: Create incidents for fired alerts
        for alert in fired_alerts:
            alert_name = alert.get("name", "unknown")
            alert_severity = alert.get("severity", "warning")
            current_value = alert.get("current_value", 0)
            metric = alert.get("metric", "unknown")
            condition = alert.get("condition", "")

            # Check dedup
            if self._is_dedup(store, alert_name):
                deduped.append({"alert": alert_name, "reason": "active incident exists within dedup window"})
                store["stats"]["incidents_deduped"] += 1
                store["history"].append({
                    "event": "deduped",
                    "alert_name": alert_name,
                    "timestamp": _now_iso(),
                })
                continue

            incident_severity = self._map_severity(store, alert_name, alert_severity)
            incident_title = f"Alert fired: {alert_name} ({metric} {condition}, value={current_value:.2f})"
            incident_desc = (
                f"Automatically created by AlertIncidentBridge.\n"
                f"Alert: {alert_name}\n"
                f"Metric: {metric}\n"
                f"Condition: {condition}\n"
                f"Current value: {current_value}\n"
                f"Alert severity: {alert_severity}\n"
                f"Mapped incident severity: {incident_severity}"
            )

            if dry_run:
                created_incidents.append({
                    "alert": alert_name,
                    "would_create": {
                        "title": incident_title,
                        "severity": incident_severity,
                    },
                    "dry_run": True,
                })
                continue

            # Create incident via IncidentResponseSkill
            detect_params = {
                "title": incident_title,
                "description": incident_desc,
                "source": "alert_incident_bridge",
                "severity": incident_severity,
            }
            detect_result = await self._call_skill("incident_response", "detect", detect_params)

            if detect_result and detect_result.get("incident_id"):
                incident_id = detect_result["incident_id"]

                # Link alert to incident
                store["active_links"][alert_name] = {
                    "incident_id": incident_id,
                    "created_at": _now_iso(),
                    "alert_severity": alert_severity,
                    "incident_severity": incident_severity,
                    "metric": metric,
                    "condition": condition,
                    "value_at_creation": current_value,
                }

                created_incidents.append({
                    "alert": alert_name,
                    "incident_id": incident_id,
                    "severity": incident_severity,
                })

                store["stats"]["incidents_created"] += 1
                store["history"].append({
                    "event": "created",
                    "alert_name": alert_name,
                    "incident_id": incident_id,
                    "severity": incident_severity,
                    "timestamp": _now_iso(),
                })

                # Auto-triage if configured
                if store["config"]["auto_triage"]:
                    triage_params = {
                        "incident_id": incident_id,
                        "severity": incident_severity,
                    }
                    if store["config"]["default_assignee"]:
                        triage_params["assignee"] = store["config"]["default_assignee"]
                    await self._call_skill("incident_response", "triage", triage_params)
            else:
                errors.append({"alert": alert_name, "error": "Failed to create incident"})

        # Step 3: Auto-resolve incidents for resolved alerts
        if store["config"]["auto_resolve"]:
            for alert in resolved_alerts:
                alert_name = alert.get("name", "unknown")
                link = store["active_links"].get(alert_name)

                if not link:
                    continue

                if dry_run:
                    resolved_incidents.append({
                        "alert": alert_name,
                        "would_resolve": link["incident_id"],
                        "dry_run": True,
                    })
                    continue

                # Resolve incident
                resolve_params = {
                    "incident_id": link["incident_id"],
                    "resolution": f"Alert '{alert_name}' has cleared. Current value: {alert.get('current_value', 'N/A')}",
                    "resolution_type": "auto_resolved",
                }
                await self._call_skill("incident_response", "resolve", resolve_params)

                resolved_incidents.append({
                    "alert": alert_name,
                    "incident_id": link["incident_id"],
                })

                store["stats"]["incidents_resolved"] += 1
                store["history"].append({
                    "event": "resolved",
                    "alert_name": alert_name,
                    "incident_id": link["incident_id"],
                    "timestamp": _now_iso(),
                })

                # Remove active link
                del store["active_links"][alert_name]

        # Update stats
        store["stats"]["monitor_cycles"] += 1
        store["stats"]["last_monitor"] = _now_iso()

        if not dry_run:
            self._save(store)

        summary_parts = []
        if created_incidents:
            summary_parts.append(f"{len(created_incidents)} incidents created")
        if resolved_incidents:
            summary_parts.append(f"{len(resolved_incidents)} incidents resolved")
        if deduped:
            summary_parts.append(f"{len(deduped)} deduped")
        if errors:
            summary_parts.append(f"{len(errors)} errors")
        if not summary_parts:
            summary_parts.append("no alert changes detected")

        prefix = "[DRY RUN] " if dry_run else ""

        return SkillResult(
            success=True,
            message=f"{prefix}Monitor cycle complete: {', '.join(summary_parts)}",
            data={
                "fired_alerts": len(fired_alerts),
                "resolved_alerts": len(resolved_alerts),
                "incidents_created": created_incidents,
                "incidents_resolved": resolved_incidents,
                "deduped": deduped,
                "errors": errors,
                "active_links": len(store["active_links"]),
                "dry_run": dry_run,
            },
        )

    # ── Action: configure ────────────────────────────────────────

    async def _configure(self, params: Dict) -> SkillResult:
        """Configure bridge behavior."""
        store = self._load()
        config = store["config"]
        changes = []

        valid_severities = {"sev1", "sev2", "sev3", "sev4"}

        if "severity_map" in params:
            sev_map = params["severity_map"]
            if isinstance(sev_map, dict):
                for v in sev_map.values():
                    if v not in valid_severities:
                        return SkillResult(
                            success=False,
                            message=f"Invalid severity '{v}'. Must be one of: {valid_severities}",
                        )
                config["severity_map"].update(sev_map)
                changes.append(f"severity_map updated: {sev_map}")

        if "auto_resolve" in params:
            config["auto_resolve"] = bool(params["auto_resolve"])
            changes.append(f"auto_resolve = {config['auto_resolve']}")

        if "auto_triage" in params:
            config["auto_triage"] = bool(params["auto_triage"])
            changes.append(f"auto_triage = {config['auto_triage']}")

        if "dedup_window_minutes" in params:
            val = int(params["dedup_window_minutes"])
            if val < 1 or val > 1440:
                return SkillResult(success=False, message="dedup_window_minutes must be 1-1440")
            config["dedup_window_minutes"] = val
            changes.append(f"dedup_window_minutes = {val}")

        if "default_assignee" in params:
            config["default_assignee"] = params["default_assignee"] or None
            changes.append(f"default_assignee = {config['default_assignee']}")

        if not changes:
            return SkillResult(
                success=True,
                message="No changes specified",
                data={"config": config},
            )

        self._save(store)
        return SkillResult(
            success=True,
            message=f"Configuration updated: {'; '.join(changes)}",
            data={"config": config},
        )

    # ── Action: status ───────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        """View bridge status."""
        store = self._load()

        return SkillResult(
            success=True,
            message=f"Alert-Incident Bridge: {len(store['active_links'])} active links, "
                    f"{store['stats']['incidents_created']} created, "
                    f"{store['stats']['incidents_resolved']} resolved",
            data={
                "active_links": store["active_links"],
                "stats": store["stats"],
                "config": store["config"],
                "recent_history": store["history"][-5:] if store["history"] else [],
            },
        )

    # ── Action: mappings ─────────────────────────────────────────

    async def _mappings(self, params: Dict) -> SkillResult:
        """View or edit alert-to-severity mappings."""
        store = self._load()

        alert_name = params.get("alert_name", "").strip()
        incident_severity = params.get("incident_severity", "").strip()

        # If setting a specific override
        if alert_name and incident_severity:
            valid_severities = {"sev1", "sev2", "sev3", "sev4"}
            if incident_severity not in valid_severities:
                return SkillResult(
                    success=False,
                    message=f"Invalid severity '{incident_severity}'. Must be one of: {valid_severities}",
                )
            store["config"]["alert_overrides"][alert_name] = incident_severity
            self._save(store)
            return SkillResult(
                success=True,
                message=f"Override set: alert '{alert_name}' -> {incident_severity}",
                data={
                    "alert_name": alert_name,
                    "incident_severity": incident_severity,
                    "all_overrides": store["config"]["alert_overrides"],
                    "default_map": store["config"]["severity_map"],
                },
            )

        # If querying a specific alert
        if alert_name:
            mapped = self._map_severity(store, alert_name, "warning")
            override = store["config"]["alert_overrides"].get(alert_name)
            return SkillResult(
                success=True,
                message=f"Alert '{alert_name}' maps to {mapped}" +
                        (f" (override)" if override else " (default)"),
                data={
                    "alert_name": alert_name,
                    "mapped_severity": mapped,
                    "is_override": override is not None,
                    "default_map": store["config"]["severity_map"],
                    "all_overrides": store["config"]["alert_overrides"],
                },
            )

        # Show all mappings
        return SkillResult(
            success=True,
            message=f"Severity mappings: {len(store['config']['severity_map'])} defaults, "
                    f"{len(store['config']['alert_overrides'])} overrides",
            data={
                "default_map": store["config"]["severity_map"],
                "overrides": store["config"]["alert_overrides"],
            },
        )

    # ── Action: history ──────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        """View recent bridge events."""
        store = self._load()
        limit = min(int(params.get("limit", 20)), MAX_HISTORY)
        event_type = params.get("event_type", "all")

        history = store.get("history", [])

        if event_type and event_type != "all":
            history = [e for e in history if e.get("event") == event_type]

        recent = history[-limit:] if len(history) > limit else history

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} bridge events" +
                    (f" (filtered: {event_type})" if event_type != "all" else ""),
            data={
                "events": recent,
                "total": len(history),
                "filter": event_type,
            },
        )

    # ── Action: link ─────────────────────────────────────────────

    async def _link(self, params: Dict) -> SkillResult:
        """Manually link an alert to an existing incident."""
        alert_name = params.get("alert_name", "").strip()
        incident_id = params.get("incident_id", "").strip()

        if not alert_name or not incident_id:
            return SkillResult(
                success=False,
                message="Both alert_name and incident_id are required",
            )

        store = self._load()

        store["active_links"][alert_name] = {
            "incident_id": incident_id,
            "created_at": _now_iso(),
            "alert_severity": "manual",
            "incident_severity": "manual",
            "metric": "manual_link",
            "condition": "manual",
            "value_at_creation": 0,
            "manual": True,
        }

        store["history"].append({
            "event": "manual_link",
            "alert_name": alert_name,
            "incident_id": incident_id,
            "timestamp": _now_iso(),
        })

        self._save(store)

        return SkillResult(
            success=True,
            message=f"Linked alert '{alert_name}' to incident '{incident_id}'",
            data={
                "alert_name": alert_name,
                "incident_id": incident_id,
                "active_links": len(store["active_links"]),
            },
        )
