#!/usr/bin/env python3
"""
IncidentResponseSkill - Autonomous incident detection, triage, response, and postmortem.

When production services encounter problems (error spikes, latency degradation,
service outages, SLA breaches), the agent needs a structured way to:

1. DETECT   - Identify incidents from metrics, errors, or external alerts
2. TRIAGE   - Classify severity (SEV1-SEV4) and impact
3. RESPOND  - Execute response playbooks (restart, rollback, scale, notify)
4. ESCALATE - Route to appropriate handler based on incident type
5. RESOLVE  - Mark incident resolved with resolution details
6. POSTMORTEM - Generate structured postmortem with root cause and learnings

Bridges Self-Improvement (learn from failures) and Revenue (keep services running).

Integrates with:
- SelfHealingSkill: can trigger healing as part of incident response
- AgentReputationSkill: record incident handling quality
- EventBus: emit incident lifecycle events
- SchedulerSkill: schedule follow-up checks after resolution
- WorkflowSkill: execute multi-step response playbooks

Actions:
- detect: Report/detect a new incident
- triage: Classify severity and assign priority
- respond: Execute a response action from the playbook
- escalate: Route incident to another agent or external system
- resolve: Close an incident with resolution details
- postmortem: Generate structured postmortem analysis
- playbook: Manage response playbooks (create/list/get)
- status: View active incidents and metrics
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

INCIDENT_FILE = Path(__file__).parent.parent / "data" / "incidents.json"

# Severity levels
SEV1 = "sev1"  # Critical: total service outage, data loss
SEV2 = "sev2"  # High: major feature broken, significant degradation
SEV3 = "sev3"  # Medium: minor feature broken, workaround exists
SEV4 = "sev4"  # Low: cosmetic issue, minor inconvenience

SEVERITY_LEVELS = {
    SEV1: {"label": "Critical", "response_minutes": 5, "update_minutes": 15},
    SEV2: {"label": "High", "response_minutes": 15, "update_minutes": 30},
    SEV3: {"label": "Medium", "response_minutes": 60, "update_minutes": 120},
    SEV4: {"label": "Low", "response_minutes": 240, "update_minutes": 480},
}

# Incident statuses
STATUS_DETECTED = "detected"
STATUS_TRIAGED = "triaged"
STATUS_RESPONDING = "responding"
STATUS_ESCALATED = "escalated"
STATUS_RESOLVED = "resolved"
STATUS_POSTMORTEM = "postmortem_complete"

# Response action types
ACTION_RESTART = "restart"
ACTION_ROLLBACK = "rollback"
ACTION_SCALE_UP = "scale_up"
ACTION_FAILOVER = "failover"
ACTION_NOTIFY = "notify"
ACTION_BLOCK_TRAFFIC = "block_traffic"
ACTION_CUSTOM = "custom"

MAX_INCIDENTS = 500
MAX_PLAYBOOKS = 50
MAX_TIMELINE_ENTRIES = 100


class IncidentResponseSkill(Skill):
    """
    Autonomous incident detection, triage, response, and postmortem.

    Manages the full lifecycle of production incidents with structured
    severity classification, response playbooks, escalation chains,
    and postmortem generation for continuous improvement.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="incident_response",
            name="Incident Response",
            version="1.0.0",
            category="operations",
            description="Autonomous incident detection, triage, response playbooks, and postmortem generation",
            actions=[
                SkillAction(
                    name="detect",
                    description="Report/detect a new incident",
                    parameters={
                        "title": {"type": "string", "required": True, "description": "Short incident title"},
                        "description": {"type": "string", "required": True, "description": "What is happening"},
                        "source": {"type": "string", "required": True, "description": "Detection source: monitoring, alert, manual, skill_error"},
                        "service": {"type": "string", "required": False, "description": "Affected service name"},
                        "severity": {"type": "string", "required": False, "description": "Initial severity: sev1, sev2, sev3, sev4"},
                        "metadata": {"type": "object", "required": False, "description": "Additional context (error codes, metrics, etc.)"},
                    },
                ),
                SkillAction(
                    name="triage",
                    description="Classify severity and assign priority to an incident",
                    parameters={
                        "incident_id": {"type": "string", "required": True, "description": "Incident ID to triage"},
                        "severity": {"type": "string", "required": True, "description": "Severity: sev1, sev2, sev3, sev4"},
                        "impact": {"type": "string", "required": False, "description": "Impact description (users affected, services impacted)"},
                        "assignee": {"type": "string", "required": False, "description": "Agent ID to assign"},
                        "tags": {"type": "list", "required": False, "description": "Classification tags"},
                    },
                ),
                SkillAction(
                    name="respond",
                    description="Execute a response action on an incident",
                    parameters={
                        "incident_id": {"type": "string", "required": True, "description": "Incident ID"},
                        "action_type": {"type": "string", "required": True, "description": "Action: restart, rollback, scale_up, failover, notify, block_traffic, custom"},
                        "details": {"type": "string", "required": False, "description": "Action details or custom command"},
                        "playbook_id": {"type": "string", "required": False, "description": "Execute a specific playbook instead"},
                    },
                ),
                SkillAction(
                    name="escalate",
                    description="Escalate incident to another handler",
                    parameters={
                        "incident_id": {"type": "string", "required": True, "description": "Incident ID to escalate"},
                        "target": {"type": "string", "required": True, "description": "Agent ID or external target"},
                        "reason": {"type": "string", "required": True, "description": "Why this is being escalated"},
                        "new_severity": {"type": "string", "required": False, "description": "Optionally upgrade severity"},
                    },
                ),
                SkillAction(
                    name="resolve",
                    description="Mark an incident as resolved",
                    parameters={
                        "incident_id": {"type": "string", "required": True, "description": "Incident ID to resolve"},
                        "resolution": {"type": "string", "required": True, "description": "What fixed it"},
                        "root_cause": {"type": "string", "required": False, "description": "Root cause if known"},
                        "follow_up_actions": {"type": "list", "required": False, "description": "Actions to prevent recurrence"},
                    },
                ),
                SkillAction(
                    name="postmortem",
                    description="Generate a structured postmortem for a resolved incident",
                    parameters={
                        "incident_id": {"type": "string", "required": True, "description": "Incident ID"},
                    },
                ),
                SkillAction(
                    name="playbook",
                    description="Manage response playbooks",
                    parameters={
                        "operation": {"type": "string", "required": True, "description": "Operation: create, list, get, delete"},
                        "playbook_id": {"type": "string", "required": False, "description": "Playbook ID (for get/delete)"},
                        "name": {"type": "string", "required": False, "description": "Playbook name (for create)"},
                        "trigger_conditions": {"type": "object", "required": False, "description": "When to auto-trigger: {severity, service, tags}"},
                        "steps": {"type": "list", "required": False, "description": "Ordered response steps [{action_type, details, timeout_seconds}]"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="View active incidents and incident metrics",
                    parameters={
                        "incident_id": {"type": "string", "required": False, "description": "Specific incident ID (omit for overview)"},
                        "filter_severity": {"type": "string", "required": False, "description": "Filter by severity"},
                        "filter_status": {"type": "string", "required": False, "description": "Filter by status"},
                        "include_metrics": {"type": "boolean", "required": False, "description": "Include aggregate metrics"},
                    },
                ),
            ],
            required_credentials=[],
        )

    # ── Persistence ───────────────────────────────────────────────

    def _ensure_data(self):
        INCIDENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not INCIDENT_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "incidents": {},
            "playbooks": {},
            "stats": {
                "total_detected": 0,
                "total_resolved": 0,
                "total_escalated": 0,
                "by_severity": {SEV1: 0, SEV2: 0, SEV3: 0, SEV4: 0},
                "mttr_seconds": [],  # mean time to resolve samples
            },
        }

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        self._ensure_data()
        with open(INCIDENT_FILE, "r") as f:
            self._store = json.load(f)
        return self._store

    def _save(self, data: Dict):
        self._store = data
        INCIDENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INCIDENT_FILE, "w") as f:
            json.dump(data, f, indent=2)

    # ── Core Logic ────────────────────────────────────────────────

    def _add_timeline_entry(self, incident: Dict, event_type: str, details: str):
        """Add a timestamped entry to the incident timeline."""
        timeline = incident.setdefault("timeline", [])
        timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "details": details,
        })
        # Keep timeline bounded
        if len(timeline) > MAX_TIMELINE_ENTRIES:
            incident["timeline"] = timeline[-MAX_TIMELINE_ENTRIES:]

    def _find_matching_playbook(self, store: Dict, incident: Dict) -> Optional[Dict]:
        """Find a playbook whose trigger conditions match this incident."""
        for pb_id, pb in store.get("playbooks", {}).items():
            conditions = pb.get("trigger_conditions", {})
            if not conditions:
                continue
            match = True
            if "severity" in conditions:
                sev_list = conditions["severity"]
                if isinstance(sev_list, str):
                    sev_list = [sev_list]
                if incident.get("severity") not in sev_list:
                    match = False
            if "service" in conditions:
                svc_list = conditions["service"]
                if isinstance(svc_list, str):
                    svc_list = [svc_list]
                if incident.get("service") not in svc_list:
                    match = False
            if "tags" in conditions:
                required_tags = set(conditions["tags"])
                incident_tags = set(incident.get("tags", []))
                if not required_tags.issubset(incident_tags):
                    match = False
            if match:
                return pb
        return None

    def _compute_mttr(self, incident: Dict) -> Optional[float]:
        """Compute time-to-resolve in seconds."""
        detected = incident.get("detected_at")
        resolved = incident.get("resolved_at")
        if detected and resolved:
            d = datetime.fromisoformat(detected)
            r = datetime.fromisoformat(resolved)
            return (r - d).total_seconds()
        return None

    # ── Execute Dispatch ──────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "detect": self._detect,
            "triage": self._triage,
            "respond": self._respond,
            "escalate": self._escalate,
            "resolve": self._resolve,
            "postmortem": self._postmortem,
            "playbook": self._playbook,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Action: detect ────────────────────────────────────────────

    async def _detect(self, params: Dict) -> SkillResult:
        title = params.get("title")
        description = params.get("description")
        source = params.get("source", "manual")
        if not title or not description:
            return SkillResult(success=False, message="title and description required")

        store = self._load()
        incidents = store["incidents"]

        # Enforce limit
        if len(incidents) >= MAX_INCIDENTS:
            # Remove oldest resolved incidents
            resolved = sorted(
                [(k, v) for k, v in incidents.items() if v["status"] == STATUS_RESOLVED],
                key=lambda x: x[1].get("detected_at", ""),
            )
            if resolved:
                del incidents[resolved[0][0]]

        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        severity = params.get("severity", SEV3)
        if severity not in SEVERITY_LEVELS:
            severity = SEV3

        now = datetime.utcnow().isoformat()
        incident = {
            "id": incident_id,
            "title": title,
            "description": description,
            "source": source,
            "service": params.get("service", "unknown"),
            "severity": severity,
            "status": STATUS_DETECTED,
            "detected_at": now,
            "updated_at": now,
            "resolved_at": None,
            "assignee": None,
            "tags": [],
            "impact": None,
            "resolution": None,
            "root_cause": None,
            "follow_up_actions": [],
            "response_actions": [],
            "escalation_history": [],
            "timeline": [],
            "metadata": params.get("metadata", {}),
            "postmortem": None,
        }

        self._add_timeline_entry(incident, "detected", f"Incident detected via {source}: {title}")

        incidents[incident_id] = incident
        store["stats"]["total_detected"] += 1
        store["stats"]["by_severity"][severity] = store["stats"]["by_severity"].get(severity, 0) + 1

        # Auto-match playbook
        playbook = self._find_matching_playbook(store, incident)
        playbook_info = None
        if playbook:
            playbook_info = {"id": playbook["id"], "name": playbook["name"]}
            self._add_timeline_entry(incident, "playbook_matched", f"Auto-matched playbook: {playbook['name']}")

        self._save(store)

        sev_info = SEVERITY_LEVELS[severity]
        return SkillResult(
            success=True,
            message=f"Incident {incident_id} detected [{sev_info['label']}]: {title}. "
                    f"Response expected within {sev_info['response_minutes']} minutes.",
            data={
                "incident_id": incident_id,
                "severity": severity,
                "severity_label": sev_info["label"],
                "status": STATUS_DETECTED,
                "matched_playbook": playbook_info,
            },
        )

    # ── Action: triage ────────────────────────────────────────────

    async def _triage(self, params: Dict) -> SkillResult:
        incident_id = params.get("incident_id")
        severity = params.get("severity")
        if not incident_id or not severity:
            return SkillResult(success=False, message="incident_id and severity required")
        if severity not in SEVERITY_LEVELS:
            return SkillResult(success=False, message=f"Invalid severity: {severity}. Use sev1-sev4.")

        store = self._load()
        incident = store["incidents"].get(incident_id)
        if not incident:
            return SkillResult(success=False, message=f"Incident {incident_id} not found")

        old_severity = incident["severity"]
        incident["severity"] = severity
        incident["status"] = STATUS_TRIAGED
        incident["updated_at"] = datetime.utcnow().isoformat()

        if params.get("impact"):
            incident["impact"] = params["impact"]
        if params.get("assignee"):
            incident["assignee"] = params["assignee"]
        if params.get("tags"):
            incident["tags"] = list(set(incident.get("tags", []) + params["tags"]))

        sev_change = f" (changed from {old_severity})" if old_severity != severity else ""
        self._add_timeline_entry(
            incident, "triaged",
            f"Triaged as {severity}{sev_change}. Assignee: {incident.get('assignee', 'unassigned')}"
        )

        # Update severity stats
        if old_severity != severity:
            store["stats"]["by_severity"][old_severity] = max(
                0, store["stats"]["by_severity"].get(old_severity, 1) - 1
            )
            store["stats"]["by_severity"][severity] = store["stats"]["by_severity"].get(severity, 0) + 1

        self._save(store)

        sev_info = SEVERITY_LEVELS[severity]
        return SkillResult(
            success=True,
            message=f"Incident {incident_id} triaged as {sev_info['label']} ({severity}). "
                    f"Impact: {incident.get('impact', 'not specified')}.",
            data={
                "incident_id": incident_id,
                "severity": severity,
                "severity_label": sev_info["label"],
                "status": STATUS_TRIAGED,
                "assignee": incident.get("assignee"),
                "tags": incident.get("tags", []),
            },
        )

    # ── Action: respond ───────────────────────────────────────────

    async def _respond(self, params: Dict) -> SkillResult:
        incident_id = params.get("incident_id")
        if not incident_id:
            return SkillResult(success=False, message="incident_id required")

        store = self._load()
        incident = store["incidents"].get(incident_id)
        if not incident:
            return SkillResult(success=False, message=f"Incident {incident_id} not found")

        # Execute a playbook if specified
        playbook_id = params.get("playbook_id")
        if playbook_id:
            playbook = store["playbooks"].get(playbook_id)
            if not playbook:
                return SkillResult(success=False, message=f"Playbook {playbook_id} not found")
            return await self._execute_playbook(store, incident, playbook)

        # Single action response
        action_type = params.get("action_type")
        if not action_type:
            return SkillResult(success=False, message="action_type or playbook_id required")

        valid_actions = [ACTION_RESTART, ACTION_ROLLBACK, ACTION_SCALE_UP,
                         ACTION_FAILOVER, ACTION_NOTIFY, ACTION_BLOCK_TRAFFIC, ACTION_CUSTOM]
        if action_type not in valid_actions:
            return SkillResult(
                success=False,
                message=f"Invalid action_type: {action_type}. Use one of: {', '.join(valid_actions)}"
            )

        details = params.get("details", "")
        now = datetime.utcnow().isoformat()

        response_entry = {
            "action_type": action_type,
            "details": details,
            "executed_at": now,
            "status": "executed",
        }
        incident["response_actions"].append(response_entry)
        incident["status"] = STATUS_RESPONDING
        incident["updated_at"] = now

        self._add_timeline_entry(
            incident, "response_action",
            f"Executed {action_type}: {details or 'no details'}"
        )

        self._save(store)

        return SkillResult(
            success=True,
            message=f"Response action '{action_type}' executed on incident {incident_id}.",
            data={
                "incident_id": incident_id,
                "action_type": action_type,
                "details": details,
                "total_actions": len(incident["response_actions"]),
            },
        )

    async def _execute_playbook(self, store: Dict, incident: Dict, playbook: Dict) -> SkillResult:
        """Execute all steps in a playbook sequentially."""
        steps = playbook.get("steps", [])
        results = []
        for i, step in enumerate(steps):
            action_type = step.get("action_type", ACTION_CUSTOM)
            details = step.get("details", "")
            now = datetime.utcnow().isoformat()

            response_entry = {
                "action_type": action_type,
                "details": details,
                "executed_at": now,
                "status": "executed",
                "playbook_step": i + 1,
                "playbook_id": playbook["id"],
            }
            incident["response_actions"].append(response_entry)
            results.append(response_entry)

            self._add_timeline_entry(
                incident, "playbook_step",
                f"Playbook '{playbook['name']}' step {i+1}: {action_type} - {details}"
            )

        incident["status"] = STATUS_RESPONDING
        incident["updated_at"] = datetime.utcnow().isoformat()
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Playbook '{playbook['name']}' executed ({len(steps)} steps) on incident {incident['id']}.",
            data={
                "incident_id": incident["id"],
                "playbook_id": playbook["id"],
                "playbook_name": playbook["name"],
                "steps_executed": len(steps),
                "results": results,
            },
        )

    # ── Action: escalate ──────────────────────────────────────────

    async def _escalate(self, params: Dict) -> SkillResult:
        incident_id = params.get("incident_id")
        target = params.get("target")
        reason = params.get("reason")
        if not incident_id or not target or not reason:
            return SkillResult(success=False, message="incident_id, target, and reason required")

        store = self._load()
        incident = store["incidents"].get(incident_id)
        if not incident:
            return SkillResult(success=False, message=f"Incident {incident_id} not found")

        now = datetime.utcnow().isoformat()

        # Optionally upgrade severity
        new_severity = params.get("new_severity")
        if new_severity and new_severity in SEVERITY_LEVELS:
            old_sev = incident["severity"]
            incident["severity"] = new_severity
            if old_sev != new_severity:
                store["stats"]["by_severity"][old_sev] = max(
                    0, store["stats"]["by_severity"].get(old_sev, 1) - 1
                )
                store["stats"]["by_severity"][new_severity] = store["stats"]["by_severity"].get(new_severity, 0) + 1

        escalation = {
            "target": target,
            "reason": reason,
            "escalated_at": now,
            "previous_assignee": incident.get("assignee"),
            "severity_at_escalation": incident["severity"],
        }
        incident["escalation_history"].append(escalation)
        incident["assignee"] = target
        incident["status"] = STATUS_ESCALATED
        incident["updated_at"] = now

        store["stats"]["total_escalated"] += 1

        self._add_timeline_entry(
            incident, "escalated",
            f"Escalated to {target}: {reason}"
        )

        self._save(store)

        return SkillResult(
            success=True,
            message=f"Incident {incident_id} escalated to {target}. Reason: {reason}",
            data={
                "incident_id": incident_id,
                "target": target,
                "reason": reason,
                "severity": incident["severity"],
                "escalation_count": len(incident["escalation_history"]),
            },
        )

    # ── Action: resolve ───────────────────────────────────────────

    async def _resolve(self, params: Dict) -> SkillResult:
        incident_id = params.get("incident_id")
        resolution = params.get("resolution")
        if not incident_id or not resolution:
            return SkillResult(success=False, message="incident_id and resolution required")

        store = self._load()
        incident = store["incidents"].get(incident_id)
        if not incident:
            return SkillResult(success=False, message=f"Incident {incident_id} not found")

        if incident["status"] == STATUS_RESOLVED:
            return SkillResult(success=False, message=f"Incident {incident_id} is already resolved")

        now = datetime.utcnow().isoformat()
        incident["status"] = STATUS_RESOLVED
        incident["resolution"] = resolution
        incident["root_cause"] = params.get("root_cause")
        incident["follow_up_actions"] = params.get("follow_up_actions", [])
        incident["resolved_at"] = now
        incident["updated_at"] = now

        self._add_timeline_entry(
            incident, "resolved",
            f"Resolved: {resolution}"
        )

        store["stats"]["total_resolved"] += 1

        # Track MTTR
        mttr = self._compute_mttr(incident)
        if mttr is not None:
            mttr_list = store["stats"]["mttr_seconds"]
            mttr_list.append(mttr)
            # Keep last 100 MTTR samples
            if len(mttr_list) > 100:
                store["stats"]["mttr_seconds"] = mttr_list[-100:]

        self._save(store)

        mttr_str = f" MTTR: {mttr:.0f}s" if mttr is not None else ""
        return SkillResult(
            success=True,
            message=f"Incident {incident_id} resolved.{mttr_str} Resolution: {resolution}",
            data={
                "incident_id": incident_id,
                "status": STATUS_RESOLVED,
                "resolution": resolution,
                "root_cause": incident.get("root_cause"),
                "mttr_seconds": mttr,
                "follow_up_actions": incident.get("follow_up_actions", []),
            },
        )

    # ── Action: postmortem ────────────────────────────────────────

    async def _postmortem(self, params: Dict) -> SkillResult:
        incident_id = params.get("incident_id")
        if not incident_id:
            return SkillResult(success=False, message="incident_id required")

        store = self._load()
        incident = store["incidents"].get(incident_id)
        if not incident:
            return SkillResult(success=False, message=f"Incident {incident_id} not found")

        if incident["status"] not in (STATUS_RESOLVED, STATUS_POSTMORTEM):
            return SkillResult(
                success=False,
                message=f"Incident must be resolved before postmortem. Current status: {incident['status']}"
            )

        # Build postmortem
        mttr = self._compute_mttr(incident)
        timeline = incident.get("timeline", [])
        response_actions = incident.get("response_actions", [])
        escalations = incident.get("escalation_history", [])

        # Calculate response time (time from detection to first response action)
        response_time = None
        if timeline and response_actions:
            detected_time = datetime.fromisoformat(incident["detected_at"])
            first_action_time = None
            for entry in timeline:
                if entry["event"] in ("response_action", "playbook_step"):
                    first_action_time = datetime.fromisoformat(entry["timestamp"])
                    break
            if first_action_time:
                response_time = (first_action_time - detected_time).total_seconds()

        # Determine if SLA was met
        sev_info = SEVERITY_LEVELS.get(incident["severity"], SEVERITY_LEVELS[SEV3])
        sla_met = None
        if response_time is not None:
            sla_met = response_time <= (sev_info["response_minutes"] * 60)

        postmortem = {
            "incident_id": incident_id,
            "title": incident["title"],
            "severity": incident["severity"],
            "service": incident["service"],
            "detected_at": incident["detected_at"],
            "resolved_at": incident["resolved_at"],
            "mttr_seconds": mttr,
            "response_time_seconds": response_time,
            "sla_met": sla_met,
            "root_cause": incident.get("root_cause", "Not determined"),
            "resolution": incident.get("resolution", "Not documented"),
            "impact": incident.get("impact", "Not assessed"),
            "timeline_summary": [
                {"time": e["timestamp"], "event": e["event"], "details": e["details"]}
                for e in timeline
            ],
            "response_actions_taken": len(response_actions),
            "escalation_count": len(escalations),
            "follow_up_actions": incident.get("follow_up_actions", []),
            "lessons_learned": [],
            "generated_at": datetime.utcnow().isoformat(),
        }

        # Auto-generate lessons learned
        lessons = []
        if mttr and mttr > 3600:
            lessons.append("Resolution took over 1 hour. Consider adding automated remediation.")
        if len(escalations) > 2:
            lessons.append(f"Incident was escalated {len(escalations)} times. Review escalation criteria.")
        if not incident.get("root_cause"):
            lessons.append("Root cause was not identified. Schedule a deeper investigation.")
        if sla_met is False:
            lessons.append(f"SLA was breached. Response time: {response_time:.0f}s, target: {sev_info['response_minutes'] * 60}s.")
        if not response_actions:
            lessons.append("No response actions were taken. Consider creating a playbook for this type of incident.")
        postmortem["lessons_learned"] = lessons

        incident["postmortem"] = postmortem
        incident["status"] = STATUS_POSTMORTEM
        incident["updated_at"] = datetime.utcnow().isoformat()

        self._save(store)

        return SkillResult(
            success=True,
            message=f"Postmortem generated for {incident_id}. "
                    f"MTTR: {mttr:.0f}s. SLA met: {sla_met}. "
                    f"{len(lessons)} lessons learned.",
            data=postmortem,
        )

    # ── Action: playbook ──────────────────────────────────────────

    async def _playbook(self, params: Dict) -> SkillResult:
        operation = params.get("operation")
        if not operation:
            return SkillResult(success=False, message="operation required (create, list, get, delete)")

        store = self._load()
        playbooks = store.setdefault("playbooks", {})

        if operation == "create":
            name = params.get("name")
            if not name:
                return SkillResult(success=False, message="name required for create")
            if len(playbooks) >= MAX_PLAYBOOKS:
                return SkillResult(success=False, message=f"Maximum {MAX_PLAYBOOKS} playbooks reached")

            pb_id = f"PB-{uuid.uuid4().hex[:8].upper()}"
            steps = params.get("steps", [])
            trigger_conditions = params.get("trigger_conditions", {})

            playbook = {
                "id": pb_id,
                "name": name,
                "trigger_conditions": trigger_conditions,
                "steps": steps,
                "created_at": datetime.utcnow().isoformat(),
                "times_executed": 0,
            }
            playbooks[pb_id] = playbook
            self._save(store)

            return SkillResult(
                success=True,
                message=f"Playbook '{name}' created with {len(steps)} steps.",
                data={"playbook_id": pb_id, "name": name, "steps": len(steps)},
            )

        elif operation == "list":
            result = []
            for pb_id, pb in playbooks.items():
                result.append({
                    "id": pb_id,
                    "name": pb["name"],
                    "steps": len(pb.get("steps", [])),
                    "trigger_conditions": pb.get("trigger_conditions", {}),
                    "times_executed": pb.get("times_executed", 0),
                })
            return SkillResult(
                success=True,
                message=f"{len(result)} playbooks available.",
                data={"playbooks": result},
            )

        elif operation == "get":
            pb_id = params.get("playbook_id")
            if not pb_id:
                return SkillResult(success=False, message="playbook_id required for get")
            pb = playbooks.get(pb_id)
            if not pb:
                return SkillResult(success=False, message=f"Playbook {pb_id} not found")
            return SkillResult(success=True, message=f"Playbook: {pb['name']}", data=pb)

        elif operation == "delete":
            pb_id = params.get("playbook_id")
            if not pb_id:
                return SkillResult(success=False, message="playbook_id required for delete")
            if pb_id not in playbooks:
                return SkillResult(success=False, message=f"Playbook {pb_id} not found")
            name = playbooks[pb_id]["name"]
            del playbooks[pb_id]
            self._save(store)
            return SkillResult(success=True, message=f"Playbook '{name}' deleted.", data={"deleted": pb_id})

        else:
            return SkillResult(success=False, message=f"Unknown operation: {operation}. Use create, list, get, delete.")

    # ── Action: status ────────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        store = self._load()

        # Single incident detail
        incident_id = params.get("incident_id")
        if incident_id:
            incident = store["incidents"].get(incident_id)
            if not incident:
                return SkillResult(success=False, message=f"Incident {incident_id} not found")
            return SkillResult(
                success=True,
                message=f"Incident {incident_id}: [{incident['severity']}] {incident['title']} - {incident['status']}",
                data=incident,
            )

        # Overview
        incidents = store["incidents"]
        filter_severity = params.get("filter_severity")
        filter_status = params.get("filter_status")

        filtered = list(incidents.values())
        if filter_severity:
            filtered = [i for i in filtered if i["severity"] == filter_severity]
        if filter_status:
            filtered = [i for i in filtered if i["status"] == filter_status]

        active = [i for i in filtered if i["status"] not in (STATUS_RESOLVED, STATUS_POSTMORTEM)]
        resolved = [i for i in filtered if i["status"] in (STATUS_RESOLVED, STATUS_POSTMORTEM)]

        result = {
            "active_incidents": [
                {
                    "id": i["id"],
                    "title": i["title"],
                    "severity": i["severity"],
                    "status": i["status"],
                    "service": i["service"],
                    "detected_at": i["detected_at"],
                    "assignee": i.get("assignee"),
                }
                for i in sorted(active, key=lambda x: list(SEVERITY_LEVELS.keys()).index(x["severity"]))
            ],
            "active_count": len(active),
            "resolved_count": len(resolved),
            "total_count": len(filtered),
        }

        if params.get("include_metrics"):
            stats = store["stats"]
            mttr_list = stats.get("mttr_seconds", [])
            avg_mttr = sum(mttr_list) / len(mttr_list) if mttr_list else None
            result["metrics"] = {
                "total_detected": stats["total_detected"],
                "total_resolved": stats["total_resolved"],
                "total_escalated": stats["total_escalated"],
                "by_severity": stats["by_severity"],
                "avg_mttr_seconds": avg_mttr,
                "resolution_rate": (
                    stats["total_resolved"] / stats["total_detected"]
                    if stats["total_detected"] > 0 else None
                ),
            }

        return SkillResult(
            success=True,
            message=f"{len(active)} active incident(s), {len(resolved)} resolved.",
            data=result,
        )
