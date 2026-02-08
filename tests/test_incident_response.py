"""Tests for IncidentResponseSkill."""

import pytest
import json
from singularity.skills.incident_response import (
    IncidentResponseSkill, INCIDENT_FILE, SEV1, SEV2, SEV3, SEV4,
    STATUS_DETECTED, STATUS_TRIAGED, STATUS_RESPONDING, STATUS_ESCALATED,
    STATUS_RESOLVED, STATUS_POSTMORTEM,
)


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data path."""
    s = IncidentResponseSkill()
    test_file = tmp_path / "incidents.json"
    import singularity.skills.incident_response as mod
    mod.INCIDENT_FILE = test_file
    s._store = None
    s._ensure_data()
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "incident_response"
    assert len(m.actions) == 8
    action_names = [a.name for a in m.actions]
    assert "detect" in action_names
    assert "postmortem" in action_names


@pytest.mark.asyncio
async def test_detect_incident(skill):
    r = await skill.execute("detect", {
        "title": "API 500 errors spike",
        "description": "Error rate jumped from 0.1% to 15%",
        "source": "monitoring",
        "service": "api-gateway",
        "severity": "sev2",
    })
    assert r.success
    assert "INC-" in r.data["incident_id"]
    assert r.data["severity"] == "sev2"
    assert r.data["status"] == STATUS_DETECTED


@pytest.mark.asyncio
async def test_detect_defaults_severity(skill):
    r = await skill.execute("detect", {
        "title": "Minor issue",
        "description": "Something happened",
        "source": "manual",
    })
    assert r.success
    assert r.data["severity"] == SEV3


@pytest.mark.asyncio
async def test_triage_incident(skill):
    r1 = await skill.execute("detect", {
        "title": "DB connection timeout",
        "description": "Pool exhausted",
        "source": "alert",
        "severity": "sev3",
    })
    inc_id = r1.data["incident_id"]

    r2 = await skill.execute("triage", {
        "incident_id": inc_id,
        "severity": "sev1",
        "impact": "All users affected",
        "assignee": "agent-ops-1",
        "tags": ["database", "connection"],
    })
    assert r2.success
    assert r2.data["severity"] == SEV1
    assert r2.data["assignee"] == "agent-ops-1"
    assert "database" in r2.data["tags"]


@pytest.mark.asyncio
async def test_respond_single_action(skill):
    r1 = await skill.execute("detect", {
        "title": "Service down", "description": "HTTP 503", "source": "monitoring",
    })
    inc_id = r1.data["incident_id"]

    r2 = await skill.execute("respond", {
        "incident_id": inc_id,
        "action_type": "restart",
        "details": "Restarting API service container",
    })
    assert r2.success
    assert r2.data["action_type"] == "restart"
    assert r2.data["total_actions"] == 1


@pytest.mark.asyncio
async def test_escalate_incident(skill):
    r1 = await skill.execute("detect", {
        "title": "Data loss", "description": "Records missing", "source": "manual",
    })
    inc_id = r1.data["incident_id"]

    r2 = await skill.execute("escalate", {
        "incident_id": inc_id,
        "target": "senior-agent-1",
        "reason": "Requires database expertise",
        "new_severity": "sev1",
    })
    assert r2.success
    assert r2.data["target"] == "senior-agent-1"
    assert r2.data["severity"] == SEV1


@pytest.mark.asyncio
async def test_resolve_and_mttr(skill):
    r1 = await skill.execute("detect", {
        "title": "Cache failure", "description": "Redis down", "source": "alert",
    })
    inc_id = r1.data["incident_id"]

    r2 = await skill.execute("resolve", {
        "incident_id": inc_id,
        "resolution": "Restarted Redis cluster",
        "root_cause": "Memory limit exceeded",
        "follow_up_actions": ["Increase Redis memory limit", "Add memory alerts"],
    })
    assert r2.success
    assert r2.data["resolution"] == "Restarted Redis cluster"
    assert r2.data["mttr_seconds"] is not None
    assert r2.data["mttr_seconds"] >= 0


@pytest.mark.asyncio
async def test_resolve_already_resolved(skill):
    r1 = await skill.execute("detect", {
        "title": "Test", "description": "Test", "source": "manual",
    })
    inc_id = r1.data["incident_id"]
    await skill.execute("resolve", {"incident_id": inc_id, "resolution": "Fixed"})
    r3 = await skill.execute("resolve", {"incident_id": inc_id, "resolution": "Fixed again"})
    assert not r3.success
    assert "already resolved" in r3.message


@pytest.mark.asyncio
async def test_postmortem(skill):
    r1 = await skill.execute("detect", {
        "title": "Outage", "description": "Total outage", "source": "monitoring",
        "severity": "sev1", "service": "payments",
    })
    inc_id = r1.data["incident_id"]
    await skill.execute("respond", {
        "incident_id": inc_id, "action_type": "restart", "details": "Restart service"
    })
    await skill.execute("resolve", {
        "incident_id": inc_id, "resolution": "Restarted",
        "root_cause": "OOM kill", "follow_up_actions": ["Add memory limits"],
    })

    r = await skill.execute("postmortem", {"incident_id": inc_id})
    assert r.success
    assert r.data["root_cause"] == "OOM kill"
    assert r.data["mttr_seconds"] is not None
    assert r.data["response_time_seconds"] is not None
    assert isinstance(r.data["lessons_learned"], list)


@pytest.mark.asyncio
async def test_playbook_crud(skill):
    r1 = await skill.execute("playbook", {
        "operation": "create",
        "name": "API Restart Playbook",
        "trigger_conditions": {"severity": ["sev1", "sev2"], "service": ["api"]},
        "steps": [
            {"action_type": "notify", "details": "Alert on-call"},
            {"action_type": "restart", "details": "Restart API pods"},
        ],
    })
    assert r1.success
    pb_id = r1.data["playbook_id"]

    r2 = await skill.execute("playbook", {"operation": "list"})
    assert r2.success
    assert len(r2.data["playbooks"]) == 1

    r3 = await skill.execute("playbook", {"operation": "get", "playbook_id": pb_id})
    assert r3.success
    assert r3.data["name"] == "API Restart Playbook"

    r4 = await skill.execute("playbook", {"operation": "delete", "playbook_id": pb_id})
    assert r4.success


@pytest.mark.asyncio
async def test_playbook_execution(skill):
    r1 = await skill.execute("playbook", {
        "operation": "create", "name": "Recovery",
        "steps": [
            {"action_type": "notify", "details": "Alert team"},
            {"action_type": "restart", "details": "Restart"},
            {"action_type": "scale_up", "details": "Add replicas"},
        ],
    })
    pb_id = r1.data["playbook_id"]

    r2 = await skill.execute("detect", {
        "title": "Overload", "description": "CPU 100%", "source": "monitoring",
    })
    inc_id = r2.data["incident_id"]

    r3 = await skill.execute("respond", {
        "incident_id": inc_id, "playbook_id": pb_id,
    })
    assert r3.success
    assert r3.data["steps_executed"] == 3


@pytest.mark.asyncio
async def test_status_overview(skill):
    await skill.execute("detect", {"title": "Inc 1", "description": "d", "source": "alert", "severity": "sev1"})
    await skill.execute("detect", {"title": "Inc 2", "description": "d", "source": "alert", "severity": "sev3"})

    r = await skill.execute("status", {"include_metrics": True})
    assert r.success
    assert r.data["active_count"] == 2
    assert r.data["metrics"]["total_detected"] == 2


@pytest.mark.asyncio
async def test_status_filter(skill):
    r1 = await skill.execute("detect", {"title": "A", "description": "d", "source": "alert", "severity": "sev1"})
    await skill.execute("detect", {"title": "B", "description": "d", "source": "alert", "severity": "sev3"})
    await skill.execute("resolve", {"incident_id": r1.data["incident_id"], "resolution": "fixed"})

    r = await skill.execute("status", {"filter_status": STATUS_RESOLVED})
    assert r.data["total_count"] == 1


@pytest.mark.asyncio
async def test_auto_match_playbook(skill):
    await skill.execute("playbook", {
        "operation": "create", "name": "DB Playbook",
        "trigger_conditions": {"severity": ["sev1"], "service": ["database"]},
        "steps": [{"action_type": "restart", "details": "Restart DB"}],
    })

    r = await skill.execute("detect", {
        "title": "DB Down", "description": "Connection refused",
        "source": "monitoring", "service": "database", "severity": "sev1",
    })
    assert r.success
    assert r.data["matched_playbook"] is not None
    assert r.data["matched_playbook"]["name"] == "DB Playbook"


@pytest.mark.asyncio
async def test_validation_errors(skill):
    r = await skill.execute("detect", {"title": "", "description": ""})
    assert not r.success

    r = await skill.execute("triage", {"incident_id": "nope", "severity": "sev1"})
    assert not r.success

    r = await skill.execute("respond", {"incident_id": "nope", "action_type": "restart"})
    assert not r.success

    r = await skill.execute("unknown_action", {})
    assert not r.success
