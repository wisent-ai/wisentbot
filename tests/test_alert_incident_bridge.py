"""Tests for AlertIncidentBridgeSkill."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from singularity.skills.alert_incident_bridge import (
    AlertIncidentBridgeSkill, BRIDGE_STATE_FILE, DEFAULT_SEVERITY_MAP,
)


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data path."""
    s = AlertIncidentBridgeSkill()
    test_file = tmp_path / "alert_incident_bridge.json"
    import singularity.skills.alert_incident_bridge as mod
    mod.BRIDGE_STATE_FILE = test_file
    s._store = None
    return s


def _mock_context(alert_result=None, detect_result=None, resolve_result=None, triage_result=None):
    """Create a mock SkillContext that returns specified results."""
    ctx = MagicMock()

    async def mock_invoke(skill_id, action, params):
        result = MagicMock()
        if skill_id == "observability" and action == "check_alerts":
            result.success = True
            result.data = alert_result or {"fired": [], "resolved": [], "total_rules": 0}
        elif skill_id == "incident_response" and action == "detect":
            result.success = True
            result.data = detect_result or {"incident_id": f"INC-{id(params)}", "severity": "sev3", "status": "detected"}
        elif skill_id == "incident_response" and action == "resolve":
            result.success = True
            result.data = resolve_result or {"incident_id": params.get("incident_id"), "status": "resolved"}
        elif skill_id == "incident_response" and action == "triage":
            result.success = True
            result.data = triage_result or {"incident_id": params.get("incident_id"), "status": "triaged"}
        else:
            result.success = False
            result.data = {}
        return result

    ctx.invoke_skill = mock_invoke
    ctx.list_skills = MagicMock(return_value=["observability", "incident_response"])
    return ctx


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "alert_incident_bridge"
    assert len(m.actions) == 6
    action_names = [a.name for a in m.actions]
    assert "monitor" in action_names
    assert "configure" in action_names
    assert "link" in action_names


@pytest.mark.asyncio
async def test_monitor_no_alerts(skill):
    skill.context = _mock_context(alert_result={"fired": [], "resolved": [], "total_rules": 0})
    r = await skill.execute("monitor", {})
    assert r.success
    assert "no alert changes" in r.message
    assert r.data["incidents_created"] == []


@pytest.mark.asyncio
async def test_monitor_creates_incident_on_fired_alert(skill):
    fired = [{"name": "high_error_rate", "severity": "critical", "current_value": 15.0, "metric": "error_rate", "condition": "above 5"}]
    skill.context = _mock_context(
        alert_result={"fired": fired, "resolved": [], "total_rules": 1},
        detect_result={"incident_id": "INC-001", "severity": "sev1", "status": "detected"},
    )
    r = await skill.execute("monitor", {})
    assert r.success
    assert len(r.data["incidents_created"]) == 1
    assert r.data["incidents_created"][0]["incident_id"] == "INC-001"
    assert r.data["incidents_created"][0]["severity"] == "sev1"


@pytest.mark.asyncio
async def test_monitor_dry_run(skill):
    fired = [{"name": "high_latency", "severity": "warning", "current_value": 500, "metric": "latency_ms", "condition": "above 200"}]
    skill.context = _mock_context(alert_result={"fired": fired, "resolved": [], "total_rules": 1})
    r = await skill.execute("monitor", {"dry_run": True})
    assert r.success
    assert "[DRY RUN]" in r.message
    assert r.data["incidents_created"][0]["dry_run"] is True


@pytest.mark.asyncio
async def test_monitor_dedup(skill):
    """Second fired alert for same name should be deduped."""
    fired = [{"name": "cpu_high", "severity": "high", "current_value": 95, "metric": "cpu", "condition": "above 90"}]
    skill.context = _mock_context(
        alert_result={"fired": fired, "resolved": [], "total_rules": 1},
        detect_result={"incident_id": "INC-100", "severity": "sev2", "status": "detected"},
    )
    r1 = await skill.execute("monitor", {})
    assert len(r1.data["incidents_created"]) == 1

    # Second cycle same alert
    r2 = await skill.execute("monitor", {})
    assert len(r2.data["deduped"]) == 1
    assert r2.data["incidents_created"] == []


@pytest.mark.asyncio
async def test_monitor_auto_resolves(skill):
    """Resolved alerts should auto-resolve linked incidents."""
    # First create an incident
    fired = [{"name": "disk_full", "severity": "critical", "current_value": 95, "metric": "disk_pct", "condition": "above 90"}]
    skill.context = _mock_context(
        alert_result={"fired": fired, "resolved": [], "total_rules": 1},
        detect_result={"incident_id": "INC-200", "severity": "sev1", "status": "detected"},
    )
    await skill.execute("monitor", {})

    # Now resolve
    skill.context = _mock_context(
        alert_result={"fired": [], "resolved": [{"name": "disk_full", "current_value": 50}], "total_rules": 1},
    )
    r = await skill.execute("monitor", {})
    assert r.success
    assert len(r.data["incidents_resolved"]) == 1
    assert r.data["incidents_resolved"][0]["incident_id"] == "INC-200"


@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {
        "auto_resolve": False,
        "dedup_window_minutes": 60,
        "default_assignee": "agent-1",
    })
    assert r.success
    assert r.data["config"]["auto_resolve"] is False
    assert r.data["config"]["dedup_window_minutes"] == 60
    assert r.data["config"]["default_assignee"] == "agent-1"


@pytest.mark.asyncio
async def test_configure_severity_map(skill):
    r = await skill.execute("configure", {"severity_map": {"info": "sev4", "critical": "sev1"}})
    assert r.success
    assert r.data["config"]["severity_map"]["info"] == "sev4"


@pytest.mark.asyncio
async def test_configure_invalid_severity(skill):
    r = await skill.execute("configure", {"severity_map": {"info": "invalid"}})
    assert not r.success


@pytest.mark.asyncio
async def test_status(skill):
    r = await skill.execute("status", {})
    assert r.success
    assert "active_links" in r.data
    assert "stats" in r.data


@pytest.mark.asyncio
async def test_mappings_view(skill):
    r = await skill.execute("mappings", {})
    assert r.success
    assert r.data["default_map"] == DEFAULT_SEVERITY_MAP


@pytest.mark.asyncio
async def test_mappings_override(skill):
    r = await skill.execute("mappings", {"alert_name": "special_alert", "incident_severity": "sev1"})
    assert r.success
    assert "Override set" in r.message
    r2 = await skill.execute("mappings", {"alert_name": "special_alert"})
    assert r2.data["mapped_severity"] == "sev1"
    assert r2.data["is_override"] is True


@pytest.mark.asyncio
async def test_history(skill):
    r = await skill.execute("history", {"limit": 5})
    assert r.success
    assert r.data["total"] == 0


@pytest.mark.asyncio
async def test_link(skill):
    r = await skill.execute("link", {"alert_name": "my_alert", "incident_id": "INC-999"})
    assert r.success
    assert "Linked" in r.message
    store = skill._load()
    assert "my_alert" in store["active_links"]
    assert store["active_links"]["my_alert"]["incident_id"] == "INC-999"
