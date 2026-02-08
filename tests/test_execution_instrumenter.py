"""Tests for SkillExecutionInstrumenter skill."""
import pytest, json, asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.skills.execution_instrumenter import SkillExecutionInstrumenter, DATA_FILE


@pytest.fixture(autouse=True)
def clean_data():
    if DATA_FILE.exists():
        DATA_FILE.unlink()
    yield
    if DATA_FILE.exists():
        DATA_FILE.unlink()


@pytest.fixture
def skill():
    s = SkillExecutionInstrumenter()
    return s


@pytest.fixture
def skill_with_context():
    s = SkillExecutionInstrumenter()
    ctx = MagicMock()
    ctx.call_skill = AsyncMock(return_value=MagicMock(data={"ok": True}))
    ctx.registry = MagicMock()
    bridge_mock = MagicMock()
    bridge_mock.emit_bridge_events = AsyncMock(return_value=[{"topic": "test.event"}])
    ctx.registry.get = MagicMock(side_effect=lambda k: bridge_mock if k == "skill_event_bridge" else MagicMock())
    s.context = ctx
    return s


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "execution_instrumenter"
    assert len(m.actions) == 6
    action_names = [a.name for a in m.actions]
    assert "instrument" in action_names
    assert "stats" in action_names
    assert "top_skills" in action_names
    assert "health" in action_names


@pytest.mark.asyncio
async def test_instrument_basic(skill):
    """Instrument without context - should still record locally."""
    r = await skill.execute("instrument", {
        "skill_id": "test_skill", "action": "test_action",
        "success": True, "latency_ms": 42.5,
    })
    assert r.success
    assert "test_skill" in r.message
    assert r.data["latency_ms"] == 42.5


@pytest.mark.asyncio
async def test_instrument_with_metrics(skill_with_context):
    """Instrument with context emits observability metrics."""
    r = await skill_with_context.execute("instrument", {
        "skill_id": "incident_response", "action": "detect",
        "success": True, "latency_ms": 150.0,
        "result_data": {"incident_id": "INC-1", "severity": "sev2"},
    })
    assert r.success
    assert len(r.data["metrics_emitted"]) >= 2  # count + latency
    assert "skill.execution.count" in r.data["metrics_emitted"]
    assert "skill.execution.latency_ms" in r.data["metrics_emitted"]


@pytest.mark.asyncio
async def test_instrument_error_emits_error_metric(skill_with_context):
    """Failed executions emit error counter."""
    r = await skill_with_context.execute("instrument", {
        "skill_id": "test_skill", "action": "broken",
        "success": False, "latency_ms": 10.0, "error": "Something broke",
    })
    assert r.success
    assert "skill.execution.errors" in r.data["metrics_emitted"]


@pytest.mark.asyncio
async def test_instrument_excluded_skill(skill):
    """Excluded skills are skipped."""
    r = await skill.execute("instrument", {
        "skill_id": "execution_instrumenter", "action": "stats",
        "success": True, "latency_ms": 1.0,
    })
    assert r.success
    assert r.data.get("skipped") is True


@pytest.mark.asyncio
async def test_stats_accumulate(skill):
    for i in range(5):
        await skill.execute("instrument", {
            "skill_id": "my_skill", "action": "run",
            "success": i != 3, "latency_ms": float(i * 10),
        })
    r = await skill.execute("stats", {})
    assert r.success
    assert r.data["global_stats"]["total_instrumented"] == 5
    assert r.data["unique_skills"] == 1


@pytest.mark.asyncio
async def test_recent(skill):
    for i in range(3):
        await skill.execute("instrument", {
            "skill_id": f"skill_{i}", "action": "run",
            "success": True, "latency_ms": 10.0,
        })
    r = await skill.execute("recent", {"limit": 2})
    assert r.success
    assert len(r.data["executions"]) == 2


@pytest.mark.asyncio
async def test_recent_with_filter(skill):
    await skill.execute("instrument", {"skill_id": "a", "action": "run", "success": True, "latency_ms": 1.0})
    await skill.execute("instrument", {"skill_id": "b", "action": "run", "success": True, "latency_ms": 1.0})
    r = await skill.execute("recent", {"skill_filter": "a"})
    assert r.success
    assert all(e["skill_id"] == "a" for e in r.data["executions"])


@pytest.mark.asyncio
async def test_top_skills(skill):
    for _ in range(5):
        await skill.execute("instrument", {"skill_id": "fast", "action": "run", "success": True, "latency_ms": 10.0})
    for _ in range(2):
        await skill.execute("instrument", {"skill_id": "slow", "action": "run", "success": True, "latency_ms": 500.0})
    r = await skill.execute("top_skills", {"sort_by": "latency"})
    assert r.success
    assert r.data["rankings"][0]["skill_id"] == "slow"

    r2 = await skill.execute("top_skills", {"sort_by": "count"})
    assert r2.data["rankings"][0]["skill_id"] == "fast"


@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {"emit_count_metric": False, "alert_check_interval": 100})
    assert r.success
    assert r.data["config"]["emit_count_metric"] is False
    assert r.data["config"]["alert_check_interval"] == 100


@pytest.mark.asyncio
async def test_health_no_context(skill):
    r = await skill.execute("health", {})
    assert r.success
    assert r.data["healthy"] is False


@pytest.mark.asyncio
async def test_health_with_context(skill_with_context):
    r = await skill_with_context.execute("health", {})
    assert r.success
    assert r.data["observability_available"] is True
    assert r.data["bridge_available"] is True


@pytest.mark.asyncio
async def test_alert_check_periodic(skill_with_context):
    """Alerts are checked periodically based on alert_check_interval."""
    await skill_with_context.execute("configure", {"alert_check_interval": 3})
    for i in range(4):
        r = await skill_with_context.execute("instrument", {
            "skill_id": "test", "action": "run",
            "success": True, "latency_ms": 1.0,
        })
    # After 3 executions, alerts should have been checked
    stats = await skill_with_context.execute("stats", {})
    assert stats.data["global_stats"]["total_alerts_checked"] >= 1


@pytest.mark.asyncio
async def test_missing_params(skill):
    r = await skill.execute("instrument", {})
    assert not r.success
    assert "required" in r.message.lower()
