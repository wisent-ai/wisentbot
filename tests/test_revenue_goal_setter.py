"""Tests for RevenueGoalAutoSetterSkill."""

import pytest
import json
from singularity.skills.revenue_goal_setter import (
    RevenueGoalAutoSetterSkill,
    SETTER_FILE,
    DASHBOARD_FILE,
    GOALS_FILE,
    DEFAULT_CONFIG,
)
import singularity.skills.revenue_goal_setter as mod


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data paths."""
    mod.SETTER_FILE = tmp_path / "revenue_goal_setter.json"
    mod.DASHBOARD_FILE = tmp_path / "revenue_analytics_dashboard.json"
    mod.GOALS_FILE = tmp_path / "goals.json"
    s = RevenueGoalAutoSetterSkill()
    return s


def _write_dashboard(snapshots, config=None):
    cfg = config or {"compute_cost_per_hour": 0.10}
    mod.DASHBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(mod.DASHBOARD_FILE, "w") as f:
        json.dump({"snapshots": snapshots, "config": cfg}, f)


def _write_goals(goals):
    mod.GOALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(mod.GOALS_FILE, "w") as f:
        json.dump({"goals": goals, "completed_goals": [], "session_log": []}, f)


def _read_goals():
    with open(mod.GOALS_FILE, "r") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "revenue_goal_setter"
    assert len(m.actions) == 6
    names = [a.name for a in m.actions]
    assert "evaluate" in names
    assert "set_goal" in names
    assert "status" in names
    assert "sync" in names


@pytest.mark.asyncio
async def test_evaluate_no_dashboard(skill):
    r = await skill.execute("evaluate", {})
    assert r.success
    assert "no_data" in r.data.get("reason", "")


@pytest.mark.asyncio
async def test_evaluate_insufficient_snapshots(skill):
    _write_dashboard([{"total_revenue": 0.1}, {"total_revenue": 0.2}])
    r = await skill.execute("evaluate", {})
    assert r.success
    assert r.data.get("reason") == "insufficient_data"


@pytest.mark.asyncio
async def test_evaluate_creates_breakeven_goal(skill):
    # Revenue below cost → should create breakeven goal
    snaps = [{"total_revenue": 0.01 * i} for i in range(1, 6)]
    _write_dashboard(snaps)
    _write_goals([])

    r = await skill.execute("evaluate", {})
    assert r.success
    assert len(r.data["goals_created"]) == 1
    assert r.data["goals_created"][0]["type"] == "breakeven"
    assert r.data["is_profitable"] is False

    # Verify goal was created in goals file
    goals = _read_goals()
    assert len(goals["goals"]) == 1
    assert "breakeven" in goals["goals"][0]["title"].lower()
    assert goals["goals"][0]["pillar"] == "revenue"
    assert goals["goals"][0]["priority"] == "critical"


@pytest.mark.asyncio
async def test_evaluate_creates_growth_goal_when_profitable(skill):
    # Revenue above cost → should create growth goal
    snaps = [{"total_revenue": 3.0 + 0.1 * i} for i in range(5)]
    _write_dashboard(snaps)
    _write_goals([])

    r = await skill.execute("evaluate", {})
    assert r.success
    created = r.data["goals_created"]
    assert len(created) == 1
    assert created[0]["type"] == "growth"
    assert r.data["is_profitable"] is True


@pytest.mark.asyncio
async def test_evaluate_no_duplicate_goals(skill):
    # First evaluate creates goal
    snaps = [{"total_revenue": 0.01 * i} for i in range(1, 6)]
    _write_dashboard(snaps)
    _write_goals([])
    r1 = await skill.execute("evaluate", {})
    assert len(r1.data["goals_created"]) == 1

    # Second evaluate should NOT create a duplicate
    r2 = await skill.execute("evaluate", {})
    assert len(r2.data["goals_created"]) == 0


@pytest.mark.asyncio
async def test_set_goal_manual(skill):
    _write_goals([])
    r = await skill.execute("set_goal", {"target_daily": 5.0, "deadline_days": 14})
    assert r.success
    assert r.data["target_daily"] == 5.0
    assert r.data["goal_id"]

    goals = _read_goals()
    assert len(goals["goals"]) == 1
    assert "$5.00" in goals["goals"][0]["title"]


@pytest.mark.asyncio
async def test_set_goal_validation(skill):
    r = await skill.execute("set_goal", {})
    assert not r.success
    assert "required" in r.message

    r2 = await skill.execute("set_goal", {"target_daily": -1})
    assert not r2.success
    assert "positive" in r2.message


@pytest.mark.asyncio
async def test_status(skill):
    _write_goals([])
    r = await skill.execute("status", {})
    assert r.success
    assert "0 auto-goal" in r.message


@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {"key": "target_margin_multiplier", "value": 2.0})
    assert r.success
    assert r.data["value"] == 2.0
    assert r.data["old_value"] == 1.5

    # Invalid key
    r2 = await skill.execute("configure", {"key": "nonexistent", "value": 1})
    assert not r2.success


@pytest.mark.asyncio
async def test_history(skill):
    r = await skill.execute("history", {})
    assert r.success
    assert r.data["total"] == 0


@pytest.mark.asyncio
async def test_sync_updates_progress(skill):
    snaps = [{"total_revenue": 0.5 + 0.1 * i} for i in range(5)]
    _write_dashboard(snaps)
    _write_goals([])

    # Create a goal first
    await skill.execute("set_goal", {"target_daily": 2.0})

    # Sync
    r = await skill.execute("sync", {})
    assert r.success
    assert r.data["synced"] == 1

    goals = _read_goals()
    notes = goals["goals"][0].get("progress_notes", [])
    assert any("[SYNC]" in n for n in notes)


@pytest.mark.asyncio
async def test_escalation_note(skill):
    # Revenue far exceeds target
    snaps = [{"total_revenue": 10.0 + 0.1 * i} for i in range(5)]
    _write_dashboard(snaps)
    _write_goals([])

    # Create auto breakeven goal (won't happen since profitable), so create manual
    await skill.execute("set_goal", {"target_daily": 1.0})

    # Now evaluate - should add escalation note since 10.4 >> 1.0
    r = await skill.execute("evaluate", {})
    assert r.success
    # Check for escalation
    updated = r.data.get("goals_updated", [])
    escalations = [u for u in updated if u.get("action") == "escalation_note"]
    assert len(escalations) >= 1


@pytest.mark.asyncio
async def test_disabled(skill):
    await skill.execute("configure", {"key": "enabled", "value": False})
    r = await skill.execute("evaluate", {})
    assert r.success
    assert r.data.get("enabled") is False


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success
