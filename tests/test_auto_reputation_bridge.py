"""Tests for AutoReputationBridgeSkill."""

import pytest
import json
from singularity.skills.auto_reputation_bridge import (
    AutoReputationBridgeSkill,
    BRIDGE_DATA_FILE,
    DELEGATION_FILE,
    REPUTATION_FILE,
    DEFAULT_CONFIG,
)
import singularity.skills.auto_reputation_bridge as mod


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data paths."""
    mod.BRIDGE_DATA_FILE = tmp_path / "auto_reputation_bridge.json"
    mod.DELEGATION_FILE = tmp_path / "task_delegations.json"
    mod.REPUTATION_FILE = tmp_path / "agent_reputation.json"
    s = AutoReputationBridgeSkill()
    return s


def _write_delegations(delegations):
    """Helper to write delegation data to the temp file."""
    mod.DELEGATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(mod.DELEGATION_FILE, "w") as f:
        json.dump({"delegations": delegations}, f)


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "auto_reputation_bridge"
    assert len(m.actions) == 6
    names = [a.name for a in m.actions]
    assert "poll" in names
    assert "configure" in names
    assert "status" in names
    assert "history" in names
    assert "reprocess" in names
    assert "stats" in names


@pytest.mark.asyncio
async def test_poll_no_delegations(skill):
    r = await skill.execute("poll", {})
    assert r.success
    assert r.data["processed"] == 0


@pytest.mark.asyncio
async def test_poll_completed_delegation(skill):
    _write_delegations([{
        "delegation_id": "del_001",
        "agent_id": "agent_alpha",
        "task_name": "code_review",
        "status": "completed",
        "budget": 10.0,
        "budget_spent": 5.0,
        "created_at": "2026-01-01T00:00:00",
        "completed_at": "2026-01-01T01:00:00",
    }])
    r = await skill.execute("poll", {})
    assert r.success
    assert r.data["processed"] == 1
    assert r.data["successes"] == 1
    assert r.data["failures"] == 0

    # Check reputation file was created
    with open(mod.REPUTATION_FILE) as f:
        rep_data = json.load(f)
    agent_rep = rep_data["reputations"]["agent_alpha"]
    assert agent_rep["competence"] > 50.0
    assert agent_rep["reliability"] > 50.0
    assert agent_rep["tasks_completed"] == 1


@pytest.mark.asyncio
async def test_poll_failed_delegation(skill):
    _write_delegations([{
        "delegation_id": "del_002",
        "agent_id": "agent_beta",
        "task_name": "deploy_service",
        "status": "failed",
        "budget": 20.0,
        "budget_spent": 15.0,
        "created_at": "2026-01-01T00:00:00",
        "completed_at": "2026-01-01T02:00:00",
    }])
    r = await skill.execute("poll", {})
    assert r.success
    assert r.data["processed"] == 1
    assert r.data["failures"] == 1

    with open(mod.REPUTATION_FILE) as f:
        rep_data = json.load(f)
    agent_rep = rep_data["reputations"]["agent_beta"]
    assert agent_rep["competence"] < 50.0
    assert agent_rep["reliability"] < 50.0
    assert agent_rep["tasks_failed"] == 1


@pytest.mark.asyncio
async def test_poll_deduplication(skill):
    """Same delegation should not be processed twice."""
    _write_delegations([{
        "delegation_id": "del_003",
        "agent_id": "agent_gamma",
        "task_name": "test_task",
        "status": "completed",
        "budget": 10.0,
        "budget_spent": 3.0,
    }])
    r1 = await skill.execute("poll", {})
    assert r1.data["processed"] == 1

    r2 = await skill.execute("poll", {})
    assert r2.data["processed"] == 0


@pytest.mark.asyncio
async def test_poll_dry_run(skill):
    _write_delegations([{
        "delegation_id": "del_004",
        "agent_id": "agent_delta",
        "task_name": "analysis",
        "status": "completed",
        "budget": 10.0,
        "budget_spent": 8.0,
    }])
    r = await skill.execute("poll", {"dry_run": True})
    assert r.success
    assert r.data["dry_run"] is True
    assert r.data["processed"] == 1
    assert not mod.REPUTATION_FILE.exists()

    # Should still be processable after dry run
    r2 = await skill.execute("poll", {})
    assert r2.data["processed"] == 1


@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {"key": "enabled", "value": False})
    assert r.success
    assert r.data["value"] is False

    # Poll should be disabled now
    r2 = await skill.execute("poll", {})
    assert not r2.success
    assert "disabled" in r2.message.lower()


@pytest.mark.asyncio
async def test_configure_invalid_key(skill):
    r = await skill.execute("configure", {"key": "fake_key", "value": 1})
    assert not r.success


@pytest.mark.asyncio
async def test_status(skill):
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["enabled"] is True
    assert r.data["total_processed"] == 0


@pytest.mark.asyncio
async def test_history(skill):
    _write_delegations([{
        "delegation_id": "del_005",
        "agent_id": "agent_eps",
        "task_name": "work",
        "status": "completed",
        "budget": 5.0,
        "budget_spent": 2.0,
    }])
    await skill.execute("poll", {})
    r = await skill.execute("history", {"limit": 10})
    assert r.success
    assert len(r.data["entries"]) == 1
    assert r.data["entries"][0]["agent_id"] == "agent_eps"


@pytest.mark.asyncio
async def test_history_filter_by_agent(skill):
    _write_delegations([
        {"delegation_id": "del_a", "agent_id": "a1", "task_name": "t1", "status": "completed", "budget": 1, "budget_spent": 0.5},
        {"delegation_id": "del_b", "agent_id": "a2", "task_name": "t2", "status": "failed", "budget": 1, "budget_spent": 0.5},
    ])
    await skill.execute("poll", {})
    r = await skill.execute("history", {"agent_id": "a1"})
    assert len(r.data["entries"]) == 1
    assert r.data["entries"][0]["agent_id"] == "a1"


@pytest.mark.asyncio
async def test_reprocess(skill):
    _write_delegations([{
        "delegation_id": "del_006",
        "agent_id": "agent_zeta",
        "task_name": "reprocess_me",
        "status": "completed",
        "budget": 10.0,
        "budget_spent": 5.0,
    }])
    await skill.execute("poll", {})
    r = await skill.execute("reprocess", {"delegation_id": "del_006"})
    assert r.success
    assert r.data["was_processed"] is True


@pytest.mark.asyncio
async def test_stats_empty(skill):
    r = await skill.execute("stats", {})
    assert r.success
    assert r.data["total"] == 0


@pytest.mark.asyncio
async def test_stats_with_data(skill):
    _write_delegations([
        {"delegation_id": "d1", "agent_id": "a1", "task_name": "t1", "status": "completed", "budget": 10, "budget_spent": 5},
        {"delegation_id": "d2", "agent_id": "a1", "task_name": "t2", "status": "failed", "budget": 10, "budget_spent": 8},
        {"delegation_id": "d3", "agent_id": "a2", "task_name": "t3", "status": "completed", "budget": 10, "budget_spent": 2},
    ])
    await skill.execute("poll", {})
    r = await skill.execute("stats", {})
    assert r.success
    assert r.data["total"] == 3
    assert r.data["successes"] == 2
    assert r.data["failures"] == 1
    assert r.data["success_rate"] > 0
    assert "a1" in r.data["agent_breakdown"]
    assert "a2" in r.data["agent_breakdown"]


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_skips_pending_delegations(skill):
    _write_delegations([
        {"delegation_id": "d_pending", "agent_id": "a1", "task_name": "t", "status": "pending", "budget": 1, "budget_spent": 0},
        {"delegation_id": "d_active", "agent_id": "a1", "task_name": "t", "status": "in_progress", "budget": 1, "budget_spent": 0},
    ])
    r = await skill.execute("poll", {})
    assert r.data["processed"] == 0
