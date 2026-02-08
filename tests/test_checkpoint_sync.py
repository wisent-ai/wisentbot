"""Tests for CrossAgentCheckpointSyncSkill."""

import json
import pytest
from pathlib import Path

from singularity.skills.checkpoint_sync import (
    CrossAgentCheckpointSyncSkill,
    SYNC_FILE,
)


@pytest.fixture(autouse=True)
def clean_sync_file():
    if SYNC_FILE.exists():
        SYNC_FILE.unlink()
    yield
    if SYNC_FILE.exists():
        SYNC_FILE.unlink()


@pytest.fixture
def skill():
    return CrossAgentCheckpointSyncSkill()


def share_params(agent_id="agent-1", checkpoint_id="cp-001", **kwargs):
    base = {
        "agent_id": agent_id,
        "checkpoint_id": checkpoint_id,
        "pillar_scores": {"self_improvement": 60, "revenue": 40, "replication": 30, "goal_setting": 50},
        "label": "session-10",
        "skills_active": ["memory", "experiment", "strategy"],
        "experiments_running": 2,
        "goals_completed": 5,
    }
    base.update(kwargs)
    return base


@pytest.mark.asyncio
async def test_share_checkpoint(skill):
    result = await skill.execute("share", share_params())
    assert result.success
    assert "agent-1" in result.message
    assert result.data["summary"]["agent_id"] == "agent-1"
    assert result.data["summary"]["pillar_scores"]["revenue"] == 40


@pytest.mark.asyncio
async def test_share_duplicate_rejected(skill):
    await skill.execute("share", share_params())
    result = await skill.execute("share", share_params())
    assert not result.success
    assert "already shared" in result.message


@pytest.mark.asyncio
async def test_share_requires_params(skill):
    result = await skill.execute("share", {})
    assert not result.success


@pytest.mark.asyncio
async def test_pull_all_peers(skill):
    await skill.execute("share", share_params("agent-1", "cp-001"))
    await skill.execute("share", share_params("agent-2", "cp-002"))
    result = await skill.execute("pull", {})
    assert result.success
    assert len(result.data["agents"]) == 2


@pytest.mark.asyncio
async def test_pull_specific_peer(skill):
    await skill.execute("share", share_params("agent-1", "cp-001"))
    await skill.execute("share", share_params("agent-2", "cp-002"))
    result = await skill.execute("pull", {"peer_agent_id": "agent-1"})
    assert result.success
    assert result.data["agents"] == ["agent-1"]


@pytest.mark.asyncio
async def test_fleet_timeline(skill):
    await skill.execute("share", share_params("agent-1", "cp-001"))
    await skill.execute("share", share_params("agent-2", "cp-002"))
    result = await skill.execute("fleet_timeline", {})
    assert result.success
    assert result.data["total_entries"] == 2
    assert len(result.data["timeline"]) == 2


@pytest.mark.asyncio
async def test_divergence_detection(skill):
    await skill.execute("share", share_params("agent-1", "cp-001",
        pillar_scores={"revenue": 80, "replication": 20}))
    await skill.execute("share", share_params("agent-2", "cp-002",
        pillar_scores={"revenue": 20, "replication": 80}))
    result = await skill.execute("divergence", {})
    assert result.success
    assert len(result.data["comparisons"]) == 1
    assert result.data["comparisons"][0]["is_divergent"]
    assert len(result.data["alerts"]) == 1


@pytest.mark.asyncio
async def test_divergence_no_alert_when_similar(skill):
    await skill.execute("share", share_params("agent-1", "cp-001",
        pillar_scores={"revenue": 50, "replication": 50}))
    await skill.execute("share", share_params("agent-2", "cp-002",
        pillar_scores={"revenue": 55, "replication": 48}))
    result = await skill.execute("divergence", {})
    assert result.success
    assert len(result.data["alerts"]) == 0


@pytest.mark.asyncio
async def test_best_practices_ranking(skill):
    await skill.execute("share", share_params("agent-1", "cp-001",
        pillar_scores={"revenue": 80, "replication": 70}))
    await skill.execute("share", share_params("agent-2", "cp-002",
        pillar_scores={"revenue": 30, "replication": 40}))
    result = await skill.execute("best_practices", {})
    assert result.success
    assert result.data["rankings"][0]["agent_id"] == "agent-1"
    assert len(result.data["recommendations"]) >= 1


@pytest.mark.asyncio
async def test_sync_policy_update(skill):
    result = await skill.execute("sync_policy", {"divergence_threshold": 0.5})
    assert result.success
    assert result.data["config"]["divergence_threshold"] == 0.5


@pytest.mark.asyncio
async def test_sync_policy_no_changes(skill):
    result = await skill.execute("sync_policy", {})
    assert result.success
    assert "No changes" in result.message


@pytest.mark.asyncio
async def test_merge_insights(skill):
    result = await skill.execute("merge_insights", {
        "source_agent_id": "agent-best",
        "insight": "Use higher temperature for creative tasks",
        "category": "config",
    })
    assert result.success
    assert result.data["insight"]["source_agent"] == "agent-best"
    assert result.data["insight"]["category"] == "config"


@pytest.mark.asyncio
async def test_merge_insights_duplicate_rejected(skill):
    params = {"source_agent_id": "agent-1", "insight": "Use caching", "category": "strategy"}
    await skill.execute("merge_insights", params)
    result = await skill.execute("merge_insights", params)
    assert not result.success
    assert "already recorded" in result.message


@pytest.mark.asyncio
async def test_status(skill):
    await skill.execute("share", share_params())
    result = await skill.execute("status", {})
    assert result.success
    assert result.data["stats"]["shares_sent"] == 1
    assert "agent-1" in result.data["peers"]


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "cross_agent_checkpoint_sync"
    assert m.category == "replication"
    assert len(m.actions) == 8
