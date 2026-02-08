"""Tests for CapabilityAwareDelegationSkill."""

import pytest
import json
from singularity.skills.capability_delegation import (
    CapabilityAwareDelegationSkill, DATA_FILE, SKILL_CATEGORIES,
)
import singularity.skills.capability_delegation as mod

MOCK_PROFILES = {
    "agent-coder": {
        "categories": {
            "self_improvement": {"score": 85, "skills": 8},
            "revenue": {"score": 40, "skills": 3},
            "replication": {"score": 30, "skills": 2},
            "operations": {"score": 60, "skills": 4},
        },
        "installed_skills": ["code_review", "self_modify", "deployment", "shell"],
    },
    "agent-sales": {
        "categories": {
            "self_improvement": {"score": 20, "skills": 2},
            "revenue": {"score": 90, "skills": 6},
            "communication": {"score": 75, "skills": 3},
            "operations": {"score": 30, "skills": 2},
        },
        "installed_skills": ["content", "payment", "email", "marketplace"],
    },
    "agent-ops": {
        "categories": {
            "operations": {"score": 95, "skills": 7},
            "self_improvement": {"score": 50, "skills": 4},
            "replication": {"score": 60, "skills": 4},
        },
        "installed_skills": ["deployment", "health_monitor", "observability", "scheduler"],
    },
}

MOCK_REPUTATIONS = {
    "agent-coder": {"overall": 75.0, "competence": 80.0, "reliability": 70.0},
    "agent-sales": {"overall": 60.0, "competence": 55.0, "reliability": 65.0},
    "agent-ops": {"overall": 85.0, "competence": 90.0, "reliability": 80.0},
}


@pytest.fixture
def skill(tmp_path):
    s = CapabilityAwareDelegationSkill()
    mod.DATA_FILE = tmp_path / "capability_delegation.json"

    async def mock_profiles():
        return dict(MOCK_PROFILES)
    async def mock_reps():
        return dict(MOCK_REPUTATIONS)

    s._get_agent_profiles = mock_profiles
    s._get_reputations = mock_reps
    return s


@pytest.fixture
def skill_no_data(tmp_path):
    s = CapabilityAwareDelegationSkill()
    mod.DATA_FILE = tmp_path / "capability_delegation.json"
    async def no_profiles():
        return {}
    async def no_reps():
        return {}
    s._get_agent_profiles = no_profiles
    s._get_reputations = no_reps
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "capability_delegation"
    assert len(m.actions) == 6
    names = [a.name for a in m.actions]
    assert "match" in names
    assert "delegate" in names


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_match_by_category(skill):
    r = await skill.execute("match", {
        "task_name": "deploy service",
        "required_categories": ["operations"],
    })
    assert r.success
    matches = r.data["matches"]
    assert len(matches) >= 1
    # agent-ops should be top match for operations
    assert matches[0]["agent_id"] == "agent-ops"


@pytest.mark.asyncio
async def test_match_by_skill(skill):
    r = await skill.execute("match", {
        "task_name": "code review",
        "required_skills": ["code_review"],
    })
    assert r.success
    matches = r.data["matches"]
    assert any(m["agent_id"] == "agent-coder" for m in matches)


@pytest.mark.asyncio
async def test_match_revenue(skill):
    r = await skill.execute("match", {
        "task_name": "create content",
        "required_categories": ["revenue", "communication"],
    })
    assert r.success
    assert r.data["matches"][0]["agent_id"] == "agent-sales"


@pytest.mark.asyncio
async def test_match_no_profiles(skill_no_data):
    r = await skill_no_data.execute("match", {
        "task_name": "test",
        "required_categories": ["operations"],
    })
    assert not r.success
    assert "No agent capability profiles" in r.message


@pytest.mark.asyncio
async def test_match_infers_categories(skill):
    r = await skill.execute("match", {"task_name": "deploy and monitor service"})
    assert r.success
    reqs = r.data["requirements"]
    assert "operations" in reqs["categories"]


@pytest.mark.asyncio
async def test_delegate(skill):
    r = await skill.execute("delegate", {
        "task_name": "review code changes",
        "required_categories": ["self_improvement"],
        "budget": 5.0,
    })
    assert r.success
    assert r.data["match"]["agent_id"] == "agent-coder"


@pytest.mark.asyncio
async def test_delegate_tracks_history(skill):
    await skill.execute("delegate", {
        "task_name": "test task",
        "required_categories": ["operations"],
    })
    r = await skill.execute("history", {})
    assert len(r.data["history"]) == 1
    assert r.data["history"][0]["task_name"] == "test task"


@pytest.mark.asyncio
async def test_profiles(skill):
    r = await skill.execute("profiles", {})
    assert r.success
    assert len(r.data["profiles"]) == 3


@pytest.mark.asyncio
async def test_profiles_filter(skill):
    r = await skill.execute("profiles", {"agent_id": "agent-coder"})
    assert r.success
    assert "agent-coder" in r.data["profiles"]
    assert len(r.data["profiles"]) == 1


@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {"capability_weight": 0.8, "reputation_weight": 0.2})
    assert r.success
    assert r.data["config"]["capability_weight"] == 0.8


@pytest.mark.asyncio
async def test_status(skill):
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["agent_count"] == 3


@pytest.mark.asyncio
async def test_min_match_threshold(skill):
    # Set very high threshold
    await skill.execute("configure", {"min_match_score": 0.99})
    r = await skill.execute("match", {
        "task_name": "impossible task",
        "required_categories": ["communication"],
    })
    assert r.success
    assert len(r.data["matches"]) == 0
