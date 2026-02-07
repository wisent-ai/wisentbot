"""Tests for GoalManagerSkill."""
import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from singularity.skills.goal_manager import GoalManagerSkill


@pytest.fixture
def skill(tmp_path):
    goals_file = tmp_path / "goals.json"
    with patch("singularity.skills.goal_manager.GOALS_FILE", goals_file):
        with patch("singularity.skills.goal_manager.GOALS_DIR", tmp_path):
            s = GoalManagerSkill()
            yield s


@pytest.mark.asyncio
async def test_add_goal(skill):
    r = await skill.execute("add", {"title": "Test Goal", "priority": "high", "pillar": "self_improvement"})
    assert r.success
    assert "goal_id" in r.data


@pytest.mark.asyncio
async def test_add_goal_no_title(skill):
    r = await skill.execute("add", {"title": ""})
    assert not r.success


@pytest.mark.asyncio
async def test_list_goals(skill):
    await skill.execute("add", {"title": "Goal 1"})
    await skill.execute("add", {"title": "Goal 2"})
    r = await skill.execute("list", {})
    assert r.success
    assert r.data["count"] == 2


@pytest.mark.asyncio
async def test_add_and_complete_task(skill):
    r = await skill.execute("add", {"title": "Goal"})
    gid = r.data["goal_id"]
    await skill.execute("add_task", {"goal_id": gid, "title": "Task 1"})
    await skill.execute("add_task", {"goal_id": gid, "title": "Task 2"})
    r = await skill.execute("complete_task", {"goal_id": gid, "task_index": 0})
    assert r.success
    assert r.data["progress"] == 50


@pytest.mark.asyncio
async def test_update_goal(skill):
    r = await skill.execute("add", {"title": "Goal"})
    gid = r.data["goal_id"]
    r = await skill.execute("update", {"goal_id": gid, "progress": 75, "note": "Making progress"})
    assert r.success
    assert r.data["goal"]["progress"] == 75


@pytest.mark.asyncio
async def test_auto_complete(skill):
    r = await skill.execute("add", {"title": "Goal"})
    gid = r.data["goal_id"]
    r = await skill.execute("update", {"goal_id": gid, "progress": 100})
    assert r.data["goal"]["status"] == "completed"


@pytest.mark.asyncio
async def test_focus(skill):
    await skill.execute("add", {"title": "Low", "priority": "low"})
    await skill.execute("add", {"title": "Critical", "priority": "critical"})
    r = await skill.execute("focus", {})
    assert r.success
    assert r.data["focus_goal"]["priority"] == "critical"


@pytest.mark.asyncio
async def test_remove_goal(skill):
    r = await skill.execute("add", {"title": "Temp"})
    gid = r.data["goal_id"]
    r = await skill.execute("remove", {"goal_id": gid})
    assert r.success
    r = await skill.execute("list", {})
    assert r.data["count"] == 0


@pytest.mark.asyncio
async def test_summary(skill):
    await skill.execute("add", {"title": "G1", "pillar": "revenue", "priority": "high"})
    await skill.execute("add", {"title": "G2", "pillar": "self_improvement", "priority": "low"})
    r = await skill.execute("summary", {})
    assert r.success
    assert r.data["total"] == 2
    assert r.data["by_pillar"]["revenue"] == 1


@pytest.mark.asyncio
async def test_get_active_goals(skill):
    await skill.execute("add", {"title": "Active", "priority": "high"})
    r = await skill.execute("add", {"title": "Done"})
    await skill.execute("update", {"goal_id": r.data["goal_id"], "status": "completed"})
    active = skill.get_active_goals()
    assert len(active) == 1
    assert active[0]["title"] == "Active"


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


def test_check_credentials(skill):
    assert skill.check_credentials() is True
