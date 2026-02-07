#!/usr/bin/env python3
"""Tests for ContextSynthesisSkill."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from singularity.skills.context_synthesis import (
    ContextSynthesisSkill,
    CONTEXT_FILE,
    GOALS_FILE,
    STRATEGY_FILE,
    FEEDBACK_FILE,
    PERFORMANCE_FILE,
    EXPERIMENT_FILE,
    DATA_DIR,
)


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data directory."""
    with patch("singularity.skills.context_synthesis.DATA_DIR", tmp_path), \
         patch("singularity.skills.context_synthesis.CONTEXT_FILE", tmp_path / "context_synthesis.json"), \
         patch("singularity.skills.context_synthesis.GOALS_FILE", tmp_path / "goals.json"), \
         patch("singularity.skills.context_synthesis.STRATEGY_FILE", tmp_path / "strategy.json"), \
         patch("singularity.skills.context_synthesis.FEEDBACK_FILE", tmp_path / "feedback_loop.json"), \
         patch("singularity.skills.context_synthesis.PERFORMANCE_FILE", tmp_path / "performance.json"), \
         patch("singularity.skills.context_synthesis.EXPERIMENT_FILE", tmp_path / "experiments.json"):
        s = ContextSynthesisSkill()
        s._tmp_path = tmp_path  # Store for helpers
        yield s


def _write_json(skill, filename, data):
    """Helper to write test data files."""
    path = skill._tmp_path / filename
    path.write_text(json.dumps(data))


@pytest.mark.asyncio
async def test_briefing_empty_sources(skill):
    """Briefing with no data sources returns empty message."""
    result = await skill.execute("briefing", {})
    assert result.success
    assert "No operational context" in result.data["briefing"]


@pytest.mark.asyncio
async def test_briefing_with_goals(skill):
    """Briefing includes active goals."""
    _write_json(skill, "goals.json", {
        "goals": [
            {"title": "Build API", "status": "active", "priority": "high", "pillar": "revenue", "milestones": [
                {"name": "Design", "completed": True},
                {"name": "Implement", "completed": False},
            ]},
            {"title": "Add tests", "status": "active", "priority": "low", "pillar": "self_improvement", "milestones": []},
            {"title": "Old goal", "status": "completed", "priority": "medium", "pillar": "other", "milestones": []},
        ],
        "completed_goals": [{"title": "Old goal"}],
    })
    result = await skill.execute("briefing", {"include_performance": False, "include_strategy": False})
    assert result.success
    assert "Build API" in result.data["briefing"]
    assert "1/2" in result.data["briefing"]  # milestone progress
    assert "goals" in result.data["sources"]


@pytest.mark.asyncio
async def test_briefing_with_strategy(skill):
    """Briefing includes strategic context."""
    _write_json(skill, "strategy.json", {
        "assessments": [{
            "scores": {"self_improvement": 60, "revenue": 20, "replication": 40, "goal_setting": 50},
            "diagnosis": "Revenue is the weakest pillar, focus there.",
        }],
        "journal": [{"entry": "Need to build payment processing next."}],
    })
    result = await skill.execute("briefing", {"include_performance": False, "include_goals": False})
    assert result.success
    assert "revenue" in result.data["briefing"].lower()
    assert "Weakest" in result.data["briefing"]


@pytest.mark.asyncio
async def test_briefing_with_performance(skill):
    """Briefing includes recent performance."""
    now = datetime.now()
    _write_json(skill, "performance.json", {
        "records": [
            {"skill_id": "shell", "action": "run", "success": True, "cost_usd": 0.001, "timestamp": now.isoformat()},
            {"skill_id": "shell", "action": "run", "success": True, "cost_usd": 0.001, "timestamp": now.isoformat()},
            {"skill_id": "github", "action": "pr", "success": False, "cost_usd": 0.01, "timestamp": now.isoformat()},
        ],
    })
    result = await skill.execute("briefing", {"include_goals": False, "include_strategy": False})
    assert result.success
    assert "shell:run" in result.data["briefing"]
    assert "performance" in result.data["sources"]


@pytest.mark.asyncio
async def test_briefing_with_adaptations(skill):
    """Briefing includes active adaptations."""
    _write_json(skill, "feedback_loop.json", {
        "adaptations": [
            {"id": "a1", "applied": True, "reverted": False, "action": "Prefer shell over github for deploys", "type": "avoid_failing_skill"},
            {"id": "a2", "applied": False, "action": "Batch API calls"},
        ],
        "adaptation_outcomes": [
            {"verdict": "improved"},
            {"verdict": "degraded"},
        ],
    })
    result = await skill.execute("briefing", {"include_performance": False, "include_goals": False, "include_strategy": False})
    assert result.success
    assert "Prefer shell" in result.data["briefing"]
    assert "1 pending" in result.data["briefing"]


@pytest.mark.asyncio
async def test_briefing_with_experiments(skill):
    """Briefing includes running experiments."""
    _write_json(skill, "experiments.json", {
        "experiments": [
            {"name": "Model comparison", "status": "running", "hypothesis": "GPT-4o is cheaper than Claude", "total_trials": 10},
            {"name": "Done experiment", "status": "concluded", "hypothesis": "Old"},
        ],
    })
    result = await skill.execute("briefing", {"include_performance": False, "include_goals": False, "include_strategy": False})
    assert result.success
    assert "Model comparison" in result.data["briefing"]


@pytest.mark.asyncio
async def test_focus_and_briefing(skill):
    """Focus area appears in briefing."""
    result = await skill.execute("focus", {"area": "revenue"})
    assert result.success
    assert result.data["focus"] == "revenue"

    # Now get briefing - should show focus
    _write_json(skill, "goals.json", {"goals": [{"title": "test", "status": "active", "priority": "medium", "pillar": "other", "milestones": []}], "completed_goals": []})
    result = await skill.execute("briefing", {})
    assert result.success
    assert "FOCUS: REVENUE" in result.data["briefing"]


@pytest.mark.asyncio
async def test_focus_clear(skill):
    """Focus area can be cleared."""
    await skill.execute("focus", {"area": "revenue"})
    result = await skill.execute("focus", {"area": "clear"})
    assert result.success
    assert "cleared" in result.message.lower()


@pytest.mark.asyncio
async def test_focus_invalid(skill):
    """Invalid focus area returns error."""
    result = await skill.execute("focus", {"area": "invalid"})
    assert not result.success


@pytest.mark.asyncio
async def test_snapshot_and_diff(skill):
    """Snapshot captures state and diff detects changes."""
    _write_json(skill, "goals.json", {"goals": [{"title": "G1", "status": "active", "priority": "high", "pillar": "revenue", "milestones": []}], "completed_goals": []})
    snap = await skill.execute("snapshot", {"label": "before_change"})
    assert snap.success
    snap_id = snap.data["snapshot_id"]

    # Modify goals
    _write_json(skill, "goals.json", {"goals": [
        {"title": "G1", "status": "active", "priority": "high", "pillar": "revenue", "milestones": []},
        {"title": "G2", "status": "active", "priority": "medium", "pillar": "self_improvement", "milestones": []},
    ], "completed_goals": []})

    diff = await skill.execute("diff", {"snapshot_id": snap_id})
    assert diff.success
    assert any("Goals" in c for c in diff.data["changes"])


@pytest.mark.asyncio
async def test_diff_no_snapshots(skill):
    """Diff fails gracefully with no snapshots."""
    result = await skill.execute("diff", {})
    assert not result.success


@pytest.mark.asyncio
async def test_sources(skill):
    """Sources lists available data files."""
    _write_json(skill, "goals.json", {"goals": [], "completed_goals": [], "last_updated": "2024-01-01"})
    result = await skill.execute("sources", {})
    assert result.success
    assert result.data["sources"]["goals"]["exists"]
    assert not result.data["sources"]["performance"]["exists"]


@pytest.mark.asyncio
async def test_history_empty(skill):
    """History with no briefings."""
    result = await skill.execute("history", {})
    assert result.success
    assert len(result.data["history"]) == 0


@pytest.mark.asyncio
async def test_history_after_briefings(skill):
    """History shows past briefings."""
    await skill.execute("briefing", {})
    await skill.execute("briefing", {})
    result = await skill.execute("history", {"count": 5})
    assert result.success
    assert len(result.data["history"]) == 2


@pytest.mark.asyncio
async def test_inject_to_prompt(skill):
    """Briefing can be injected into prompt."""
    prompt_additions = []
    skill.set_cognition_hooks(append_prompt=lambda text: prompt_additions.append(text))

    _write_json(skill, "goals.json", {"goals": [{"title": "Test Goal", "status": "active", "priority": "high", "pillar": "revenue", "milestones": []}], "completed_goals": []})
    result = await skill.execute("briefing", {"inject_to_prompt": True})
    assert result.success
    assert result.data["injected"]
    assert len(prompt_additions) == 1
    assert "CONTEXT BRIEFING" in prompt_additions[0]


@pytest.mark.asyncio
async def test_manifest(skill):
    """Manifest has correct structure."""
    m = skill.manifest
    assert m.skill_id == "context"
    assert m.name == "Context Synthesis"
    assert len(m.actions) == 6
    action_names = {a.name for a in m.actions}
    assert action_names == {"briefing", "snapshot", "diff", "sources", "focus", "history"}


@pytest.mark.asyncio
async def test_unknown_action(skill):
    """Unknown action returns error."""
    result = await skill.execute("nonexistent", {})
    assert not result.success
