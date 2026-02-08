#!/usr/bin/env python3
"""Tests for DecisionReplay + ConflictDetection integration into AutonomousLoopSkill LEARN phase."""

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.autonomous_loop import AutonomousLoopSkill, LoopPhase
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        s = AutonomousLoopSkill()
        yield s


def _make_context(replay_data=None, impact_data=None, conflict_data=None, replay_detail=None):
    """Create a mock context supporting replay, conflict, and distillation calls."""
    ctx = MagicMock()

    async def mock_call_skill(skill_id, action, params=None):
        if skill_id == "feedback_loop" and action == "analyze":
            return SkillResult(success=True, data={"adaptations": [], "patterns": []})
        elif skill_id == "learning_distillation" and action == "distill":
            return SkillResult(success=True, data={"rules_created": 1, "total_rules": 5, "sources_analyzed": []})
        elif skill_id == "learning_distillation" and action == "expire":
            return SkillResult(success=True, message="OK")
        elif skill_id == "learning_distillation" and action == "weaken":
            return SkillResult(success=True, message="Weakened", data={"rule_id": (params or {}).get("rule_id"), "old_confidence": 0.8, "new_confidence": 0.56})
        elif skill_id == "decision_replay" and action == "batch_replay":
            if replay_data:
                return SkillResult(success=True, message="Replayed", data=replay_data)
            return SkillResult(success=True, data={"regressions": 0, "improvements": 2, "reversals": 2, "reversal_rate": 0.1, "results": []})
        elif skill_id == "decision_replay" and action == "replay":
            if replay_detail:
                return SkillResult(success=True, data=replay_detail)
            return SkillResult(success=True, data={"replay": {"applicable_rules": [{"rule_id": "r1", "relevance_score": 0.8}]}})
        elif skill_id == "decision_replay" and action == "impact_report":
            if impact_data:
                return SkillResult(success=True, data=impact_data)
            return SkillResult(success=True, data={"learning_quality_score": 0.7})
        elif skill_id == "rule_conflict_detection" and action == "scan_and_resolve":
            if conflict_data:
                return SkillResult(success=True, data=conflict_data)
            return SkillResult(success=True, data={"resolved": 0, "conflicts_found": 0})
        return SkillResult(success=False, message=f"Unknown: {skill_id}:{action}")

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    return ctx


def _set_iteration(skill, count, config_overrides=None):
    """Set iteration count and optionally override config."""
    state = skill._load()
    state["stats"]["total_iterations"] = count
    if config_overrides:
        state["config"].update(config_overrides)
    skill._save(state)


@pytest.mark.asyncio
async def test_replay_config_defaults(skill):
    """New config options should have correct defaults."""
    state = skill._load()
    cfg = state["config"]
    assert cfg["replay_enabled"] is True
    assert cfg["replay_interval"] == 5
    assert cfg["replay_batch_size"] == 20
    assert cfg["auto_weaken_regressions"] is True
    assert cfg["conflict_scan_enabled"] is True
    assert cfg["conflict_scan_interval"] == 10


@pytest.mark.asyncio
async def test_replay_stats_defaults(skill):
    """New stats fields should start at zero."""
    state = skill._load()
    stats = state["stats"]
    assert stats["replay_runs"] == 0
    assert stats["replay_regressions_found"] == 0
    assert stats["rules_auto_weakened"] == 0
    assert stats["conflict_scans"] == 0
    assert stats["conflicts_resolved"] == 0


@pytest.mark.asyncio
async def test_replay_runs_on_interval(skill):
    """Replay should run when iteration count matches interval."""
    ctx = _make_context()
    skill.context = ctx
    _set_iteration(skill, 5)  # 5 % 5 == 0 -> should run

    learning = {"adaptations_count": 0, "learned_at": "now"}
    await skill._run_decision_replay(learning)

    assert learning.get("replay_ran") is True
    # Verify batch_replay was called
    calls = [c for c in ctx.call_skill.call_args_list if c[0][0] == "decision_replay" and c[0][1] == "batch_replay"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_replay_skips_off_interval(skill):
    """Replay should NOT run when iteration count doesn't match interval."""
    ctx = _make_context()
    skill.context = ctx
    _set_iteration(skill, 3)  # 3 % 5 != 0 -> should skip

    learning = {}
    await skill._run_decision_replay(learning)

    assert learning.get("replay_ran") is None
    calls = [c for c in ctx.call_skill.call_args_list if c[0][0] == "decision_replay"]
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_replay_disabled(skill):
    """Replay should not run when disabled via config."""
    ctx = _make_context()
    skill.context = ctx
    _set_iteration(skill, 5, {"replay_enabled": False})

    learning = {}
    await skill._run_decision_replay(learning)

    assert learning.get("replay_ran") is None


@pytest.mark.asyncio
async def test_auto_weaken_regressions(skill):
    """Rules causing regressions should be auto-weakened."""
    replay_data = {
        "regressions": 1, "improvements": 1, "reversals": 2,
        "reversal_rate": 0.1,
        "results": [
            {"decision_id": "d1", "regression": True, "original_choice": "X"},
            {"decision_id": "d2", "improvement": True, "original_choice": "Y"},
        ]
    }
    detail = {"replay": {"applicable_rules": [{"rule_id": "r1", "relevance_score": 0.8}, {"rule_id": "r2", "relevance_score": 0.1}]}}
    ctx = _make_context(replay_data=replay_data, replay_detail=detail)
    skill.context = ctx
    _set_iteration(skill, 5)

    learning = {}
    await skill._run_decision_replay(learning)

    assert learning["replay_ran"] is True
    assert learning["replay_regressions"] == 1
    # r1 should be weakened (relevance 0.8 > 0.3), r2 should NOT (relevance 0.1 < 0.3)
    assert learning["rules_weakened"] == 1
    weaken_calls = [c for c in ctx.call_skill.call_args_list if c[0][0] == "learning_distillation" and c[0][1] == "weaken"]
    assert len(weaken_calls) == 1
    assert weaken_calls[0][0][2]["rule_id"] == "r1"


@pytest.mark.asyncio
async def test_auto_weaken_disabled(skill):
    """Auto-weaken should not run when disabled."""
    replay_data = {
        "regressions": 1, "improvements": 0, "reversals": 1,
        "reversal_rate": 0.05,
        "results": [{"decision_id": "d1", "regression": True}]
    }
    ctx = _make_context(replay_data=replay_data)
    skill.context = ctx
    _set_iteration(skill, 5, {"auto_weaken_regressions": False})

    learning = {}
    await skill._run_decision_replay(learning)

    assert learning["replay_ran"] is True
    weaken_calls = [c for c in ctx.call_skill.call_args_list if c[0][0] == "learning_distillation" and c[0][1] == "weaken"]
    assert len(weaken_calls) == 0


@pytest.mark.asyncio
async def test_conflict_scan_runs_on_interval(skill):
    """Conflict scan should run at configured interval."""
    conflict_data = {"resolved": 2, "conflicts_found": 3}
    ctx = _make_context(conflict_data=conflict_data)
    skill.context = ctx
    _set_iteration(skill, 10)  # 10 % 10 == 0

    learning = {}
    await skill._run_conflict_scan(learning)

    assert learning["conflict_scan_ran"] is True
    assert learning["conflicts_resolved"] == 2
    state = skill._load()
    assert state["stats"]["conflict_scans"] == 1
    assert state["stats"]["conflicts_resolved"] == 2


@pytest.mark.asyncio
async def test_conflict_scan_skips_off_interval(skill):
    """Conflict scan should skip when not on interval."""
    ctx = _make_context()
    skill.context = ctx
    _set_iteration(skill, 7)  # 7 % 10 != 0

    learning = {}
    await skill._run_conflict_scan(learning)

    assert learning.get("conflict_scan_ran") is None


@pytest.mark.asyncio
async def test_conflict_scan_disabled(skill):
    """Conflict scan should not run when disabled."""
    ctx = _make_context()
    skill.context = ctx
    _set_iteration(skill, 10, {"conflict_scan_enabled": False})

    learning = {}
    await skill._run_conflict_scan(learning)

    assert learning.get("conflict_scan_ran") is None


@pytest.mark.asyncio
async def test_replay_updates_stats(skill):
    """Replay should update loop stats."""
    replay_data = {"regressions": 3, "improvements": 5, "reversals": 8, "reversal_rate": 0.4, "results": []}
    ctx = _make_context(replay_data=replay_data)
    skill.context = ctx
    _set_iteration(skill, 5)

    learning = {}
    await skill._run_decision_replay(learning)

    state = skill._load()
    assert state["stats"]["replay_runs"] == 1
    assert state["stats"]["replay_regressions_found"] == 3


@pytest.mark.asyncio
async def test_configure_new_keys(skill):
    """New config keys should be settable via configure action."""
    result = await skill.execute("configure", {
        "replay_enabled": False,
        "replay_interval": 10,
        "replay_batch_size": 50,
        "auto_weaken_regressions": False,
        "conflict_scan_enabled": False,
        "conflict_scan_interval": 20,
    })
    assert result.success
    state = skill._load()
    cfg = state["config"]
    assert cfg["replay_enabled"] is False
    assert cfg["replay_interval"] == 10
    assert cfg["replay_batch_size"] == 50
    assert cfg["auto_weaken_regressions"] is False
    assert cfg["conflict_scan_enabled"] is False
    assert cfg["conflict_scan_interval"] == 20


@pytest.mark.asyncio
async def test_run_learning_calls_replay_and_conflict(skill):
    """_run_learning should call replay and conflict scan."""
    ctx = _make_context()
    skill.context = ctx
    # Set iterations to match both intervals (LCM of 3, 5, 10 = 30)
    # distillation at 3, replay at 5, conflict at 10 -> iteration 30 hits all
    _set_iteration(skill, 30, {"distillation_interval": 3, "replay_interval": 5, "conflict_scan_interval": 10})

    measurement = {"success": True}
    learning = await skill._run_learning(measurement)

    # Verify all three were called
    skill_actions = [(c[0][0], c[0][1]) for c in ctx.call_skill.call_args_list]
    assert ("feedback_loop", "analyze") in skill_actions
    assert ("learning_distillation", "distill") in skill_actions
    assert ("decision_replay", "batch_replay") in skill_actions
    assert ("rule_conflict_detection", "scan_and_resolve") in skill_actions


@pytest.mark.asyncio
async def test_replay_no_context(skill):
    """Replay should gracefully do nothing without context."""
    skill.context = None
    learning = {}
    await skill._run_decision_replay(learning)
    assert learning.get("replay_ran") is None


@pytest.mark.asyncio
async def test_conflict_no_context(skill):
    """Conflict scan should gracefully do nothing without context."""
    skill.context = None
    learning = {}
    await skill._run_conflict_scan(learning)
    assert learning.get("conflict_scan_ran") is None
