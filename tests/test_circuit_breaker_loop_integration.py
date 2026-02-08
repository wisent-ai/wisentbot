"""Tests for circuit breaker integration in AutonomousLoopSkill."""

import pytest
import time
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.autonomous_loop import AutonomousLoopSkill, LoopPhase
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    """Create an AutonomousLoopSkill with a temporary data path."""
    test_file = tmp_path / "autonomous_loop.json"
    with patch("singularity.skills.autonomous_loop.LOOP_STATE_FILE", test_file):
        s = AutonomousLoopSkill()
        yield s


def make_context(circuit_check_result=None, skill_result=None, record_result=None):
    """Create a mock context with configurable circuit breaker behavior."""
    ctx = MagicMock()

    async def mock_call_skill(skill_id, action, params=None):
        if skill_id == "circuit_breaker" and action == "check":
            if circuit_check_result is not None:
                return circuit_check_result
            return SkillResult(
                success=True,
                message="ALLOW",
                data={"allowed": True, "reason": "circuit_closed"},
            )
        if skill_id == "circuit_breaker" and action == "record":
            if record_result is not None:
                return record_result
            return SkillResult(success=True, message="Recorded")
        if skill_id == "outcome_tracker":
            return SkillResult(success=True, message="Logged")
        if skill_id == "feedback_loop":
            return SkillResult(success=True, message="Analyzed", data={
                "adaptations": [], "patterns": [],
            })
        # Default skill response
        if skill_result is not None:
            return skill_result
        return SkillResult(success=True, message="Executed", data={})

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    return ctx


@pytest.mark.asyncio
async def test_circuit_breaker_allows_execution(skill):
    """When circuit is closed, skill should execute normally."""
    ctx = make_context()
    skill.set_context(ctx)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "code_review", "action": "review",
             "params": {"code": "x=1"}, "description": "Review code"},
        ]
    }
    results = await skill._run_actions(plan)

    assert results["success"] is True
    assert results["steps_succeeded"] == 1
    assert results["steps_denied"] == 0

    # Verify circuit_breaker.check was called
    calls = [c for c in ctx.call_skill.call_args_list
             if c[0][0] == "circuit_breaker" and c[0][1] == "check"]
    assert len(calls) == 1
    assert calls[0][0][2]["skill_id"] == "code_review"

    # Verify circuit_breaker.record was called with success
    record_calls = [c for c in ctx.call_skill.call_args_list
                    if c[0][0] == "circuit_breaker" and c[0][1] == "record"]
    assert len(record_calls) == 1
    assert record_calls[0][0][2]["success"] is True


@pytest.mark.asyncio
async def test_circuit_breaker_denies_execution(skill):
    """When circuit is open, skill should be skipped."""
    deny_result = SkillResult(
        success=True,
        message="DENY: code_review circuit is open",
        data={"allowed": False, "reason": "circuit_open"},
    )
    ctx = make_context(circuit_check_result=deny_result)
    skill.set_context(ctx)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "code_review", "action": "review",
             "params": {}, "description": "Review code"},
        ]
    }
    results = await skill._run_actions(plan)

    assert results["steps_denied"] == 1
    assert results["steps_succeeded"] == 0
    assert results["step_results"][0]["success"] is False
    assert "DENIED by circuit breaker" in results["step_results"][0]["message"]
    assert plan["steps"][0]["status"] == "circuit_denied"


@pytest.mark.asyncio
async def test_circuit_breaker_records_failure(skill):
    """When a skill fails, the failure is recorded to circuit breaker."""
    fail_result = SkillResult(success=False, message="API timeout")
    ctx = make_context(skill_result=fail_result)
    skill.set_context(ctx)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "email", "action": "send",
             "params": {}, "description": "Send email"},
        ]
    }
    results = await skill._run_actions(plan)

    record_calls = [c for c in ctx.call_skill.call_args_list
                    if c[0][0] == "circuit_breaker" and c[0][1] == "record"]
    assert len(record_calls) == 1
    assert record_calls[0][0][2]["success"] is False
    assert "API timeout" in record_calls[0][0][2]["error"]


@pytest.mark.asyncio
async def test_circuit_breaker_skips_internal_skills(skill):
    """Internal skills (autonomous_loop, circuit_breaker, etc.) should bypass circuit check."""
    ctx = make_context()
    skill.set_context(ctx)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "outcome_tracker", "action": "log",
             "params": {}, "description": "Log outcome"},
        ]
    }
    results = await skill._run_actions(plan)

    # Circuit breaker check should NOT be called for internal skills
    cb_check_calls = [c for c in ctx.call_skill.call_args_list
                      if c[0][0] == "circuit_breaker" and c[0][1] == "check"]
    assert len(cb_check_calls) == 0


@pytest.mark.asyncio
async def test_circuit_breaker_disabled(skill):
    """When circuit_breaker_enabled=False, should skip all CB checks."""
    ctx = make_context()
    skill.set_context(ctx)

    # Disable circuit breaker
    state = skill._load()
    state["config"]["circuit_breaker_enabled"] = False
    skill._save(state)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "code_review", "action": "review",
             "params": {}, "description": "Review code"},
        ]
    }
    results = await skill._run_actions(plan)

    assert results["success"] is True
    # No circuit breaker calls
    cb_calls = [c for c in ctx.call_skill.call_args_list
                if c[0][0] == "circuit_breaker"]
    assert len(cb_calls) == 0


@pytest.mark.asyncio
async def test_circuit_breaker_unavailable_allows_execution(skill):
    """If circuit_breaker skill isn't registered, execution should proceed."""
    ctx = MagicMock()

    async def mock_call_skill(skill_id, action, params=None):
        if skill_id == "circuit_breaker":
            return SkillResult(success=False, message="Skill 'circuit_breaker' not found")
        return SkillResult(success=True, message="Done")

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    skill.set_context(ctx)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "code_review", "action": "review",
             "params": {}, "description": "Review"},
        ]
    }
    results = await skill._run_actions(plan)
    assert results["success"] is True


@pytest.mark.asyncio
async def test_circuit_breaker_exception_allows_execution(skill):
    """If circuit_breaker raises an exception, execution should proceed."""
    ctx = MagicMock()
    call_count = {"n": 0}

    async def mock_call_skill(skill_id, action, params=None):
        call_count["n"] += 1
        if skill_id == "circuit_breaker" and action == "check":
            raise ConnectionError("Circuit breaker unavailable")
        if skill_id == "circuit_breaker" and action == "record":
            raise ConnectionError("Circuit breaker unavailable")
        return SkillResult(success=True, message="Done")

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    skill.set_context(ctx)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "code_review", "action": "review",
             "params": {}, "description": "Review"},
        ]
    }
    results = await skill._run_actions(plan)
    assert results["success"] is True


@pytest.mark.asyncio
async def test_circuit_breaker_tracks_duration(skill):
    """Circuit breaker recordings should include execution duration."""
    ctx = make_context()
    skill.set_context(ctx)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "code_review", "action": "review",
             "params": {}, "description": "Review"},
        ]
    }
    results = await skill._run_actions(plan)

    record_calls = [c for c in ctx.call_skill.call_args_list
                    if c[0][0] == "circuit_breaker" and c[0][1] == "record"]
    assert len(record_calls) == 1
    duration = record_calls[0][0][2]["duration_ms"]
    assert duration >= 0  # Should have a non-negative duration


@pytest.mark.asyncio
async def test_mixed_steps_with_circuit_breaker(skill):
    """Mix of allowed, denied, and recommendation steps."""
    call_idx = {"n": 0}
    ctx = MagicMock()

    async def mock_call_skill(skill_id, action, params=None):
        if skill_id == "circuit_breaker" and action == "check":
            target = params.get("skill_id", "")
            if target == "broken_skill":
                return SkillResult(
                    success=True, message="DENY",
                    data={"allowed": False, "reason": "circuit_open"},
                )
            return SkillResult(
                success=True, message="ALLOW",
                data={"allowed": True, "reason": "circuit_closed"},
            )
        if skill_id == "circuit_breaker" and action == "record":
            return SkillResult(success=True, message="Recorded")
        return SkillResult(success=True, message="Done")

    ctx.call_skill = AsyncMock(side_effect=mock_call_skill)
    skill.set_context(ctx)

    plan = {
        "steps": [
            {"step": 1, "skill_id": "code_review", "action": "review",
             "params": {}, "description": "Review code"},
            {"step": 2, "skill_id": "broken_skill", "action": "run",
             "params": {}, "description": "Broken step"},
            {"step": 3, "description": "Manual recommendation"},
        ]
    }
    results = await skill._run_actions(plan)

    assert results["steps_executed"] == 3
    assert results["steps_succeeded"] == 2  # code_review + recommendation
    assert results["steps_denied"] == 1     # broken_skill
    assert results["success"] is True       # At least one succeeded
