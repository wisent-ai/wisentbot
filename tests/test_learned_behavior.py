"""Tests for LearnedBehaviorSkill - persistent cross-session behavioral rules."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from singularity.skills.learned_behavior import LearnedBehaviorSkill, BEHAVIOR_FILE


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    """Use temp files for all tests."""
    bf = tmp_path / "learned_behaviors.json"
    monkeypatch.setattr("singularity.skills.learned_behavior.BEHAVIOR_FILE", bf)
    return bf


@pytest.fixture
def skill():
    s = LearnedBehaviorSkill()
    return s


@pytest.fixture
def skill_with_hooks():
    s = LearnedBehaviorSkill()
    prompt_parts = []
    s.set_cognition_hooks(
        append_prompt=lambda text: prompt_parts.append(text),
        get_prompt=lambda: "base prompt",
    )
    return s, prompt_parts


@pytest.mark.asyncio
async def test_add_rule(skill):
    result = await skill.execute("add_rule", {
        "rule_type": "prompt_addition",
        "content": "Always check balance before expensive operations",
        "source": "manual",
        "confidence": 0.8,
    })
    assert result.success
    assert "rule_id" in result.data
    assert result.data["rule"]["rule_type"] == "prompt_addition"
    assert result.data["rule"]["confidence"] == 0.8


@pytest.mark.asyncio
async def test_add_rule_invalid_type(skill):
    result = await skill.execute("add_rule", {
        "rule_type": "invalid_type",
        "content": "test",
    })
    assert not result.success
    assert "Invalid rule_type" in result.message


@pytest.mark.asyncio
async def test_add_duplicate_rule(skill):
    await skill.execute("add_rule", {"rule_type": "strategy", "content": "be frugal"})
    result = await skill.execute("add_rule", {"rule_type": "strategy", "content": "be frugal"})
    assert not result.success
    assert "Duplicate" in result.message


@pytest.mark.asyncio
async def test_list_rules(skill):
    await skill.execute("add_rule", {"rule_type": "strategy", "content": "rule1"})
    await skill.execute("add_rule", {"rule_type": "heuristic", "content": "rule2"})
    result = await skill.execute("list_rules", {})
    assert result.success
    assert result.data["count"] == 2


@pytest.mark.asyncio
async def test_list_rules_filter_by_type(skill):
    await skill.execute("add_rule", {"rule_type": "strategy", "content": "s1"})
    await skill.execute("add_rule", {"rule_type": "heuristic", "content": "h1"})
    result = await skill.execute("list_rules", {"rule_type": "strategy"})
    assert result.success
    assert result.data["count"] == 1


@pytest.mark.asyncio
async def test_load_rules_applies_prompts(skill_with_hooks):
    skill, prompt_parts = skill_with_hooks
    await skill.execute("add_rule", {"rule_type": "prompt_addition", "content": "be careful with money"})
    await skill.execute("add_rule", {"rule_type": "strategy", "content": "focus on revenue"})
    result = await skill.execute("load_rules", {})
    assert result.success
    assert result.data["rules_loaded"] == 2
    assert result.data["prompt_additions_applied"] == 1
    assert len(prompt_parts) == 1
    assert "be careful with money" in prompt_parts[0]


@pytest.mark.asyncio
async def test_record_outcome_updates_effectiveness(skill):
    r = await skill.execute("add_rule", {"rule_type": "strategy", "content": "test rule"})
    rule_id = r.data["rule_id"]
    # Record successes
    for _ in range(5):
        await skill.execute("record_outcome", {"rule_id": rule_id, "success": True})
    result = await skill.execute("list_rules", {})
    rule = result.data["rules"][0]
    assert rule["effectiveness"] > 0.7
    assert rule["success_count"] == 5


@pytest.mark.asyncio
async def test_prune_removes_bad_rules(skill):
    r = await skill.execute("add_rule", {"rule_type": "strategy", "content": "bad rule"})
    rule_id = r.data["rule_id"]
    # Simulate 3 sessions and many failures
    data = skill._load()
    for rule in data["rules"]:
        if rule["rule_id"] == rule_id:
            rule["sessions_active"] = 5
            rule["failure_count"] = 8
            rule["success_count"] = 1
    skill._save(data)
    result = await skill.execute("prune", {"min_sessions": 3, "max_failure_rate": 0.7})
    assert result.success
    assert len(result.data["pruned"]) == 1


@pytest.mark.asyncio
async def test_deactivate_rule(skill):
    r = await skill.execute("add_rule", {"rule_type": "heuristic", "content": "test"})
    rule_id = r.data["rule_id"]
    result = await skill.execute("deactivate", {"rule_id": rule_id})
    assert result.success
    # Verify it's not in active list
    list_result = await skill.execute("list_rules", {"active_only": True})
    assert list_result.data["count"] == 0


@pytest.mark.asyncio
async def test_get_context_prompt(skill):
    await skill.execute("add_rule", {"rule_type": "prompt_addition", "content": "insight1"})
    await skill.execute("add_rule", {"rule_type": "avoidance_rule", "content": "avoid X"})
    await skill.execute("add_rule", {"rule_type": "strategy", "content": "do Y"})
    result = await skill.execute("get_context_prompt", {})
    assert result.success
    assert "insight1" in result.data["prompt"]
    assert "AVOID: avoid X" in result.data["prompt"]
    assert result.data["rule_count"] == 3


@pytest.mark.asyncio
async def test_import_from_feedback(skill, clean_data, tmp_path):
    """Test importing adaptations from FeedbackLoop."""
    fb_file = tmp_path / "feedback_loop.json"
    fb_data = {
        "adaptations": [
            {"id": "a1", "type": "general", "prompt_addition": "learned thing", "applied": True, "evaluated": True, "improved": True},
            {"id": "a2", "type": "avoid_skill", "description": "avoid broken skill", "applied": True},
            {"id": "a3", "type": "general", "prompt_addition": "not applied", "applied": False},
        ],
        "reviews": [],
        "adaptation_outcomes": [],
    }
    with open(fb_file, "w") as f:
        json.dump(fb_data, f)

    # Monkeypatch the feedback file path inside the skill
    import singularity.skills.learned_behavior as lb_mod
    original = Path(lb_mod.__file__).parent.parent / "data" / "feedback_loop.json"
    # Write to where the skill actually looks
    (Path(lb_mod.__file__).parent.parent / "data").mkdir(parents=True, exist_ok=True)
    with open(original, "w") as f:
        json.dump(fb_data, f)

    try:
        result = await skill.execute("import_from_feedback", {})
        assert result.success
        assert result.data["imported"] == 2
        assert result.data["skipped"] >= 1
    finally:
        if original.exists():
            original.unlink()


@pytest.mark.asyncio
async def test_min_confidence_filter(skill_with_hooks):
    skill, prompt_parts = skill_with_hooks
    await skill.execute("add_rule", {"rule_type": "prompt_addition", "content": "low conf", "confidence": 0.1})
    await skill.execute("add_rule", {"rule_type": "prompt_addition", "content": "high conf", "confidence": 0.8})
    result = await skill.execute("load_rules", {"min_confidence": 0.5})
    assert result.success
    assert result.data["rules_loaded"] == 1
