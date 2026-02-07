"""Tests for the skills base module - Skill, SkillRegistry, and data classes."""

import pytest

from singularity.skills.base import (
    SkillAction,
    SkillManifest,
    SkillResult,
)

from .conftest import DummySkill, NoCredSkill

# ── SkillResult ─────────────────────────────────────────────────────────


class TestSkillResult:
    def test_success_result(self):
        result = SkillResult(success=True, message="OK", data={"key": "value"})
        assert result.success is True
        assert result.message == "OK"
        assert result.data == {"key": "value"}
        assert result.cost == 0
        assert result.revenue == 0

    def test_failure_result(self):
        result = SkillResult(success=False, message="Something went wrong")
        assert result.success is False
        assert result.data == {}

    def test_result_with_cost_and_revenue(self):
        result = SkillResult(
            success=True,
            message="Created product",
            cost=0.50,
            revenue=10.0,
            asset_created={"type": "product", "id": "123"},
        )
        assert result.cost == 0.50
        assert result.revenue == 10.0
        assert result.asset_created["type"] == "product"


# ── SkillAction ─────────────────────────────────────────────────────────


class TestSkillAction:
    def test_basic_action(self):
        action = SkillAction(
            name="greet",
            description="Say hello",
            parameters={"name": {"type": "string", "required": True}},
        )
        assert action.name == "greet"
        assert action.estimated_cost == 0
        assert action.estimated_duration_seconds == 10
        assert action.success_probability == 0.8

    def test_action_with_custom_values(self):
        action = SkillAction(
            name="deploy",
            description="Deploy to production",
            parameters={"project": {"type": "string"}},
            estimated_cost=0.50,
            estimated_duration_seconds=120,
            success_probability=0.7,
        )
        assert action.estimated_cost == 0.50
        assert action.estimated_duration_seconds == 120
        assert action.success_probability == 0.7


# ── SkillManifest ───────────────────────────────────────────────────────


class TestSkillManifest:
    def test_manifest(self):
        manifest = SkillManifest(
            skill_id="test",
            name="Test Skill",
            version="1.0.0",
            category="test",
            description="A test skill",
            actions=[],
            required_credentials=["API_KEY"],
        )
        assert manifest.skill_id == "test"
        assert manifest.install_cost == 0
        assert manifest.author == "system"


# ── Skill (via DummySkill) ──────────────────────────────────────────────


class TestSkill:
    def test_check_credentials_with_valid_creds(self, dummy_skill):
        assert dummy_skill.check_credentials() is True

    def test_check_credentials_without_creds(self):
        skill = DummySkill(credentials={})
        assert skill.check_credentials() is False

    def test_check_credentials_with_empty_value(self):
        skill = DummySkill(credentials={"DUMMY_API_KEY": ""})
        assert skill.check_credentials() is False

    def test_get_missing_credentials(self):
        skill = DummySkill(credentials={})
        missing = skill.get_missing_credentials()
        assert "DUMMY_API_KEY" in missing

    def test_get_missing_credentials_when_all_present(self, dummy_skill):
        missing = dummy_skill.get_missing_credentials()
        assert len(missing) == 0

    def test_no_credentials_needed(self, nocred_skill):
        assert nocred_skill.check_credentials() is True
        assert nocred_skill.get_missing_credentials() == []

    def test_get_actions(self, dummy_skill):
        actions = dummy_skill.get_actions()
        assert len(actions) == 2
        names = [a.name for a in actions]
        assert "greet" in names
        assert "fail" in names

    def test_get_action_by_name(self, dummy_skill):
        action = dummy_skill.get_action("greet")
        assert action is not None
        assert action.name == "greet"

    def test_get_action_nonexistent(self, dummy_skill):
        action = dummy_skill.get_action("nonexistent")
        assert action is None

    def test_estimate_cost(self, dummy_skill):
        cost = dummy_skill.estimate_cost("greet", {})
        assert cost == 0.01

    def test_estimate_cost_unknown_action(self, dummy_skill):
        cost = dummy_skill.estimate_cost("unknown", {})
        assert cost == 0

    @pytest.mark.asyncio
    async def test_execute_success(self, dummy_skill):
        result = await dummy_skill.execute("greet", {"name": "Alice"})
        assert result.success is True
        assert "Hello, Alice!" in result.message
        assert result.data["greeting"] == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_execute_failure(self, dummy_skill):
        result = await dummy_skill.execute("fail", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, dummy_skill):
        result = await dummy_skill.execute("unknown_action", {})
        assert result.success is False
        assert "Unknown action" in result.message

    @pytest.mark.asyncio
    async def test_initialize_with_valid_creds(self, dummy_skill):
        result = await dummy_skill.initialize()
        assert result is True
        assert dummy_skill.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_without_creds(self):
        skill = DummySkill(credentials={})
        result = await skill.initialize()
        assert result is False
        assert skill.initialized is False

    def test_record_usage(self, dummy_skill):
        dummy_skill.record_usage(cost=0.01, revenue=0.05)
        dummy_skill.record_usage(cost=0.02, revenue=0.10)
        stats = dummy_skill.stats
        assert stats["usage_count"] == 2
        assert abs(stats["total_cost"] - 0.03) < 1e-9
        assert abs(stats["total_revenue"] - 0.15) < 1e-9
        assert abs(stats["profit"] - 0.12) < 1e-9

    def test_to_dict(self, dummy_skill):
        d = dummy_skill.to_dict()
        assert d["skill_id"] == "dummy"
        assert d["name"] == "Dummy Skill"
        assert d["category"] == "test"
        assert len(d["actions"]) == 2
        assert "stats" in d


# ── SkillRegistry ───────────────────────────────────────────────────────


class TestSkillRegistry:
    def test_install_skill(self, skill_registry, dummy_credentials):
        skill_registry.set_credentials(dummy_credentials)
        result = skill_registry.install(DummySkill)
        assert result is True
        assert "dummy" in skill_registry.skills

    def test_get_skill(self, skill_registry, dummy_credentials):
        skill_registry.set_credentials(dummy_credentials)
        skill_registry.install(DummySkill)
        skill = skill_registry.get("dummy")
        assert skill is not None
        assert skill.manifest.skill_id == "dummy"

    def test_get_nonexistent_skill(self, skill_registry):
        skill = skill_registry.get("nonexistent")
        assert skill is None

    def test_uninstall_skill(self, skill_registry, dummy_credentials):
        skill_registry.set_credentials(dummy_credentials)
        skill_registry.install(DummySkill)
        assert skill_registry.uninstall("dummy") is True
        assert skill_registry.get("dummy") is None

    def test_uninstall_nonexistent(self, skill_registry):
        assert skill_registry.uninstall("nonexistent") is False

    def test_list_skills(self, skill_registry, dummy_credentials):
        skill_registry.set_credentials(dummy_credentials)
        skill_registry.install(DummySkill)
        skill_registry.install(NoCredSkill)
        skills_list = skill_registry.list_skills()
        assert len(skills_list) == 2

    def test_list_all_actions(self, skill_registry, dummy_credentials):
        skill_registry.set_credentials(dummy_credentials)
        skill_registry.install(DummySkill)
        actions = skill_registry.list_all_actions()
        assert len(actions) == 2
        action_names = [a["action"] for a in actions]
        assert "greet" in action_names
        assert "fail" in action_names

    @pytest.mark.asyncio
    async def test_execute_via_registry(self, skill_registry, dummy_credentials):
        skill_registry.set_credentials(dummy_credentials)
        skill_registry.install(DummySkill)
        result = await skill_registry.execute("dummy", "greet", {"name": "World"})
        assert result.success is True
        assert "Hello, World!" in result.message

    @pytest.mark.asyncio
    async def test_execute_nonexistent_skill(self, skill_registry):
        result = await skill_registry.execute("missing", "action", {})
        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_uninitialized_skill_without_creds(self, skill_registry):
        skill_registry.install(DummySkill)  # No credentials set
        result = await skill_registry.execute("dummy", "greet", {"name": "Test"})
        assert result.success is False
        assert "credentials" in result.message.lower()

    def test_set_credentials_updates_existing_skills(self, skill_registry):
        skill_registry.install(DummySkill)
        skill = skill_registry.get("dummy")
        assert skill.check_credentials() is False

        skill_registry.set_credentials({"DUMMY_API_KEY": "new-key"})
        assert skill.check_credentials() is True

    def test_get_skills_for_llm(self, skill_registry, dummy_credentials):
        skill_registry.set_credentials(dummy_credentials)
        skill_registry.install(DummySkill)
        text = skill_registry.get_skills_for_llm()
        assert "INSTALLED SKILLS:" in text
        assert "dummy" in text
        assert "greet" in text

    def test_install_multiple_skills(self, skill_registry, dummy_credentials):
        skill_registry.set_credentials(dummy_credentials)
        skill_registry.install(DummySkill)
        skill_registry.install(NoCredSkill)
        assert len(skill_registry.skills) == 2
        assert skill_registry.get("dummy") is not None
        assert skill_registry.get("nocred") is not None
