"""Tests for the skill system — base classes, registry, and plugin loader."""

import json
import pytest
from pathlib import Path
from typing import Dict

from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest
from singularity.skills.base.skill import Skill
from singularity.skills.base.registry import SkillRegistry
from singularity.skills.loader.loader import PluginLoader
from singularity.skills.loader.registry import (
    SkillMetadata, MCPServerInfo, WIRING_HOOKS, SKILL_DIRECTORIES, MCP_REGISTRY_URL, MARKETPLACES,
)


# ─── Test Skill Implementation ──────────────────────────────────────────


class MockSkill(Skill):
    """A concrete Skill for testing."""

    def __init__(self, credentials=None, skill_id="mock", required_creds=None):
        super().__init__(credentials)
        self._skill_id = skill_id
        self._required_creds = required_creds or []

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id=self._skill_id,
            name="Mock Skill",
            version="1.0.0",
            category="test",
            description="A mock skill for testing",
            actions=[
                SkillAction(
                    name="do_thing",
                    description="Do a thing",
                    parameters={"param1": {"type": "str", "required": True}},
                    estimated_cost=0.01,
                ),
                SkillAction(
                    name="do_other",
                    description="Do another thing",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=self._required_creds,
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if action == "do_thing":
            return SkillResult(success=True, message=f"Did thing with {params.get('param1')}", cost=0.01)
        if action == "do_other":
            return SkillResult(success=True, message="Did other thing")
        return SkillResult(success=False, message=f"Unknown action: {action}")


# ─── SkillResult ─────────────────────────────────────────────────────────


class TestSkillResult:
    def test_defaults(self):
        r = SkillResult(success=True)
        assert r.success is True
        assert r.message == ""
        assert r.data == {}
        assert r.cost == 0
        assert r.revenue == 0
        assert r.asset_created is None

    def test_full_result(self):
        r = SkillResult(
            success=True, message="Created repo",
            data={"url": "https://github.com/test/repo"},
            cost=0.01, revenue=5.0,
            asset_created={"type": "repo", "name": "test"},
        )
        assert r.data["url"] == "https://github.com/test/repo"
        assert r.revenue == 5.0
        assert r.asset_created["type"] == "repo"

    def test_failure_result(self):
        r = SkillResult(success=False, message="Missing API key")
        assert r.success is False


# ─── SkillAction ─────────────────────────────────────────────────────────


class TestSkillAction:
    def test_creation(self):
        a = SkillAction(
            name="create_repo", description="Create GitHub repo",
            parameters={"name": {"type": "str"}},
            estimated_cost=0.01, estimated_duration_seconds=5,
            success_probability=0.95,
        )
        assert a.name == "create_repo"
        assert a.estimated_cost == 0.01
        assert a.success_probability == 0.95

    def test_defaults(self):
        a = SkillAction(name="x", description="y", parameters={})
        assert a.estimated_cost == 0
        assert a.estimated_duration_seconds == 10
        assert a.success_probability == 0.8


# ─── SkillManifest ───────────────────────────────────────────────────────


class TestSkillManifest:
    def test_creation(self):
        m = SkillManifest(
            skill_id="github", name="GitHub", version="1.0.0",
            category="dev", description="GitHub management",
            actions=[], required_credentials=["GITHUB_TOKEN"],
        )
        assert m.skill_id == "github"
        assert m.required_credentials == ["GITHUB_TOKEN"]

    def test_defaults(self):
        m = SkillManifest(
            skill_id="x", name="X", version="1.0", category="test",
            description="test", actions=[], required_credentials=[],
        )
        assert m.install_cost == 0
        assert m.author == "system"


# ─── Skill Base Class ───────────────────────────────────────────────────


class TestSkillBase:
    def test_init_defaults(self):
        s = MockSkill()
        assert s.credentials == {}
        assert s.initialized is False
        assert s._usage_count == 0

    def test_init_with_credentials(self):
        s = MockSkill(credentials={"KEY": "value"})
        assert s.credentials["KEY"] == "value"

    def test_get_actions(self):
        s = MockSkill()
        actions = s.get_actions()
        assert len(actions) == 2
        assert actions[0].name == "do_thing"

    def test_get_action_found(self):
        s = MockSkill()
        a = s.get_action("do_thing")
        assert a is not None
        assert a.name == "do_thing"

    def test_get_action_not_found(self):
        s = MockSkill()
        assert s.get_action("nonexistent") is None

    def test_estimate_cost(self):
        s = MockSkill()
        assert s.estimate_cost("do_thing", {}) == 0.01
        assert s.estimate_cost("do_other", {}) == 0
        assert s.estimate_cost("nonexistent", {}) == 0

    def test_check_credentials_no_requirements(self):
        s = MockSkill(required_creds=[])
        assert s.check_credentials() is True

    def test_check_credentials_all_present(self):
        s = MockSkill(credentials={"API_KEY": "val"}, required_creds=["API_KEY"])
        assert s.check_credentials() is True

    def test_check_credentials_missing(self):
        s = MockSkill(credentials={}, required_creds=["API_KEY"])
        assert s.check_credentials() is False

    def test_check_credentials_empty_value(self):
        s = MockSkill(credentials={"API_KEY": ""}, required_creds=["API_KEY"])
        assert s.check_credentials() is False

    def test_get_missing_credentials(self):
        s = MockSkill(credentials={"A": "val"}, required_creds=["A", "B", "C"])
        missing = s.get_missing_credentials()
        assert "B" in missing
        assert "C" in missing
        assert "A" not in missing

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        s = MockSkill(required_creds=[])
        result = await s.initialize()
        assert result is True
        assert s.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure_missing_creds(self):
        s = MockSkill(credentials={}, required_creds=["MISSING_KEY"])
        result = await s.initialize()
        assert result is False
        assert s.initialized is False

    def test_record_usage(self):
        s = MockSkill()
        s.record_usage(cost=0.01, revenue=5.0)
        assert s._usage_count == 1
        assert s._total_cost == 0.01
        assert s._total_revenue == 5.0

    def test_record_usage_cumulative(self):
        s = MockSkill()
        s.record_usage(cost=0.01, revenue=1.0)
        s.record_usage(cost=0.02, revenue=2.0)
        assert s._usage_count == 2
        assert abs(s._total_cost - 0.03) < 1e-9
        assert abs(s._total_revenue - 3.0) < 1e-9

    def test_stats(self):
        s = MockSkill()
        s.record_usage(cost=1.0, revenue=5.0)
        stats = s.stats
        assert stats["usage_count"] == 1
        assert stats["total_cost"] == 1.0
        assert stats["total_revenue"] == 5.0
        assert stats["profit"] == 4.0

    def test_to_dict(self):
        s = MockSkill()
        d = s.to_dict()
        assert d["skill_id"] == "mock"
        assert d["name"] == "Mock Skill"
        assert d["category"] == "test"
        assert len(d["actions"]) == 2
        assert d["initialized"] is False

    @pytest.mark.asyncio
    async def test_execute_success(self):
        s = MockSkill()
        result = await s.execute("do_thing", {"param1": "hello"})
        assert result.success is True
        assert "hello" in result.message

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self):
        s = MockSkill()
        result = await s.execute("nonexistent", {})
        assert result.success is False


# ─── SkillRegistry ───────────────────────────────────────────────────────


class TestSkillRegistry:
    def test_init_empty(self):
        reg = SkillRegistry()
        assert reg.skills == {}
        assert reg.credentials == {}
        assert reg.loader is None

    def test_set_credentials(self):
        reg = SkillRegistry()
        reg.set_credentials({"API_KEY": "test"})
        assert reg.credentials["API_KEY"] == "test"

    def test_set_credentials_updates_existing_skills(self):
        reg = SkillRegistry()
        skill = MockSkill()
        reg.skills["mock"] = skill
        reg.set_credentials({"NEW_KEY": "value"})
        assert skill.credentials["NEW_KEY"] == "value"

    def test_install_by_class(self):
        reg = SkillRegistry()
        result = reg.install(MockSkill)
        assert result is True
        assert "mock" in reg.skills

    def test_install_by_id_no_loader(self):
        reg = SkillRegistry()
        result = reg.install("some_skill")
        assert result is False

    def test_uninstall(self):
        reg = SkillRegistry()
        reg.install(MockSkill)
        assert reg.uninstall("mock") is True
        assert "mock" not in reg.skills

    def test_uninstall_nonexistent(self):
        reg = SkillRegistry()
        assert reg.uninstall("nonexistent") is False

    def test_get(self):
        reg = SkillRegistry()
        reg.install(MockSkill)
        skill = reg.get("mock")
        assert skill is not None
        assert skill.manifest.skill_id == "mock"

    def test_get_nonexistent(self):
        reg = SkillRegistry()
        assert reg.get("nope") is None

    def test_list_skills(self):
        reg = SkillRegistry()
        reg.install(MockSkill)
        skills = reg.list_skills()
        assert len(skills) == 1
        assert skills[0]["skill_id"] == "mock"

    def test_list_all_actions(self):
        reg = SkillRegistry()
        reg.install(MockSkill)
        actions = reg.list_all_actions()
        assert len(actions) == 2
        assert actions[0]["skill_id"] == "mock"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        reg = SkillRegistry()
        reg.install(MockSkill)
        result = await reg.execute("mock", "do_thing", {"param1": "test"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_skill_not_found(self):
        reg = SkillRegistry()
        result = await reg.execute("nonexistent", "action", {})
        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_execute_initializes_skill(self):
        reg = SkillRegistry()
        skill = MockSkill(required_creds=[])
        reg.skills["mock"] = skill
        assert skill.initialized is False
        await reg.execute("mock", "do_thing", {"param1": "x"})
        assert skill.initialized is True

    @pytest.mark.asyncio
    async def test_execute_records_usage(self):
        reg = SkillRegistry()
        reg.install(MockSkill)
        await reg.execute("mock", "do_thing", {"param1": "x"})
        skill = reg.get("mock")
        assert skill._usage_count == 1

    def test_get_skills_for_llm(self):
        reg = SkillRegistry()
        reg.install(MockSkill)
        text = reg.get_skills_for_llm()
        assert "INSTALLED SKILLS" in text
        assert "[mock]" in text
        assert "do_thing" in text
        assert "do_other" in text


# ─── SkillMetadata ───────────────────────────────────────────────────────


class TestSkillMetadata:
    def test_creation(self):
        m = SkillMetadata(
            skill_id="github", module="singularity.skills.builtin.github",
            class_name="GitHubSkill", name="GitHub", version="1.0.0",
            category="dev", description="GitHub management",
            required_credentials=["GITHUB_TOKEN"],
        )
        assert m.skill_id == "github"
        assert m.source_type == "python"

    def test_defaults(self):
        m = SkillMetadata(
            skill_id="x", module="m", class_name="C", name="X",
            version="1.0", category="test", description="test",
            required_credentials=[],
        )
        assert m.wiring is None
        assert m.actions == []
        assert m.install_cost == 0
        assert m.author == "system"
        assert m.user_invocable is True
        assert m.requires_bins == []
        assert m.requires_env == []
        assert m.os_platforms == []


# ─── MCPServerInfo ───────────────────────────────────────────────────────


class TestMCPServerInfo:
    def test_creation(self):
        s = MCPServerInfo(name="test-mcp", description="Test MCP server")
        assert s.name == "test-mcp"
        assert s.transport == "stdio"
        assert s.args == []
        assert s.env == {}


# ─── PluginLoader ────────────────────────────────────────────────────────


class TestPluginLoader:
    def test_loads_default_registry(self):
        loader = PluginLoader()
        available = loader.list_available()
        assert len(available) > 10  # Should have 20+ skills from registry.json

    def test_get_manifest(self):
        loader = PluginLoader()
        manifest = loader.get_manifest("github")
        assert manifest is not None
        assert manifest.skill_id == "github"
        assert manifest.class_name == "GitHubSkill"

    def test_get_manifest_nonexistent(self):
        loader = PluginLoader()
        assert loader.get_manifest("nonexistent_skill_xyz") is None

    def test_list_available_by_category(self):
        loader = PluginLoader()
        social = loader.list_available(category="social")
        assert all(s.category == "social" for s in social)

    def test_register_new_skill(self):
        loader = PluginLoader()
        meta = SkillMetadata(
            skill_id="test_new", module="test.module",
            class_name="TestSkill", name="Test", version="1.0",
            category="test", description="test skill",
            required_credentials=[],
        )
        loader.register(meta)
        assert loader.get_manifest("test_new") is not None

    def test_is_loaded_false_initially(self):
        loader = PluginLoader()
        assert loader.is_loaded("github") is False

    def test_list_loaded_empty_initially(self):
        loader = PluginLoader()
        assert loader.list_loaded() == []


# ─── ValidationMixin ─────────────────────────────────────────────────────


class TestValidationMixin:
    def test_check_credentials_all_present(self):
        loader = PluginLoader()
        result = loader.check_credentials("github", {"GITHUB_TOKEN": "tok"})
        assert result is True

    def test_check_credentials_missing(self):
        loader = PluginLoader()
        result = loader.check_credentials("github", {})
        assert result is False

    def test_check_credentials_unknown_skill(self):
        loader = PluginLoader()
        result = loader.check_credentials("nonexistent", {"KEY": "val"})
        assert result is False

    def test_get_missing_credentials(self):
        loader = PluginLoader()
        missing = loader.get_missing_credentials("github", {})
        assert "GITHUB_TOKEN" in missing

    def test_no_cred_skills_always_pass(self):
        loader = PluginLoader()
        # Find a skill with no required credentials
        for meta in loader.list_available():
            if not meta.required_credentials:
                assert loader.check_credentials(meta.skill_id, {}) is True
                break


# ─── Constants ───────────────────────────────────────────────────────────


class TestLoaderConstants:
    def test_skill_directories_defined(self):
        assert len(SKILL_DIRECTORIES) > 0
        assert all(isinstance(d, Path) for d in SKILL_DIRECTORIES)

    def test_mcp_registry_url(self):
        assert MCP_REGISTRY_URL.startswith("https://")

    def test_marketplaces_defined(self):
        assert "anthropic" in MARKETPLACES

    def test_wiring_hooks_defined(self):
        assert "cognition_hooks" in WIRING_HOOKS
        assert "llm" in WIRING_HOOKS
        assert "agent_info" in WIRING_HOOKS
        assert callable(WIRING_HOOKS["cognition_hooks"])


# ─── Registry JSON Validation ────────────────────────────────────────────


class TestRegistryJson:
    def test_registry_file_exists(self):
        registry_path = Path(__file__).parent.parent / "singularity" / "skills" / "registry.json"
        assert registry_path.exists()

    def test_registry_is_valid_json(self):
        registry_path = Path(__file__).parent.parent / "singularity" / "skills" / "registry.json"
        data = json.loads(registry_path.read_text())
        assert "version" in data
        assert "skills" in data

    def test_registry_skills_have_required_fields(self):
        registry_path = Path(__file__).parent.parent / "singularity" / "skills" / "registry.json"
        data = json.loads(registry_path.read_text())
        for skill_id, skill_data in data["skills"].items():
            assert "module" in skill_data, f"{skill_id} missing 'module'"
            assert "class" in skill_data, f"{skill_id} missing 'class'"
            assert "manifest" in skill_data, f"{skill_id} missing 'manifest'"
            manifest = skill_data["manifest"]
            assert "skill_id" in manifest, f"{skill_id} missing 'manifest.skill_id'"
            assert "name" in manifest, f"{skill_id} missing 'manifest.name'"

    def test_registry_skill_ids_match(self):
        """Registry key should match manifest.skill_id."""
        registry_path = Path(__file__).parent.parent / "singularity" / "skills" / "registry.json"
        data = json.loads(registry_path.read_text())
        for skill_id, skill_data in data["skills"].items():
            manifest_id = skill_data["manifest"].get("skill_id")
            assert manifest_id == skill_id, (
                f"Registry key '{skill_id}' doesn't match manifest skill_id '{manifest_id}'"
            )
