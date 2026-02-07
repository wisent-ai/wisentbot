"""Tests for SkillComposerSkill - dynamic skill creation and registration."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from singularity.skills.skill_composer import (
    SkillComposerSkill,
    validate_skill_code,
    _to_class_name,
    _to_skill_id,
)
from singularity.skills.base import SkillRegistry


@pytest.fixture
def tmp_skills_dir(tmp_path):
    return str(tmp_path / "dynamic_skills")


@pytest.fixture
def composer(tmp_skills_dir):
    return SkillComposerSkill(dynamic_skills_dir=tmp_skills_dir)


def test_manifest():
    s = SkillComposerSkill()
    assert s.manifest.skill_id == "skill_composer"
    assert s.manifest.category == "self_improvement"
    assert len(s.manifest.actions) == 7


def test_to_class_name():
    assert _to_class_name("data_analyzer") == "DataAnalyzerSkill"
    assert _to_class_name("my_cool") == "MyCoolSkill"
    assert _to_class_name("SimpleSkill") == "SimpleSkill"


def test_to_skill_id():
    assert _to_skill_id("data_analyzer") == "data_analyzer"
    assert _to_skill_id("DataAnalyzer") == "data_analyzer"
    assert _to_skill_id("MySkill") == "my"


def test_validate_valid_code():
    code = '''
from singularity.skills.base import Skill, SkillManifest, SkillAction, SkillResult
class TestSkill(Skill):
    @property
    def manifest(self):
        return SkillManifest(skill_id="test", name="Test", version="1.0", category="test",
                             description="Test", actions=[], required_credentials=[])
    async def execute(self, action, params):
        return SkillResult(success=True)
'''
    result = validate_skill_code(code)
    assert result["valid"] is True
    assert result["class_name"] == "TestSkill"
    assert len(result["errors"]) == 0


def test_validate_syntax_error():
    result = validate_skill_code("def foo(\n")
    assert result["valid"] is False
    assert any("Syntax error" in e for e in result["errors"])


def test_validate_no_skill_class():
    result = validate_skill_code("class Foo:\n    pass\n")
    assert result["valid"] is False
    assert any("No class extending" in e for e in result["errors"])


def test_validate_missing_methods():
    code = "from singularity.skills.base import Skill\nclass BadSkill(Skill):\n    pass\n"
    result = validate_skill_code(code)
    assert result["valid"] is False
    assert any("manifest" in e for e in result["errors"])
    assert any("execute" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_create_skill(composer):
    result = await composer.execute("create", {
        "name": "greeter",
        "description": "A greeting skill",
        "actions": [
            {"name": "hello", "description": "Say hello", "parameters": {"name": {"type": "str", "required": True, "description": "Who to greet"}}},
        ],
    })
    assert result.success is True
    assert "greeter" in result.data["skill_id"]
    assert Path(result.data["filepath"]).exists()


@pytest.mark.asyncio
async def test_create_missing_name(composer):
    result = await composer.execute("create", {"actions": [{"name": "a"}]})
    assert result.success is False


@pytest.mark.asyncio
async def test_create_missing_actions(composer):
    result = await composer.execute("create", {"name": "empty"})
    assert result.success is False


@pytest.mark.asyncio
async def test_create_and_register(composer, tmp_skills_dir):
    # Create a skill
    result = await composer.execute("create", {
        "name": "math_helper",
        "description": "Do math",
        "actions": [
            {"name": "add", "description": "Add numbers", "parameters": {}},
        ],
    })
    assert result.success is True
    skill_id = result.data["skill_id"]

    # Register it
    registry = SkillRegistry()
    composer.set_registry(registry)
    reg_result = await composer.execute("register", {"skill_id": skill_id})
    assert reg_result.success is True
    assert reg_result.data["registered"] is True
    assert skill_id in [s["skill_id"] for s in registry.list_skills()]


@pytest.mark.asyncio
async def test_register_not_found(composer):
    result = await composer.execute("register", {"skill_id": "nonexistent"})
    assert result.success is False


@pytest.mark.asyncio
async def test_create_from_code(composer):
    code = '''
from singularity.skills.base import Skill, SkillManifest, SkillAction, SkillResult
from typing import Dict
class CustomSkill(Skill):
    @property
    def manifest(self):
        return SkillManifest(skill_id="custom", name="Custom", version="1.0",
                             category="test", description="Custom", actions=[], required_credentials=[])
    async def execute(self, action: str, params: Dict):
        return SkillResult(success=True, message="custom")
'''
    result = await composer.execute("create_from_code", {"name": "custom", "source_code": code})
    assert result.success is True
    assert result.data["class_name"] == "CustomSkill"


@pytest.mark.asyncio
async def test_create_from_bad_code(composer):
    result = await composer.execute("create_from_code", {"name": "bad", "source_code": "not valid python {"})
    assert result.success is False


@pytest.mark.asyncio
async def test_list_dynamic(composer):
    result = await composer.execute("list_dynamic", {})
    assert result.success is True
    assert result.data["count"] == 0

    # Create one
    await composer.execute("create", {
        "name": "test_list",
        "description": "test",
        "actions": [{"name": "a", "description": "a", "parameters": {}}],
    })
    result = await composer.execute("list_dynamic", {})
    assert result.data["count"] == 1


@pytest.mark.asyncio
async def test_export(composer):
    await composer.execute("create", {
        "name": "exportable",
        "description": "test export",
        "actions": [{"name": "run", "description": "run it", "parameters": {}}],
    })
    result = await composer.execute("export", {"skill_id": "exportable"})
    assert result.success is True
    assert "class" in result.data["source_code"]


@pytest.mark.asyncio
async def test_export_not_found(composer):
    result = await composer.execute("export", {"skill_id": "nope"})
    assert result.success is False


@pytest.mark.asyncio
async def test_describe_template(composer):
    result = await composer.execute("describe_template", {})
    assert result.success is True
    assert "example_spec" in result.data


@pytest.mark.asyncio
async def test_validate_action(composer):
    code = '''
from singularity.skills.base import Skill, SkillManifest, SkillResult
class GoodSkill(Skill):
    @property
    def manifest(self):
        return SkillManifest(skill_id="g", name="G", version="1.0", category="t",
                             description="G", actions=[], required_credentials=[])
    async def execute(self, action, params):
        return SkillResult(success=True)
'''
    result = await composer.execute("validate", {"source_code": code})
    assert result.success is True


@pytest.mark.asyncio
async def test_unknown_action(composer):
    result = await composer.execute("bogus_action", {})
    assert result.success is False
