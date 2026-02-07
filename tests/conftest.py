"""Shared test fixtures for singularity tests."""

import tempfile
from pathlib import Path
from typing import Dict

import pytest

from singularity.skills.base import (
    Skill,
    SkillAction,
    SkillManifest,
    SkillRegistry,
    SkillResult,
)


class DummySkill(Skill):
    """A minimal skill for testing purposes."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="dummy",
            name="Dummy Skill",
            version="1.0.0",
            category="test",
            description="A dummy skill for testing",
            required_credentials=["DUMMY_API_KEY"],
            install_cost=0,
            actions=[
                SkillAction(
                    name="greet",
                    description="Say hello",
                    parameters={"name": {"type": "string", "required": True}},
                    estimated_cost=0.01,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="fail",
                    description="Always fails",
                    parameters={},
                    estimated_cost=0,
                    success_probability=0.0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if action == "greet":
            name = params.get("name", "World")
            return SkillResult(
                success=True,
                message=f"Hello, {name}!",
                data={"greeting": f"Hello, {name}!"},
            )
        elif action == "fail":
            return SkillResult(success=False, message="This always fails")
        return SkillResult(success=False, message=f"Unknown action: {action}")


class NoCredSkill(Skill):
    """A skill that requires no credentials."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="nocred",
            name="No Credentials Skill",
            version="1.0.0",
            category="test",
            description="A skill that needs no credentials",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="ping",
                    description="Responds with pong",
                    parameters={},
                    estimated_cost=0,
                    success_probability=1.0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if action == "ping":
            return SkillResult(success=True, message="pong", data={"response": "pong"})
        return SkillResult(success=False, message=f"Unknown action: {action}")


@pytest.fixture
def dummy_credentials():
    """Provide dummy credentials for testing."""
    return {"DUMMY_API_KEY": "test-key-12345"}


@pytest.fixture
def empty_credentials():
    """Provide empty credentials for testing."""
    return {}


@pytest.fixture
def dummy_skill(dummy_credentials):
    """Create a DummySkill with valid credentials."""
    return DummySkill(credentials=dummy_credentials)


@pytest.fixture
def nocred_skill():
    """Create a NoCredSkill."""
    return NoCredSkill()


@pytest.fixture
def skill_registry():
    """Create an empty SkillRegistry."""
    return SkillRegistry()


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for filesystem tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
