"""Tests for PlaybookPipelineSkill - playbook to pipeline conversion."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from singularity.skills.playbook_pipeline import PlaybookPipelineSkill, PIPELINE_DATA_FILE


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data path."""
    with patch("singularity.skills.playbook_pipeline.PIPELINE_DATA_FILE", tmp_path / "pp.json"):
        s = PlaybookPipelineSkill()
        yield s


@pytest.fixture
def sample_playbook():
    return {
        "name": "deploy-flow",
        "task_pattern": "deploy application",
        "steps": [
            "Run tests to verify everything passes",
            "Git commit all changes",
            "Create a pull request",
            "Deploy to production",
        ],
        "pitfalls": ["Don't skip tests"],
        "tags": ["deploy", "ci"],
    }


class TestStepMapping:
    @pytest.mark.asyncio
    async def test_match_run_tests(self, skill):
        result = await skill.execute("match_step", {"step_text": "Run tests to verify"})
        assert result.success
        assert result.data["match"]["tool"] == "shell:run"
        assert result.data["confidence"] > 0

    @pytest.mark.asyncio
    async def test_match_git_commit(self, skill):
        result = await skill.execute("match_step", {"step_text": "Git commit all changes"})
        assert result.success
        assert "shell:run" in result.data["match"]["tool"]

    @pytest.mark.asyncio
    async def test_match_create_pr(self, skill):
        result = await skill.execute("match_step", {"step_text": "Create a pull request"})
        assert result.success
        assert result.data["match"]["tool"] == "github:create_pr"

    @pytest.mark.asyncio
    async def test_no_match(self, skill):
        result = await skill.execute("match_step", {"step_text": "quantum entangle the flux capacitor"})
        assert result.success
        conf = result.data.get("confidence", 0)
        above = result.data.get("above_threshold", False)
        assert conf < 0.5 or not above

    @pytest.mark.asyncio
    async def test_match_step_missing_param(self, skill):
        result = await skill.execute("match_step", {})
        assert not result.success


class TestConversion:
    @pytest.mark.asyncio
    async def test_convert_playbook(self, skill, sample_playbook):
        result = await skill.execute("convert", {
            "playbook_name": "deploy-flow",
            "playbook": sample_playbook,
        })
        assert result.success
        conv = result.data["conversion"]
        assert conv["total_steps"] == 4
        assert conv["mapped_count"] >= 0
        # Should have attempted all 4 steps
        assert conv["mapped_count"] + conv["unmapped_count"] == 4

    @pytest.mark.asyncio
    async def test_convert_missing_playbook(self, skill):
        result = await skill.execute("convert", {"playbook_name": "nonexistent"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_convert_with_overrides(self, skill):
        """Test that param overrides are applied to the correct pipeline step."""
        playbook = {
            "name": "test-flow",
            "task_pattern": "test",
            "steps": ["run tests"],  # Short step that exactly matches keyword
        }
        result = await skill.execute("convert", {
            "playbook_name": "test-flow",
            "playbook": playbook,
            "param_overrides": {"0": {"command": "pytest -v --tb=short"}},
        })
        assert result.success
        steps = result.data["conversion"]["pipeline_steps"]
        assert len(steps) == 1
        assert steps[0]["params"]["command"] == "pytest -v --tb=short"

    @pytest.mark.asyncio
    async def test_convert_preserves_step_order(self, skill):
        """Pipeline steps should be in same order as playbook steps."""
        playbook = {
            "name": "ordered",
            "task_pattern": "test",
            "steps": ["run tests", "git commit", "create pr"],
        }
        result = await skill.execute("convert", {
            "playbook_name": "ordered",
            "playbook": playbook,
        })
        assert result.success
        report = result.data["conversion"]["mapping_report"]
        # Step indices should be in order
        indices = [r["step_index"] for r in report]
        assert indices == sorted(indices)


class TestExecution:
    @pytest.mark.asyncio
    async def test_dry_run(self, skill):
        playbook = {
            "name": "dry",
            "task_pattern": "test",
            "steps": ["run tests", "git commit"],
        }
        result = await skill.execute("execute", {
            "playbook_name": "dry",
            "playbook": playbook,
            "dry_run": True,
        })
        assert result.success
        assert result.data["dry_run"] is True
        assert len(result.data["pipeline_steps"]) > 0

    @pytest.mark.asyncio
    async def test_execute_no_context(self, skill):
        playbook = {
            "name": "no-ctx",
            "task_pattern": "test",
            "steps": ["run tests"],
        }
        result = await skill.execute("execute", {
            "playbook_name": "no-ctx",
            "playbook": playbook,
        })
        assert not result.success

    @pytest.mark.asyncio
    async def test_execute_missing_name(self, skill):
        result = await skill.execute("execute", {})
        assert not result.success

    @pytest.mark.asyncio
    async def test_execute_no_mappable_steps(self, skill):
        playbook = {
            "name": "unmappable",
            "task_pattern": "test",
            "steps": ["quantum entangle the flux capacitor"],
        }
        result = await skill.execute("execute", {
            "playbook_name": "unmappable",
            "playbook": playbook,
        })
        assert not result.success


class TestMappings:
    @pytest.mark.asyncio
    async def test_add_mapping(self, skill):
        result = await skill.execute("add_mapping", {
            "mapping_id": "custom_lint",
            "keywords": ["lint", "linting", "check style"],
            "tool": "shell:run",
            "params": {"command": "flake8 ."},
            "description": "Run linter",
        })
        assert result.success
        match = await skill.execute("match_step", {"step_text": "Run linting on code"})
        assert match.success
        assert match.data["match"]["tool"] == "shell:run"

    @pytest.mark.asyncio
    async def test_remove_mapping(self, skill):
        await skill.execute("add_mapping", {
            "mapping_id": "temp_map",
            "keywords": ["temp"],
            "tool": "shell:run",
        })
        result = await skill.execute("remove_mapping", {"mapping_id": "temp_map"})
        assert result.success

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, skill):
        result = await skill.execute("remove_mapping", {"mapping_id": "nope"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_list_mappings(self, skill):
        result = await skill.execute("list_mappings", {})
        assert result.success
        assert len(result.data["mappings"]) > 0

    @pytest.mark.asyncio
    async def test_add_mapping_missing_params(self, skill):
        result = await skill.execute("add_mapping", {"mapping_id": "x"})
        assert not result.success


class TestStatus:
    @pytest.mark.asyncio
    async def test_status(self, skill):
        result = await skill.execute("status", {})
        assert result.success
        assert "stats" in result.data
        assert "total_mappings" in result.data

    @pytest.mark.asyncio
    async def test_history_empty(self, skill):
        result = await skill.execute("history", {})
        assert result.success
        assert result.data["history"] == []

    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        result = await skill.execute("bogus_action", {})
        assert not result.success


class TestManifest:
    def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "playbook_pipeline"
        assert len(m.actions) == 8
        action_names = [a.name for a in m.actions]
        assert "convert" in action_names
        assert "execute" in action_names
        assert "add_mapping" in action_names
