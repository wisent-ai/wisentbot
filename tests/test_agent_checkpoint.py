"""Tests for AgentCheckpointSkill."""
import json
import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch

from singularity.skills.agent_checkpoint import (
    AgentCheckpointSkill,
    _collect_state_files,
    _hash_file,
    _load_index,
    _save_index,
    CHECKPOINT_DIR,
    CHECKPOINT_INDEX,
    DATA_DIR,
)


@pytest.fixture
def tmp_data(tmp_path):
    """Set up temporary data directory with some state files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cp_dir = data_dir / "checkpoints"
    cp_dir.mkdir()
    # Create some fake skill state files
    (data_dir / "goals.json").write_text('{"goals": ["be_better"]}')
    (data_dir / "experiments.json").write_text('{"exps": []}')
    (data_dir / "strategy.json").write_text('{"pillar_scores": {}}')
    sub = data_dir / "subdir"
    sub.mkdir()
    (sub / "nested.json").write_text('{"nested": true}')
    return data_dir, cp_dir


@pytest.fixture
def skill(tmp_data):
    """Create a checkpoint skill with temp dirs."""
    data_dir, cp_dir = tmp_data
    s = AgentCheckpointSkill()
    s._data_dir = data_dir
    s._checkpoint_dir = cp_dir
    # Patch module-level constants for index operations
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        yield s


@pytest.mark.asyncio
async def test_save_creates_checkpoint(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        result = await skill.execute("save", {"label": "test-save", "reason": "testing"})
    assert result.success
    assert result.data["files_count"] >= 3  # goals, experiments, strategy
    assert result.data["checkpoint_id"]
    # Verify files were copied
    cp_id = result.data["checkpoint_id"]
    cp_path = cp_dir / cp_id
    assert cp_path.exists()
    assert (cp_path / "metadata.json").exists()
    assert (cp_path / "goals.json").exists()


@pytest.mark.asyncio
async def test_list_checkpoints(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        await skill.execute("save", {"label": "first"})
        await skill.execute("save", {"label": "second"})
        result = await skill.execute("list", {})
    assert result.success
    assert result.data["total_count"] == 2
    assert len(result.data["checkpoints"]) == 2


@pytest.mark.asyncio
async def test_restore_checkpoint(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        # Save checkpoint
        save_result = await skill.execute("save", {"label": "before-change"})
        cp_id = save_result.data["checkpoint_id"]
        # Modify a file
        (data_dir / "goals.json").write_text('{"goals": ["modified"]}')
        # Restore
        restore_result = await skill.execute("restore", {"checkpoint_id": cp_id})
    assert restore_result.success
    assert restore_result.data["restored_count"] >= 3
    # Verify file was restored
    restored = json.loads((data_dir / "goals.json").read_text())
    assert restored["goals"] == ["be_better"]


@pytest.mark.asyncio
async def test_restore_dry_run(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        save_result = await skill.execute("save", {"label": "dry"})
        cp_id = save_result.data["checkpoint_id"]
        (data_dir / "goals.json").write_text('{"goals": ["changed"]}')
        result = await skill.execute("restore", {"checkpoint_id": cp_id, "dry_run": True})
    assert result.success
    assert "Dry run" in result.message
    # File should NOT be restored
    current = json.loads((data_dir / "goals.json").read_text())
    assert current["goals"] == ["changed"]


@pytest.mark.asyncio
async def test_diff_checkpoints(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        save_a = await skill.execute("save", {"label": "a"})
        (data_dir / "goals.json").write_text('{"goals": ["new_goal"]}')
        (data_dir / "new_file.json").write_text('{"new": true}')
        save_b = await skill.execute("save", {"label": "b"})
        result = await skill.execute("diff", {
            "checkpoint_a": save_a.data["checkpoint_id"],
            "checkpoint_b": save_b.data["checkpoint_id"],
        })
    assert result.success
    assert len(result.data["modified"]) >= 1  # goals.json was modified
    assert "new_file.json" in result.data["added"]


@pytest.mark.asyncio
async def test_diff_with_current(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        save_a = await skill.execute("save", {"label": "before"})
        (data_dir / "goals.json").write_text('{"goals": ["changed"]}')
        result = await skill.execute("diff", {
            "checkpoint_a": save_a.data["checkpoint_id"],
            "checkpoint_b": "current",
        })
    assert result.success
    modified_files = [m["file"] for m in result.data["modified"]]
    assert "goals.json" in modified_files


@pytest.mark.asyncio
async def test_export_and_import(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    export_dir = data_dir / "exports"
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        save_result = await skill.execute("save", {"label": "exportme"})
        cp_id = save_result.data["checkpoint_id"]
        export_path = str(export_dir / f"{cp_id}.tar.gz")
        export_result = await skill.execute("export", {
            "checkpoint_id": cp_id,
            "export_path": export_path,
        })
    assert export_result.success
    assert os.path.exists(export_path)
    assert export_result.data["export_size_bytes"] > 0


@pytest.mark.asyncio
async def test_prune_removes_old(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        for i in range(5):
            await skill.execute("save", {"label": "" if i < 3 else f"keep-{i}"})
        result = await skill.execute("prune", {"keep_count": 2, "keep_labeled": True})
    assert result.success
    assert len(result.data["pruned"]) >= 1


@pytest.mark.asyncio
async def test_prune_dry_run(skill, tmp_data):
    data_dir, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"), \
         patch("singularity.skills.agent_checkpoint.DATA_DIR", data_dir):
        for i in range(5):
            await skill.execute("save", {})
        result = await skill.execute("prune", {"keep_count": 2, "dry_run": True})
    assert result.success
    assert "Dry run" in result.message
    assert len(result.data["to_prune"]) >= 3


@pytest.mark.asyncio
async def test_auto_policy(skill, tmp_data):
    _, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"):
        result = await skill.execute("auto_policy", {
            "enabled": True,
            "triggers": ["pre_self_modify", "hourly"],
            "max_checkpoints": 15,
        })
    assert result.success
    assert result.data["policy"]["enabled"] is True
    assert "pre_self_modify" in result.data["policy"]["triggers"]


@pytest.mark.asyncio
async def test_auto_policy_invalid_trigger(skill, tmp_data):
    _, cp_dir = tmp_data
    with patch("singularity.skills.agent_checkpoint.CHECKPOINT_DIR", cp_dir), \
         patch("singularity.skills.agent_checkpoint.CHECKPOINT_INDEX", cp_dir / "index.json"):
        result = await skill.execute("auto_policy", {"triggers": ["invalid_trigger"]})
    assert not result.success
    assert "Invalid triggers" in result.message


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message


@pytest.mark.asyncio
async def test_restore_nonexistent(skill):
    result = await skill.execute("restore", {"checkpoint_id": "cp_does_not_exist"})
    assert not result.success
    assert "not found" in result.message


def test_collect_state_files(tmp_data):
    data_dir, _ = tmp_data
    files = _collect_state_files(data_dir)
    assert "goals.json" in files
    assert "experiments.json" in files
    assert "subdir/nested.json" in files
    # checkpoints dir should be excluded
    assert not any("checkpoint" in f for f in files)


def test_hash_file(tmp_data):
    data_dir, _ = tmp_data
    h = _hash_file(data_dir / "goals.json")
    assert len(h) == 64  # SHA-256 hex digest


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "agent_checkpoint"
    assert len(m.actions) == 8
    action_names = {a.name for a in m.actions}
    assert action_names == {"save", "restore", "list", "diff", "export", "import_checkpoint", "prune", "auto_policy"}
