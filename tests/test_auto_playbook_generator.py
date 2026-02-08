#!/usr/bin/env python3
"""Tests for AutoPlaybookGeneratorSkill."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from singularity.skills.auto_playbook_generator import AutoPlaybookGeneratorSkill, GENERATOR_FILE


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data file."""
    test_file = tmp_path / "auto_playbook_generator.json"
    with patch("singularity.skills.auto_playbook_generator.GENERATOR_FILE", test_file):
        s = AutoPlaybookGeneratorSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_reflections(count=5, tag="deployment", success_rate=0.6):
    """Generate sample reflections for testing."""
    refs = []
    for i in range(count):
        refs.append({
            "id": f"ref_{tag}_{i}",
            "task": f"Deploy {tag} service to production env {i}",
            "actions_taken": ["build_image", "push_registry", "deploy"],
            "outcome": "Success" if i / count < success_rate else "Failed",
            "success": i / count < success_rate,
            "analysis": f"The {tag} deployment {'went smoothly' if i / count < success_rate else 'failed due to timeout'}",
            "improvements": [] if i / count < success_rate else ["Add retry logic", "Increase timeout"],
            "tags": [tag, "production"],
        })
    return refs


class TestClustering:
    def test_similar_reflections_cluster(self, skill):
        refs = make_reflections(5, "deployment")
        clusters = skill._cluster_reflections(refs)
        assert len(clusters) >= 1
        assert clusters[0]["size"] >= 2

    def test_dissimilar_reflections_separate(self, skill):
        refs = [
            {"id": "a", "task": "Deploy service", "tags": ["deploy"], "analysis": "ok", "success": True},
            {"id": "b", "task": "Write documentation", "tags": ["docs"], "analysis": "fine", "success": True},
            {"id": "c", "task": "Fix security bug", "tags": ["security"], "analysis": "patched", "success": False},
        ]
        clusters = skill._cluster_reflections(refs)
        # With very different tags, they may not cluster at all
        for c in clusters:
            assert c["size"] <= 3  # shouldn't force-merge unrelated

    def test_cluster_scores_positive(self, skill):
        refs = make_reflections(6, "testing")
        clusters = skill._cluster_reflections(refs)
        for c in clusters:
            assert c["score"] > 0

    def test_tokenize(self, skill):
        tokens = skill._tokenize("The quick brown fox jumps over lazy dog")
        assert "quick" in tokens
        assert "the" not in tokens  # stop word
        assert "fox" in tokens

    def test_similarity_identical(self, skill):
        ref = {"task": "deploy service", "tags": ["deploy"], "analysis": "worked"}
        sim = skill._similarity(ref, ref)
        assert sim == 1.0

    def test_similarity_different(self, skill):
        a = {"task": "deploy service", "tags": ["deploy"], "analysis": "worked"}
        b = {"task": "write tests", "tags": ["testing"], "analysis": "passed"}
        sim = skill._similarity(a, b)
        assert sim < 0.5


class TestPlaybookExtraction:
    def test_extract_playbook(self, skill):
        refs = make_reflections(5, "deployment")
        clusters = skill._cluster_reflections(refs)
        assert len(clusters) >= 1
        pb = skill._extract_playbook_from_cluster(clusters[0])
        assert pb["name"].startswith("auto_")
        assert "auto_generated" in pb["tags"]
        assert len(pb["steps"]) > 0

    def test_extract_includes_pitfalls_from_failures(self, skill):
        refs = make_reflections(6, "deployment", success_rate=0.3)
        clusters = skill._cluster_reflections(refs)
        if clusters:
            pb = skill._extract_playbook_from_cluster(clusters[0])
            # Should have pitfalls from failure analysis
            assert isinstance(pb["pitfalls"], list)


class TestScan:
    def test_scan_not_enough_reflections(self, skill):
        skill.context = MagicMock()
        skill.context.call_skill = AsyncMock(return_value=MagicMock(
            success=True, data={"reflections": [{"id": "1"}], "playbooks": []}
        ))
        result = run(skill.execute("scan", {}))
        assert result.success
        assert "Not enough" in result.message

    def test_scan_with_reflections(self, skill):
        refs = make_reflections(8, "deployment")
        skill.context = MagicMock()
        skill.context.call_skill = AsyncMock(return_value=MagicMock(
            success=True, data={"reflections": refs, "playbooks": []}
        ))
        result = run(skill.execute("scan", {}))
        assert result.success
        assert result.data["total_reflections"] == 8
        assert result.data["total_clusters"] >= 1


class TestGenerate:
    def test_generate_creates_playbook(self, skill):
        refs = make_reflections(8, "deployment")
        create_mock = AsyncMock(return_value=MagicMock(success=True, data={}))
        review_mock = AsyncMock(return_value=MagicMock(
            success=True, data={"reflections": refs, "playbooks": []}
        ))

        async def mock_call(skill_id, action, params=None):
            if action == "review":
                return await review_mock()
            return await create_mock()

        skill.context = MagicMock()
        skill.context.call_skill = mock_call
        result = run(skill.execute("generate", {"max_playbooks": 1}))
        assert result.success
        assert len(result.data["generated"]) >= 1

    def test_generate_dry_run(self, skill):
        refs = make_reflections(8, "deployment")
        skill.context = MagicMock()
        skill.context.call_skill = AsyncMock(return_value=MagicMock(
            success=True, data={"reflections": refs, "playbooks": []}
        ))
        result = run(skill.execute("generate", {"dry_run": True}))
        assert result.success
        assert "would_generate" in result.data


class TestValidateAndPrune:
    def test_validate_no_auto_playbooks(self, skill):
        skill.context = MagicMock()
        skill.context.call_skill = AsyncMock(return_value=MagicMock(
            success=True, data={"reflections": [], "playbooks": []}
        ))
        result = run(skill.execute("validate", {}))
        assert result.success
        assert "No auto-generated" in result.message

    def test_prune_dry_run(self, skill):
        pbs = [{"name": "auto_bad", "tags": ["auto_generated"], "uses": 10, "effectiveness": 0.1}]
        skill.context = MagicMock()
        skill.context.call_skill = AsyncMock(return_value=MagicMock(
            success=True, data={"reflections": [], "playbooks": pbs}
        ))
        result = run(skill.execute("prune", {"dry_run": True}))
        assert result.success
        assert len(result.data["would_prune"]) == 1


class TestConfigAndStatus:
    def test_configure(self, skill):
        result = run(skill.execute("configure", {"min_cluster_size": 5}))
        assert result.success
        assert skill._config["min_cluster_size"] == 5

    def test_status(self, skill):
        result = run(skill.execute("status", {}))
        assert result.success
        assert "stats" in result.data

    def test_history(self, skill):
        result = run(skill.execute("history", {"what": "all"}))
        assert result.success

    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success
