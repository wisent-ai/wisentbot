"""Tests for DecisionReplaySkill - counterfactual decision analysis."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
from singularity.skills.decision_replay import DecisionReplaySkill, REPLAY_DATA_FILE


@pytest.fixture
def skill(tmp_path):
    """Create a skill instance with temp data paths."""
    with patch("singularity.skills.decision_replay.REPLAY_DATA_FILE", tmp_path / "replay.json"):
        s = DecisionReplaySkill()
        s._data_file = tmp_path / "replay.json"
        # Monkey-patch _ensure_data to use tmp_path
        s._save(s._default_state())
        yield s


@pytest.fixture
def sample_decisions():
    return [
        {
            "id": "dec-001",
            "category": "skill_selection",
            "context": "need to deploy a web service with code_review",
            "choice": "vercel",
            "alternatives": ["docker", "serverless"],
            "reasoning": "Quick deployment",
            "confidence": 0.7,
            "tags": ["deploy", "web"],
            "timestamp": 1700000000,
            "created_at": "2024-01-01T00:00:00",
            "outcome": {"success": True, "details": "deployed ok"},
        },
        {
            "id": "dec-002",
            "category": "error_handling",
            "context": "api endpoint returning 500 errors with docker",
            "choice": "restart service",
            "alternatives": ["rollback", "debug logs"],
            "reasoning": "Fastest fix",
            "confidence": 0.5,
            "tags": ["error", "api"],
            "timestamp": 1700001000,
            "created_at": "2024-01-01T01:00:00",
            "outcome": {"success": False, "details": "error persisted"},
        },
        {
            "id": "dec-003",
            "category": "strategy",
            "context": "prioritize revenue generation or self-improvement",
            "choice": "revenue",
            "alternatives": ["self-improvement", "replication"],
            "reasoning": "Need funds",
            "confidence": 0.8,
            "tags": ["strategy", "priority"],
            "timestamp": 1700002000,
            "created_at": "2024-01-01T02:00:00",
            "outcome": None,
        },
    ]


@pytest.fixture
def sample_rules():
    return [
        {
            "id": "rule-001",
            "rule_text": "prefer docker for reliable deployments",
            "category": "skill_preference",
            "confidence": 0.8,
            "skill_id": "docker",
        },
        {
            "id": "rule-002",
            "rule_text": "avoid restart service for 500 errors, use debug logs instead",
            "category": "failure_pattern",
            "confidence": 0.9,
            "skill_id": "",
        },
        {
            "id": "rule-003",
            "rule_text": "self-improvement has high success rate for long-term growth",
            "category": "success_pattern",
            "confidence": 0.7,
            "skill_id": "",
        },
    ]


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDecisionReplayBasics:
    def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "decision_replay"
        assert len(m.actions) == 6

    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success

    def test_replay_missing_id(self, skill):
        result = run(skill.execute("replay", {}))
        assert not result.success
        assert "required" in result.message.lower()

    def test_default_state(self, skill):
        data = skill._load()
        assert data["replays"] == []
        assert data["stats"]["total_replays"] == 0


class TestRuleRelevance:
    def test_keyword_overlap(self, skill):
        rule = {"rule_text": "docker deployments are fast", "skill_id": "", "category": ""}
        score = skill._rule_relevance_score(rule, "need docker for deploy", "", [], "", [])
        assert score > 0  # "docker" and "deployments"/"deploy" overlap

    def test_skill_id_match(self, skill):
        rule = {"rule_text": "some rule", "skill_id": "vercel", "category": ""}
        score = skill._rule_relevance_score(rule, "deploy on vercel", "", [], "", [])
        assert score >= 0.5

    def test_choice_in_rule(self, skill):
        rule = {"rule_text": "prefer docker for stability", "skill_id": "", "category": ""}
        score = skill._rule_relevance_score(rule, "need deployment", "docker", [], "", [])
        assert score >= 0.4

    def test_category_alignment(self, skill):
        rule = {"rule_text": "some pattern", "skill_id": "", "category": "failure_pattern"}
        score = skill._rule_relevance_score(rule, "context", "", [], "error_handling", [])
        assert score >= 0.3


class TestReplay:
    def test_replay_decision_not_found(self, skill):
        with patch.object(skill, "_get_decisions", return_value=[]):
            result = run(skill.execute("replay", {"decision_id": "missing"}))
            assert not result.success
            assert "not found" in result.message.lower()

    def test_replay_no_rules(self, skill, sample_decisions):
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=[]):
            result = run(skill.execute("replay", {"decision_id": "dec-001"}))
            assert not result.success
            assert "no learned rules" in result.message.lower()

    def test_replay_with_rules(self, skill, sample_decisions, sample_rules):
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=sample_rules):
            result = run(skill.execute("replay", {"decision_id": "dec-001"}))
            assert result.success
            assert "replay" in result.data
            data = skill._load()
            assert data["stats"]["total_replays"] == 1

    def test_replay_records_reversal(self, skill, sample_decisions, sample_rules):
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=sample_rules):
            result = run(skill.execute("replay", {"decision_id": "dec-002"}))
            assert result.success
            # Rule says "avoid restart service" -> should suggest change
            replay = result.data["replay"]
            assert "decision_id" in replay


class TestBatchReplay:
    def test_batch_no_decisions(self, skill):
        with patch.object(skill, "_get_decisions", return_value=[]):
            result = run(skill.execute("batch_replay", {"count": 5}))
            assert not result.success

    def test_batch_replay_success(self, skill, sample_decisions, sample_rules):
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=sample_rules):
            result = run(skill.execute("batch_replay", {"count": 10}))
            assert result.success
            assert result.data["total_replayed"] == 3
            assert "reversal_rate" in result.data

    def test_batch_category_filter(self, skill, sample_decisions, sample_rules):
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=sample_rules):
            result = run(skill.execute("batch_replay", {"count": 10, "category": "strategy"}))
            assert result.success
            assert result.data["total_replayed"] == 1


class TestImpactReport:
    def test_no_decisions(self, skill):
        with patch.object(skill, "_get_decisions", return_value=[]):
            result = run(skill.execute("impact_report", {}))
            assert not result.success

    def test_report_generation(self, skill, sample_decisions, sample_rules):
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=sample_rules):
            result = run(skill.execute("impact_report", {"window_days": 30}))
            assert result.success
            assert "learning_quality_score" in result.data
            assert "top_impactful_rules" in result.data
            data = skill._load()
            assert len(data["reports"]) == 1


class TestFindReversals:
    def test_no_decisions(self, skill):
        with patch.object(skill, "_get_decisions", return_value=[]):
            result = run(skill.execute("find_reversals", {}))
            assert not result.success

    def test_find_reversals(self, skill, sample_decisions, sample_rules):
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=sample_rules):
            result = run(skill.execute("find_reversals", {"min_confidence": 0.5}))
            assert result.success
            assert "total_reversals" in result.data


class TestTimeline:
    def test_no_replay_data(self, skill):
        result = run(skill.execute("timeline", {}))
        assert not result.success

    def test_timeline_with_data(self, skill, sample_decisions, sample_rules):
        # First do a batch replay to generate data
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=sample_rules):
            run(skill.execute("batch_replay", {"count": 10}))
            result = run(skill.execute("timeline", {}))
            assert result.success
            assert "timeline" in result.data
            assert "trend" in result.data


class TestWhatIf:
    def test_missing_params(self, skill):
        result = run(skill.execute("what_if", {}))
        assert not result.success

    def test_what_if_analysis(self, skill, sample_decisions):
        custom_rules = [
            {"rule_text": "prefer serverless for cost savings", "confidence": 0.9},
            {"rule_text": "avoid vercel for production workloads", "confidence": 0.8},
        ]
        with patch.object(skill, "_get_decisions", return_value=sample_decisions), \
             patch.object(skill, "_get_rules", return_value=[]):
            result = run(skill.execute("what_if", {
                "decision_id": "dec-001",
                "custom_rules": custom_rules,
            }))
            assert result.success
            assert "custom_rules_result" in result.data
            assert "actual_rules_result" in result.data
