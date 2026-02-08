"""Tests for PipelineLearningSkill."""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from singularity.skills.pipeline_learning import PipelineLearningSkill, LEARNING_DATA_FILE


@pytest.fixture
def skill():
    s = PipelineLearningSkill()
    s.context = MagicMock()
    return s


@pytest.fixture(autouse=True)
def clean_data():
    if LEARNING_DATA_FILE.exists():
        LEARNING_DATA_FILE.unlink()
    yield
    if LEARNING_DATA_FILE.exists():
        LEARNING_DATA_FILE.unlink()


def run(skill, action, params=None):
    return asyncio.get_event_loop().run_until_complete(skill.execute(action, params or {}))


def make_steps(tools, success=True, duration=100, cost=0.01):
    return [{"tool": t, "success": success, "duration_ms": duration, "cost": cost, "retries": 0} for t in tools]


class TestManifest:
    def test_manifest(self, skill):
        m = skill.manifest()
        assert m.skill_id == "pipeline_learning"
        actions = [a.name for a in m.actions]
        assert "ingest" in actions
        assert "recommend" in actions
        assert "bottlenecks" in actions
        assert "status" in actions


class TestIngest:
    def test_ingest_basic(self, skill):
        r = run(skill, "ingest", {
            "pipeline_id": "p1",
            "steps": make_steps(["shell:run", "github:create_pr"]),
            "overall_success": True,
            "strategy_used": "reliability",
            "pipeline_type": "deploy",
        })
        assert r.success
        assert "shell:run" in r.data["tools_updated"]

    def test_ingest_requires_pipeline_id(self, skill):
        r = run(skill, "ingest", {"steps": [{"tool": "x"}], "overall_success": True})
        assert not r.success

    def test_ingest_requires_steps(self, skill):
        r = run(skill, "ingest", {"pipeline_id": "p1", "overall_success": True})
        assert not r.success

    def test_ingest_updates_tool_stats(self, skill):
        run(skill, "ingest", {"pipeline_id": "p1", "steps": make_steps(["shell:run"], duration=200, cost=0.02), "overall_success": True})
        run(skill, "ingest", {"pipeline_id": "p2", "steps": make_steps(["shell:run"], success=False, duration=300, cost=0.03), "overall_success": False})
        r = run(skill, "tool_profile", {"tool": "shell:run"})
        assert r.success
        assert r.data["executions"] == 2
        assert r.data["success_rate"] == 0.5

    def test_ingest_updates_strategy_stats(self, skill):
        for i in range(3):
            run(skill, "ingest", {"pipeline_id": f"p{i}", "steps": make_steps(["shell:run"]), "overall_success": True, "strategy_used": "cost"})
        r = run(skill, "strategy_effectiveness", {})
        assert r.success
        strats = {s["strategy"]: s for s in r.data["strategies"]}
        assert strats["cost"]["success_rate"] == 1.0

    def test_ingest_tracks_pipeline_type(self, skill):
        for i in range(3):
            run(skill, "ingest", {"pipeline_id": f"p{i}", "steps": make_steps(["shell:run"]), "overall_success": True, "strategy_used": "speed", "pipeline_type": "deploy"})
        r = run(skill, "strategy_effectiveness", {})
        assert "deploy" in r.data["pipeline_type_breakdown"]
        assert r.data["pipeline_type_breakdown"]["deploy"]["best_strategy"] == "speed"


class TestRecommend:
    def _seed(self, skill, tool="shell:run", n=5, success=True, duration=100, cost=0.01):
        for i in range(n):
            run(skill, "ingest", {"pipeline_id": f"s{i}", "steps": [{"tool": tool, "success": success, "duration_ms": duration + i * 10, "cost": cost, "retries": 0}], "overall_success": success})

    def test_recommend_with_data(self, skill):
        self._seed(skill, "shell:run", 5, True, 100, 0.01)
        r = run(skill, "recommend", {"pipeline": [{"tool": "shell:run", "timeout_seconds": 30, "max_cost": 0.05}], "goal": "cost"})
        assert r.success
        assert r.data["strategy"] == "cost"
        assert r.data["adjustments_count"] >= 0

    def test_recommend_auto_strategy(self, skill):
        # Seed reliability as best
        for i in range(4):
            run(skill, "ingest", {"pipeline_id": f"r{i}", "steps": make_steps(["shell:run"]), "overall_success": True, "strategy_used": "reliability"})
        for i in range(4):
            run(skill, "ingest", {"pipeline_id": f"c{i}", "steps": make_steps(["shell:run"]), "overall_success": i < 1, "strategy_used": "cost"})
        r = run(skill, "recommend", {"pipeline": [{"tool": "shell:run"}], "goal": "auto"})
        assert r.data["strategy"] == "reliability"

    def test_recommend_empty_pipeline(self, skill):
        r = run(skill, "recommend", {"pipeline": []})
        assert not r.success

    def test_recommend_reliability_for_flaky_tools(self, skill):
        # Make shell:run flaky
        for i in range(5):
            run(skill, "ingest", {"pipeline_id": f"f{i}", "steps": [{"tool": "shell:run", "success": i < 2, "duration_ms": 100, "cost": 0.01, "retries": 1}], "overall_success": i < 2})
        r = run(skill, "recommend", {"pipeline": [{"tool": "shell:run"}], "goal": "reliability"})
        assert r.success
        tuned = r.data["pipeline"][0]
        assert tuned.get("retry_count", 0) >= 1


class TestBottlenecks:
    def test_no_bottlenecks(self, skill):
        for i in range(5):
            run(skill, "ingest", {"pipeline_id": f"p{i}", "steps": make_steps(["shell:run"]), "overall_success": True})
        r = run(skill, "bottlenecks", {})
        assert r.success
        assert len(r.data["bottlenecks"]) == 0

    def test_detects_flaky_tool(self, skill):
        for i in range(5):
            run(skill, "ingest", {"pipeline_id": f"p{i}", "steps": [{"tool": "deploy:push", "success": i < 1, "duration_ms": 5000, "cost": 0.05, "retries": 2}], "overall_success": i < 1})
        r = run(skill, "bottlenecks", {})
        assert len(r.data["bottlenecks"]) == 1
        assert r.data["bottlenecks"][0]["tool"] == "deploy:push"
        assert r.data["bottlenecks"][0]["severity"] in ("critical", "high")


class TestTuneStep:
    def test_tune_with_data(self, skill):
        for i in range(5):
            run(skill, "ingest", {"pipeline_id": f"p{i}", "steps": [{"tool": "shell:run", "success": True, "duration_ms": 100 + i * 20, "cost": 0.01 + i * 0.002, "retries": 0}], "overall_success": True})
        r = run(skill, "tune_step", {"tool": "shell:run", "current_timeout": 60, "current_retries": 3})
        assert r.success
        assert r.data["recommended"]["retry_count"] == 0  # High success rate
        assert len(r.data["suggestions"]) > 0

    def test_tune_insufficient_data(self, skill):
        r = run(skill, "tune_step", {"tool": "unknown:tool"})
        assert not r.success


class TestStatus:
    def test_empty_status(self, skill):
        r = run(skill, "status", {})
        assert r.success
        assert r.data["total_tools_tracked"] == 0

    def test_status_after_ingestion(self, skill):
        run(skill, "ingest", {"pipeline_id": "p1", "steps": make_steps(["shell:run", "github:pr"]), "overall_success": True})
        r = run(skill, "status", {})
        assert r.data["total_tools_tracked"] == 2
        assert r.data["total_executions"] == 2


class TestUnknownAction:
    def test_unknown_action(self, skill):
        r = run(skill, "nonexistent", {})
        assert not r.success
