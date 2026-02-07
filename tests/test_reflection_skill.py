"""Tests for ReflectionSkill - self-improvement feedback loop."""

import pytest
import asyncio
from singularity.skills.reflection import ReflectionSkill


def make_action(tool, status="success", cost=0.001, tokens=100, cycle=1, message=""):
    """Helper to create action dicts matching autonomous_agent format."""
    return {
        "cycle": cycle,
        "tool": tool,
        "params": {},
        "result": {"status": status, "message": message, "data": {}},
        "api_cost_usd": cost,
        "tokens": tokens,
    }


def make_skill(actions=None, balance=50.0, cycle=10):
    """Create a ReflectionSkill wired to test data."""
    skill = ReflectionSkill()
    _actions = actions if actions is not None else []
    skill.set_agent_hooks(
        get_recent_actions=lambda: _actions,
        get_balance=lambda: balance,
        get_cycle=lambda: cycle,
    )
    return skill


class TestManifest:
    def test_skill_id(self):
        skill = ReflectionSkill()
        assert skill.manifest.skill_id == "reflection"

    def test_has_actions(self):
        skill = ReflectionSkill()
        names = [a.name for a in skill.manifest.actions]
        assert "analyze" in names
        assert "patterns" in names
        assert "suggest" in names
        assert "report" in names
        assert "set_strategy" in names
        assert "get_strategies" in names
        assert "get_reflections" in names


class TestCheckCredentials:
    def test_not_connected(self):
        skill = ReflectionSkill()
        assert not skill.check_credentials()

    def test_connected(self):
        skill = make_skill()
        assert skill.check_credentials()


class TestAnalyze:
    def test_empty_actions(self):
        skill = make_skill([])
        result = asyncio.get_event_loop().run_until_complete(skill.execute("analyze", {}))
        assert result.success
        assert result.data["total_actions"] == 0

    def test_all_successes(self):
        actions = [make_action("fs:write", "success") for _ in range(5)]
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("analyze", {}))
        assert result.success
        assert result.data["success_rate_pct"] == 100.0
        assert result.data["successes"] == 5
        assert result.data["failures"] == 0

    def test_mixed_results(self):
        actions = [
            make_action("fs:write", "success"),
            make_action("fs:write", "success"),
            make_action("shell:bash", "failed"),
            make_action("shell:bash", "error"),
        ]
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("analyze", {}))
        assert result.data["success_rate_pct"] == 50.0
        assert result.data["failures"] == 1
        assert result.data["errors"] == 1

    def test_cost_tracking(self):
        actions = [
            make_action("fs:write", "success", cost=0.01),
            make_action("fs:write", "success", cost=0.02),
        ]
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("analyze", {}))
        assert result.data["total_cost_usd"] == 0.03
        assert result.data["cost_per_action_usd"] == 0.015

    def test_last_n_filter(self):
        actions = [make_action("a", "failed")] * 5 + [make_action("b", "success")] * 5
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("analyze", {"last_n": 5}))
        assert result.data["total_actions"] == 5
        assert result.data["success_rate_pct"] == 100.0

    def test_includes_balance(self):
        skill = make_skill([], balance=42.5)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("analyze", {}))
        assert result.data.get("current_balance_usd") == 42.5


class TestPatterns:
    def test_empty(self):
        skill = make_skill([])
        result = asyncio.get_event_loop().run_until_complete(skill.execute("patterns", {}))
        assert result.success
        assert result.data["tool_patterns"] == {}

    def test_per_tool_stats(self):
        actions = [
            make_action("fs:write", "success"),
            make_action("fs:write", "success"),
            make_action("shell:bash", "failed", message="permission denied"),
            make_action("shell:bash", "failed", message="timeout"),
        ]
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("patterns", {}))
        p = result.data["tool_patterns"]
        assert p["fs:write"]["success_rate_pct"] == 100.0
        assert p["shell:bash"]["success_rate_pct"] == 0.0
        assert result.data["best_performing_tool"] == "fs:write"
        assert result.data["worst_performing_tool"] == "shell:bash"

    def test_repeated_failures_detected(self):
        actions = [make_action("bad_tool", "failed")] * 4
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("patterns", {}))
        assert "bad_tool" in result.data.get("repeated_failure_tools", [])


class TestSuggest:
    def test_empty(self):
        skill = make_skill([])
        result = asyncio.get_event_loop().run_until_complete(skill.execute("suggest", {}))
        assert result.success

    def test_low_success_rate_suggestion(self):
        actions = [make_action("t", "failed")] * 8 + [make_action("t", "success")] * 2
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("suggest", {}))
        priorities = [s["priority"] for s in result.data["suggestions"]]
        assert "high" in priorities

    def test_good_performance(self):
        actions = [make_action(f"tool{i}", "success") for i in range(10)]
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("suggest", {}))
        assert any(s["category"] == "positive" for s in result.data["suggestions"])

    def test_stores_reflection(self):
        actions = [make_action("t", "success")]
        skill = make_skill(actions)
        asyncio.get_event_loop().run_until_complete(skill.execute("suggest", {}))
        assert len(skill._reflections) == 1


class TestStrategy:
    def test_set_strategy(self):
        skill = make_skill([])
        result = asyncio.get_event_loop().run_until_complete(
            skill.execute("set_strategy", {"strategy": "Focus on coding", "reason": "Best ROI"})
        )
        assert result.success
        assert len(skill._strategies) == 1

    def test_empty_strategy_rejected(self):
        skill = make_skill([])
        result = asyncio.get_event_loop().run_until_complete(
            skill.execute("set_strategy", {"strategy": "", "reason": "test"})
        )
        assert not result.success

    def test_get_strategies(self):
        skill = make_skill([])
        asyncio.get_event_loop().run_until_complete(
            skill.execute("set_strategy", {"strategy": "S1", "reason": "R1"})
        )
        result = asyncio.get_event_loop().run_until_complete(
            skill.execute("get_strategies", {})
        )
        assert result.data["count"] == 1
        assert result.data["latest"]["strategy"] == "S1"


class TestReport:
    def test_generates_full_report(self):
        actions = [
            make_action("fs:write", "success"),
            make_action("shell:bash", "failed"),
        ]
        skill = make_skill(actions)
        result = asyncio.get_event_loop().run_until_complete(skill.execute("report", {}))
        assert result.success
        assert "metrics" in result.data
        assert "patterns" in result.data
        assert "suggestions" in result.data


class TestNotConnected:
    def test_fails_gracefully(self):
        skill = ReflectionSkill()
        result = asyncio.get_event_loop().run_until_complete(skill.execute("analyze", {}))
        assert not result.success
        assert "not connected" in result.message.lower()


class TestUnknownAction:
    def test_unknown_action(self):
        skill = make_skill([])
        result = asyncio.get_event_loop().run_until_complete(skill.execute("nonexistent", {}))
        assert not result.success
