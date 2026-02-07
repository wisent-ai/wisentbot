"""Tests for GovernorSkill - agent safety, budget, and guardrails."""

import asyncio
import time
import pytest
from singularity.skills.governor import GovernorSkill, BudgetLimit, CircuitBreaker, RateLimit


@pytest.fixture
def governor(tmp_path, monkeypatch):
    monkeypatch.setattr(GovernorSkill, "DATA_DIR", tmp_path / "governor")
    return GovernorSkill()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestCheck:
    def test_allow_by_default(self, governor):
        result = run(governor.execute("check", {"action": "shell:bash"}))
        assert result.success
        assert result.data["allowed"] is True

    def test_block_over_budget(self, governor):
        run(governor.execute("set_budget", {"scope": "global", "max_amount": 0.01, "period": "1h"}))
        run(governor.execute("record_action", {"action": "test", "success": True, "cost": 0.009}))
        result = run(governor.execute("check", {"action": "test", "estimated_cost": 0.005}))
        assert result.data["allowed"] is False
        assert not result.data["checks_passed"]["budget"]

    def test_block_by_rate_limit(self, governor):
        run(governor.execute("set_rate_limit", {"action": "fast", "max_calls": 2, "period": "1h"}))
        run(governor.execute("record_action", {"action": "fast", "success": True}))
        run(governor.execute("record_action", {"action": "fast", "success": True}))
        result = run(governor.execute("check", {"action": "fast"}))
        assert result.data["allowed"] is False
        assert not result.data["checks_passed"]["rate_limit"]

    def test_block_by_guardrail(self, governor):
        run(governor.execute("set_guardrail", {
            "name": "no_shell", "rule_type": "block_action",
            "config": {"actions": ["shell:bash"]},
        }))
        result = run(governor.execute("check", {"action": "shell:bash"}))
        assert result.data["allowed"] is False

    def test_block_by_max_cost_guardrail(self, governor):
        run(governor.execute("set_guardrail", {
            "name": "cheap_only", "rule_type": "max_cost",
            "config": {"max_cost": 0.01},
        }))
        result = run(governor.execute("check", {"action": "expensive", "estimated_cost": 0.05}))
        assert result.data["allowed"] is False


class TestCircuitBreaker:
    def test_opens_after_failures(self, governor):
        for _ in range(5):
            run(governor.execute("record_action", {"action": "flaky", "success": False}))
        result = run(governor.execute("circuit_status", {"action": "flaky"}))
        assert result.data["state"] == "open"

    def test_blocks_when_open(self, governor):
        for _ in range(5):
            run(governor.execute("record_action", {"action": "flaky", "success": False}))
        result = run(governor.execute("check", {"action": "flaky"}))
        assert result.data["allowed"] is False

    def test_reset_circuit(self, governor):
        for _ in range(5):
            run(governor.execute("record_action", {"action": "flaky", "success": False}))
        run(governor.execute("circuit_status", {"action": "flaky", "command": "reset"}))
        result = run(governor.execute("circuit_status", {"action": "flaky"}))
        assert result.data["state"] == "closed"

    def test_configure_threshold(self, governor):
        run(governor.execute("circuit_status", {
            "action": "custom", "command": "configure", "threshold": 3, "cooldown": 60,
        }))
        for _ in range(3):
            run(governor.execute("record_action", {"action": "custom", "success": False}))
        result = run(governor.execute("circuit_status", {"action": "custom"}))
        assert result.data["state"] == "open"


class TestLoopDetection:
    def test_detects_dominant_action(self, governor):
        for _ in range(15):
            run(governor.execute("record_action", {"action": "stuck:loop", "success": True}))
        result = run(governor.execute("detect_loops", {"window": 20, "threshold": 0.5}))
        assert result.data["loop_detected"] is True

    def test_no_loop_when_varied(self, governor):
        for i in range(10):
            run(governor.execute("record_action", {"action": f"action_{i}", "success": True}))
        result = run(governor.execute("detect_loops", {}))
        assert result.data["loop_detected"] is False

    def test_detects_repeated_failures(self, governor):
        for _ in range(5):
            run(governor.execute("record_action", {"action": "broken", "success": False}))
        result = run(governor.execute("detect_loops", {"window": 10}))
        patterns = result.data.get("patterns", [])
        failure_patterns = [p for p in patterns if p["type"] == "repeated_failures"]
        assert len(failure_patterns) > 0


class TestBudgetAndRateLimit:
    def test_budget_window_reset(self, governor):
        run(governor.execute("set_budget", {"scope": "global", "max_amount": 1.0, "period": "1s"}))
        run(governor.execute("record_action", {"action": "a", "success": True, "cost": 0.9}))
        # Simulate window expiry
        governor._budgets["global"].window_start = time.time() - 2
        result = run(governor.execute("check", {"action": "a", "estimated_cost": 0.5}))
        assert result.data["allowed"] is True

    def test_set_budget_returns_info(self, governor):
        result = run(governor.execute("set_budget", {"scope": "shell", "max_amount": 5.0, "period": "7d"}))
        assert result.success
        assert result.data["period_seconds"] == 604800


class TestReport:
    def test_report_structure(self, governor):
        run(governor.execute("record_action", {"action": "test", "success": True, "cost": 0.01}))
        result = run(governor.execute("report", {}))
        assert result.success
        assert "health_score" in result.data
        assert "actions" in result.data
        assert result.data["actions"]["total"] == 1


class TestResetAndViolations:
    def test_violations_tracked(self, governor):
        run(governor.execute("set_guardrail", {
            "name": "block_all", "rule_type": "block_action",
            "config": {"actions": ["blocked"]},
        }))
        run(governor.execute("check", {"action": "blocked"}))
        result = run(governor.execute("violations", {}))
        assert result.data["total"] >= 1

    def test_reset_all(self, governor):
        run(governor.execute("set_budget", {"scope": "g", "max_amount": 1}))
        run(governor.execute("reset", {"scope": "all"}))
        result = run(governor.execute("report", {}))
        assert result.data["actions"]["total"] == 0

    def test_reset_specific(self, governor):
        run(governor.execute("set_budget", {"scope": "g", "max_amount": 1}))
        run(governor.execute("record_action", {"action": "a", "success": True}))
        run(governor.execute("reset", {"scope": "budgets"}))
        # History should remain
        result = run(governor.execute("report", {}))
        assert result.data["actions"]["total"] > 0


class TestPersistence:
    def test_state_persists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(GovernorSkill, "DATA_DIR", tmp_path / "governor")
        g1 = GovernorSkill()
        run(g1.execute("set_budget", {"scope": "global", "max_amount": 10.0, "period": "24h"}))
        run(g1.execute("record_action", {"action": "persisted", "success": True, "cost": 1.0}))
        # New instance should load state
        g2 = GovernorSkill()
        assert "global" in g2._budgets
        assert len(g2._action_history) == 1
