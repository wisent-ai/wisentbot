"""Tests for OutcomeTracker skill."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
from singularity.skills.outcome_tracker import OutcomeTracker, OUTCOMES_FILE


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with temp storage."""
    test_file = tmp_path / "outcomes.json"
    with patch("singularity.skills.outcome_tracker.OUTCOMES_FILE", test_file):
        t = OutcomeTracker()
        yield t


@pytest.fixture
def populated_tracker(tmp_path):
    """Create a tracker with existing outcomes."""
    test_file = tmp_path / "outcomes.json"
    with patch("singularity.skills.outcome_tracker.OUTCOMES_FILE", test_file):
        t = OutcomeTracker()
        # Record some outcomes synchronously
        for i in range(5):
            t.record_sync("github:create_issue", True, cost=0.002, duration_ms=500)
        for i in range(3):
            t.record_sync("github:create_pr", False, cost=0.003, error="API rate limit")
        t.record_sync("shell:run", True, cost=0.0, duration_ms=100)
        t.record_sync("content:generate", True, cost=0.01, duration_ms=2000)
        yield t


@pytest.fixture
def failing_tracker(tmp_path):
    """Create a tracker with a skill that mostly fails."""
    test_file = tmp_path / "outcomes.json"
    with patch("singularity.skills.outcome_tracker.OUTCOMES_FILE", test_file):
        t = OutcomeTracker()
        # A skill with very low success rate
        for i in range(4):
            t.record_sync("email:send", False, cost=0.001, error="SMTP timeout")
        t.record_sync("email:send", True, cost=0.001)
        # Add more data for 5+ outcomes threshold
        t.record_sync("shell:run", True, cost=0.0)
        yield t


def test_manifest(tracker):
    m = tracker.manifest
    assert m.skill_id == "outcomes"
    assert len(m.actions) == 8
    action_names = [a.name for a in m.actions]
    assert "record" in action_names
    assert "report" in action_names
    assert "recommendations" in action_names


def test_check_credentials(tracker):
    assert tracker.check_credentials() is True


@pytest.mark.asyncio
async def test_record_success(tracker):
    result = await tracker.execute("record", {
        "tool": "github:create_issue",
        "success": True,
        "cost": 0.002,
        "duration_ms": 450,
    })
    assert result.success
    assert "success" in result.message.lower()
    assert result.data["outcome"]["success"] is True
    assert result.data["outcome"]["skill_id"] == "github"
    assert result.data["outcome"]["action"] == "create_issue"


@pytest.mark.asyncio
async def test_record_failure(tracker):
    result = await tracker.execute("record", {
        "tool": "shell:run",
        "success": False,
        "error": "Permission denied",
    })
    assert result.success  # Recording succeeded
    assert "FAILURE" in result.message
    assert result.data["outcome"]["error"] == "Permission denied"


@pytest.mark.asyncio
async def test_record_requires_tool(tracker):
    result = await tracker.execute("record", {"success": True})
    assert not result.success


def test_record_sync(tracker):
    tracker.record_sync("github:star", True, cost=0.001, duration_ms=200)
    assert len(tracker._session_outcomes) == 1
    assert tracker._session_outcomes[0]["success"] is True


@pytest.mark.asyncio
async def test_report_empty(tracker):
    result = await tracker.execute("report", {})
    assert result.success
    assert result.data["total"] == 0


@pytest.mark.asyncio
async def test_report_with_data(populated_tracker):
    result = await populated_tracker.execute("report", {})
    assert result.success
    assert result.data["total_actions"] == 10
    assert result.data["successes"] == 7
    assert result.data["failures"] == 3
    assert result.data["success_rate"] == 70.0
    assert "github" in result.data["per_skill"]


@pytest.mark.asyncio
async def test_report_skill_filter(populated_tracker):
    result = await populated_tracker.execute("report", {"skill_filter": "github"})
    assert result.success
    assert result.data["total_actions"] == 8  # 5 issues + 3 PRs


@pytest.mark.asyncio
async def test_failures_list(populated_tracker):
    result = await populated_tracker.execute("failures", {"last_n": 5})
    assert result.success
    assert result.data["total_failures"] == 3
    assert len(result.data["common_errors"]) > 0
    assert result.data["common_errors"][0]["error"].startswith("API rate limit")


@pytest.mark.asyncio
async def test_trends(populated_tracker):
    result = await populated_tracker.execute("trends", {})
    assert result.success
    assert len(result.data["sessions"]) >= 1


@pytest.mark.asyncio
async def test_recommendations_insufficient_data(tracker):
    tracker.record_sync("test:action", True)
    result = await tracker.execute("recommendations", {})
    assert result.success
    assert result.data.get("reason") == "insufficient_data"


@pytest.mark.asyncio
async def test_recommendations_flags_low_success(failing_tracker):
    result = await failing_tracker.execute("recommendations", {})
    assert result.success
    recs = result.data["recommendations"]
    # email skill has 20% success rate (1/5) -> should be flagged
    low_success = [r for r in recs if r["type"] == "low_success_rate"]
    assert len(low_success) > 0
    assert any("email" in r.get("skill", "") for r in low_success)


@pytest.mark.asyncio
async def test_cost_analysis(populated_tracker):
    result = await populated_tracker.execute("cost_analysis", {})
    assert result.success
    assert result.data["total_cost"] > 0
    assert result.data["wasted_cost"] > 0
    assert len(result.data["per_skill"]) > 0


@pytest.mark.asyncio
async def test_session_summary(populated_tracker):
    result = await populated_tracker.execute("session_summary", {})
    assert result.success
    assert result.data["total_actions"] == 10
    assert result.data["success_rate"] == 70.0


@pytest.mark.asyncio
async def test_end_session(populated_tracker):
    result = await populated_tracker.execute("end_session", {})
    assert result.success
    assert "session_end" in result.data


@pytest.mark.asyncio
async def test_unknown_action(tracker):
    result = await tracker.execute("nonexistent", {})
    assert not result.success
