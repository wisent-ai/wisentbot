#!/usr/bin/env python3
"""Tests for GoalProgressEventBridgeSkill."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from singularity.skills.goal_progress_events import (
    GoalProgressEventBridgeSkill,
    BRIDGE_STATE_FILE,
)
from singularity.skills.base import SkillResult


@pytest.fixture
def tmp_data(tmp_path):
    """Patch data file to use tmp dir."""
    test_file = tmp_path / "goal_progress_events.json"
    with patch.object(
        GoalProgressEventBridgeSkill, "_load_state"
    ) as mock_load:
        mock_load.side_effect = lambda self=None: None
        skill = GoalProgressEventBridgeSkill()
    skill._init_empty()
    # Patch save to use tmp
    def patched_save():
        test_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_snapshot": skill._last_snapshot,
            "event_history": skill._event_history,
            "config": skill._config,
            "stats": skill._stats,
        }
        with open(test_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    skill._save_state = patched_save
    return skill


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestInstantiation:
    def test_manifest(self, tmp_data):
        skill = tmp_data
        assert skill.manifest.skill_id == "goal_progress_events"
        assert skill.manifest.version == "1.0.0"
        assert skill.manifest.category == "meta"

    def test_actions(self, tmp_data):
        names = [a.name for a in tmp_data.manifest.actions]
        assert "monitor" in names
        assert "configure" in names
        assert "status" in names
        assert "history" in names
        assert "emit_test" in names
        assert "stall_check" in names

    def test_default_config(self, tmp_data):
        assert tmp_data._config["emit_on_created"] is True
        assert tmp_data._config["emit_on_completed"] is True
        assert tmp_data._config["stall_threshold_hours"] == 24

    def test_unknown_action(self, tmp_data):
        r = run(tmp_data.execute("nope", {}))
        assert not r.success
        assert "Unknown action" in r.message


class TestMonitor:
    def test_no_context(self, tmp_data):
        """Monitor without context returns gracefully."""
        tmp_data.context = None
        r = run(tmp_data.execute("monitor", {}))
        assert r.success
        assert r.data["events_emitted"] == 0
        assert tmp_data._stats["monitors_run"] == 1

    def test_detects_new_goals(self, tmp_data):
        """Monitor detects newly created goals."""
        goals_data = {
            "goals": [
                {"id": "g1", "title": "Test Goal", "pillar": "revenue",
                 "priority": "high", "milestones": [], "deadline": None}
            ],
            "completed_goals": [],
        }
        tmp_data._get_goals_state = AsyncMock(return_value=goals_data)
        tmp_data._emit_event = AsyncMock(return_value=True)
        r = run(tmp_data.execute("monitor", {}))
        assert r.success
        assert r.data["events_emitted"] == 1
        assert "created:Test Goal" in r.data["events_detail"]
        assert tmp_data._stats["goals_created_detected"] == 1

    def test_detects_completed_goals(self, tmp_data):
        """Monitor detects newly completed goals."""
        tmp_data._last_snapshot["goal_ids"] = ["g1"]
        goals_data = {
            "goals": [],
            "completed_goals": [
                {"id": "g1", "title": "Done Goal", "pillar": "revenue",
                 "priority": "high", "status": "completed", "outcome": "success",
                 "duration_hours": 2.5, "milestones": []}
            ],
        }
        tmp_data._get_goals_state = AsyncMock(return_value=goals_data)
        tmp_data._emit_event = AsyncMock(return_value=True)
        r = run(tmp_data.execute("monitor", {}))
        assert r.success
        assert r.data["events_emitted"] == 1
        assert "completed:Done Goal" in r.data["events_detail"]

    def test_detects_abandoned_goals(self, tmp_data):
        """Monitor detects newly abandoned goals."""
        tmp_data._last_snapshot["goal_ids"] = ["g2"]
        goals_data = {
            "goals": [],
            "completed_goals": [
                {"id": "g2", "title": "Dropped", "pillar": "other",
                 "priority": "low", "status": "abandoned", "abandon_reason": "not needed",
                 "milestones": []}
            ],
        }
        tmp_data._get_goals_state = AsyncMock(return_value=goals_data)
        tmp_data._emit_event = AsyncMock(return_value=True)
        r = run(tmp_data.execute("monitor", {}))
        assert r.success
        assert r.data["events_emitted"] == 1
        assert "abandoned:Dropped" in r.data["events_detail"]

    def test_detects_milestone_completion(self, tmp_data):
        """Monitor detects newly completed milestones."""
        tmp_data._last_snapshot["goal_ids"] = ["g3"]
        tmp_data._last_snapshot["milestone_states"] = {"g3": []}
        goals_data = {
            "goals": [
                {"id": "g3", "title": "In Progress", "pillar": "self_improvement",
                 "priority": "medium", "milestones": [
                     {"index": 0, "title": "Step 1", "completed": True, "completed_at": "2026-01-01T00:00:00"},
                     {"index": 1, "title": "Step 2", "completed": False, "completed_at": None},
                 ]}
            ],
            "completed_goals": [],
        }
        tmp_data._get_goals_state = AsyncMock(return_value=goals_data)
        tmp_data._emit_event = AsyncMock(return_value=True)
        r = run(tmp_data.execute("monitor", {}))
        assert r.success
        assert tmp_data._stats["milestones_completed_detected"] == 1

    def test_detects_pillar_shift(self, tmp_data):
        """Monitor detects significant pillar distribution change."""
        tmp_data._last_snapshot["pillar_distribution"] = {"revenue": 1.0}
        tmp_data._last_snapshot["goal_ids"] = ["g1"]
        goals_data = {
            "goals": [
                {"id": "g1", "title": "A", "pillar": "self_improvement",
                 "priority": "medium", "milestones": []},
            ],
            "completed_goals": [],
        }
        tmp_data._get_goals_state = AsyncMock(return_value=goals_data)
        tmp_data._emit_event = AsyncMock(return_value=True)
        r = run(tmp_data.execute("monitor", {}))
        assert r.success
        assert tmp_data._stats["pillar_shifts_detected"] == 1

    def test_no_duplicate_events(self, tmp_data):
        """Second monitor with no changes emits nothing."""
        tmp_data._last_snapshot = {
            "goal_ids": ["g1"],
            "completed_ids": [],
            "abandoned_ids": [],
            "milestone_states": {"g1": []},
            "pillar_distribution": {"revenue": 1.0},
            "last_monitor_ts": "2026-01-01T00:00:00Z",
        }
        goals_data = {
            "goals": [
                {"id": "g1", "title": "Stable", "pillar": "revenue",
                 "priority": "medium", "milestones": []}
            ],
            "completed_goals": [],
        }
        tmp_data._get_goals_state = AsyncMock(return_value=goals_data)
        tmp_data._emit_event = AsyncMock(return_value=True)
        r = run(tmp_data.execute("monitor", {}))
        assert r.success
        assert r.data["events_emitted"] == 0


class TestStallCheck:
    def test_no_context(self, tmp_data):
        tmp_data.context = None
        r = run(tmp_data.execute("stall_check", {}))
        assert r.success
        assert r.data["stalled_goals"] == 0

    def test_detects_stalled_goal(self, tmp_data):
        """Stall check detects goals idle past threshold."""
        goals_data = {
            "goals": [
                {"id": "g1", "title": "Stale", "pillar": "revenue",
                 "priority": "high", "created_at": "2020-01-01T00:00:00",
                 "milestones": [], "progress_notes": [], "deadline": None}
            ],
            "completed_goals": [],
        }
        tmp_data._get_goals_state = AsyncMock(return_value=goals_data)
        tmp_data._emit_event = AsyncMock(return_value=True)
        r = run(tmp_data.execute("stall_check", {}))
        assert r.success
        assert r.data["stalled_goals"] == 1
        assert tmp_data._stats["stalls_detected"] == 1


class TestConfigure:
    def test_update_flags(self, tmp_data):
        r = run(tmp_data.execute("configure", {"emit_on_created": False}))
        assert r.success
        assert tmp_data._config["emit_on_created"] is False

    def test_update_thresholds(self, tmp_data):
        r = run(tmp_data.execute("configure", {
            "stall_threshold_hours": 48,
            "pillar_shift_threshold": 0.3,
        }))
        assert r.success
        assert tmp_data._config["stall_threshold_hours"] == 48
        assert tmp_data._config["pillar_shift_threshold"] == 0.3

    def test_no_params(self, tmp_data):
        r = run(tmp_data.execute("configure", {}))
        assert not r.success


class TestStatus:
    def test_returns_stats(self, tmp_data):
        r = run(tmp_data.execute("status", {}))
        assert r.success
        assert "stats" in r.data
        assert "config" in r.data


class TestHistory:
    def test_empty(self, tmp_data):
        r = run(tmp_data.execute("history", {}))
        assert r.success
        assert r.data["total"] == 0

    def test_with_events(self, tmp_data):
        tmp_data._event_history = [
            {"topic": "goal.created", "timestamp": "t1", "emitted": True},
            {"topic": "goal.completed", "timestamp": "t2", "emitted": True},
        ]
        r = run(tmp_data.execute("history", {"limit": 10}))
        assert r.success
        assert r.data["total"] == 2

    def test_topic_filter(self, tmp_data):
        tmp_data._event_history = [
            {"topic": "goal.created", "timestamp": "t1", "emitted": True},
            {"topic": "goal.completed", "timestamp": "t2", "emitted": True},
        ]
        r = run(tmp_data.execute("history", {"topic_filter": "goal.created"}))
        assert r.success
        assert r.data["total"] == 1


class TestEmitTest:
    def test_emit_no_bus(self, tmp_data):
        r = run(tmp_data.execute("emit_test", {}))
        assert r.success
        assert r.data["emitted"] is False


class TestHelpers:
    def test_pillar_distribution(self, tmp_data):
        goals = [
            {"pillar": "revenue"},
            {"pillar": "revenue"},
            {"pillar": "self_improvement"},
        ]
        dist = tmp_data._calc_pillar_distribution(goals)
        assert dist["revenue"] == pytest.approx(0.667, abs=0.001)
        assert dist["self_improvement"] == pytest.approx(0.333, abs=0.001)

    def test_pillar_shift(self, tmp_data):
        prev = {"revenue": 0.5, "self_improvement": 0.5}
        curr = {"revenue": 0.8, "self_improvement": 0.2}
        shift = tmp_data._calc_pillar_shift(prev, curr)
        assert shift == pytest.approx(0.3, abs=0.01)

    def test_empty_distribution(self, tmp_data):
        assert tmp_data._calc_pillar_distribution([]) == {}

    def test_empty_shift(self, tmp_data):
        assert tmp_data._calc_pillar_shift({}, {}) == 0.0
