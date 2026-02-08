"""Tests for the 4 new scheduler presets: goal_stall_monitoring, revenue_goal_evaluation, dashboard_auto_check, fleet_health_auto_heal."""

import json
import asyncio
import pytest
from pathlib import Path
from singularity.skills.scheduler_presets import (
    SchedulerPresetsSkill, PRESETS_FILE, DATA_DIR, BUILTIN_PRESETS,
    FULL_AUTONOMY_PRESETS,
)


@pytest.fixture(autouse=True)
def clean_data():
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    sched = DATA_DIR / "scheduler.json"
    if sched.exists():
        sched.unlink()
    yield
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    if sched.exists():
        sched.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_skill():
    return SchedulerPresetsSkill()


# --- Preset existence and structure ---

def test_goal_stall_monitoring_preset_exists():
    assert "goal_stall_monitoring" in BUILTIN_PRESETS
    preset = BUILTIN_PRESETS["goal_stall_monitoring"]
    assert preset.name == "Goal Stall Monitoring"
    assert preset.pillar == "goal_setting"
    assert len(preset.schedules) == 2
    actions = [s.action for s in preset.schedules]
    assert "stall_check" in actions
    assert "monitor" in actions


def test_revenue_goal_evaluation_preset_exists():
    assert "revenue_goal_evaluation" in BUILTIN_PRESETS
    preset = BUILTIN_PRESETS["revenue_goal_evaluation"]
    assert preset.name == "Revenue Goal Evaluation"
    assert preset.pillar == "revenue"
    assert len(preset.schedules) == 3
    actions = [s.action for s in preset.schedules]
    assert "status" in actions
    assert "report" in actions
    assert "history" in actions


def test_dashboard_auto_check_preset_exists():
    assert "dashboard_auto_check" in BUILTIN_PRESETS
    preset = BUILTIN_PRESETS["dashboard_auto_check"]
    assert preset.name == "Dashboard Auto-Check"
    assert preset.pillar == "operations"
    assert len(preset.schedules) == 4
    actions = [s.action for s in preset.schedules]
    assert "latest" in actions
    assert "trends" in actions
    assert "subsystem_health" in actions
    assert "alerts" in actions


def test_fleet_health_auto_heal_preset_exists():
    assert "fleet_health_auto_heal" in BUILTIN_PRESETS
    preset = BUILTIN_PRESETS["fleet_health_auto_heal"]
    assert preset.name == "Fleet Health Auto-Heal"
    assert preset.pillar == "replication"
    assert len(preset.schedules) == 2
    actions = [s.action for s in preset.schedules]
    assert "monitor" in actions
    assert "fleet_check" in actions


# --- Full autonomy includes new presets ---

def test_full_autonomy_includes_new_presets():
    for preset_id in ["goal_stall_monitoring", "revenue_goal_evaluation",
                       "dashboard_auto_check", "fleet_health_auto_heal"]:
        assert preset_id in FULL_AUTONOMY_PRESETS


# --- Listing includes new presets ---

def test_list_presets_includes_new():
    skill = make_skill()
    result = run(skill.execute("list_presets", {}))
    assert result.success
    ids = [p["preset_id"] for p in result.data["presets"]]
    assert "goal_stall_monitoring" in ids
    assert "revenue_goal_evaluation" in ids
    assert "dashboard_auto_check" in ids
    assert "fleet_health_auto_heal" in ids


def test_list_presets_filter_goal_setting():
    skill = make_skill()
    result = run(skill.execute("list_presets", {"pillar": "goal_setting"}))
    assert result.success
    ids = [p["preset_id"] for p in result.data["presets"]]
    assert "goal_stall_monitoring" in ids


def test_list_presets_filter_replication():
    skill = make_skill()
    result = run(skill.execute("list_presets", {"pillar": "replication"}))
    assert result.success
    ids = [p["preset_id"] for p in result.data["presets"]]
    assert "fleet_health_auto_heal" in ids


# --- Schedule intervals are reasonable ---

def test_goal_stall_intervals():
    preset = BUILTIN_PRESETS["goal_stall_monitoring"]
    for s in preset.schedules:
        assert s.interval_seconds >= 300  # at least 5 min
        assert s.interval_seconds <= 86400  # at most 1 day


def test_dashboard_intervals():
    preset = BUILTIN_PRESETS["dashboard_auto_check"]
    for s in preset.schedules:
        assert s.interval_seconds >= 300
        assert s.interval_seconds <= 86400


def test_total_builtin_presets_count():
    # We went from 12 to 16 presets
    assert len(BUILTIN_PRESETS) >= 16
