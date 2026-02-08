#!/usr/bin/env python3
"""Tests for CheckpointEventBridgeSkill."""

import pytest
from singularity.skills.checkpoint_event_bridge import (
    CheckpointEventBridgeSkill,
    REACTIVE_TRIGGERS,
    CHECKPOINT_EVENTS,
)


@pytest.fixture
def skill():
    s = CheckpointEventBridgeSkill()
    s._state = s._default_state()
    return s


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "checkpoint_event_bridge"
    assert m.category == "infrastructure"
    assert len(m.actions) == 8


def test_wire_all(skill):
    result = skill.execute("wire")
    assert result.success
    assert len(result.data["activated"]) == len(REACTIVE_TRIGGERS)
    assert result.data["total_active"] == len(REACTIVE_TRIGGERS)


def test_wire_specific(skill):
    result = skill.execute("wire", {"trigger_ids": ["pre_self_modify", "pre_deploy"]})
    assert result.success
    assert sorted(result.data["activated"]) == ["pre_deploy", "pre_self_modify"]
    assert result.data["total_active"] == 2


def test_wire_duplicate(skill):
    skill.execute("wire", {"trigger_ids": ["pre_self_modify"]})
    result = skill.execute("wire", {"trigger_ids": ["pre_self_modify"]})
    assert result.success
    assert result.data["already_active"] == ["pre_self_modify"]


def test_wire_invalid(skill):
    result = skill.execute("wire", {"trigger_ids": ["nonexistent"]})
    assert not result.success
    assert result.data["invalid"] == ["nonexistent"]


def test_unwire(skill):
    skill.execute("wire", {"trigger_ids": ["pre_self_modify", "pre_deploy"]})
    result = skill.execute("unwire", {"trigger_ids": ["pre_self_modify"]})
    assert result.success
    assert result.data["removed"] == ["pre_self_modify"]
    assert result.data["remaining_active"] == 1


def test_unwire_not_found(skill):
    result = skill.execute("unwire", {"trigger_ids": ["not_active"]})
    assert not result.success


def test_emit_valid(skill):
    result = skill.execute("emit", {"event_type": "saved", "data": {"checkpoint_id": "cp-123"}})
    assert result.success
    assert result.data["event"]["topic"] == "checkpoint.saved"


def test_emit_invalid_type(skill):
    result = skill.execute("emit", {"event_type": "invalid"})
    assert not result.success


def test_health_check(skill):
    result = skill.execute("health_check")
    assert result.success
    assert "health_score" in result.data
    assert "alerts" in result.data
    assert result.data["health_score"] <= 100


def test_simulate_matching(skill):
    skill.execute("wire")
    result = skill.execute("simulate", {"topic": "self_modify.prompt"})
    assert result.success
    assert result.data["would_checkpoint"] is True
    assert len(result.data["matching_triggers"]) >= 1


def test_simulate_no_match(skill):
    skill.execute("wire")
    result = skill.execute("simulate", {"topic": "random.unrelated.topic"})
    assert result.success
    assert result.data["would_checkpoint"] is False


def test_simulate_inactive_match(skill):
    result = skill.execute("simulate", {"topic": "self_modify.prompt"})
    assert result.success
    assert result.data["would_checkpoint"] is False
    assert len(result.data["inactive_matches"]) >= 1


def test_configure(skill):
    result = skill.execute("configure", {"stale_threshold_hours": 12, "storage_threshold_mb": 200})
    assert result.success
    assert skill._state["health_config"]["stale_threshold_hours"] == 12
    assert skill._state["health_config"]["storage_threshold_mb"] == 200


def test_configure_invalid(skill):
    result = skill.execute("configure", {"stale_threshold_hours": 0})
    assert not result.success


def test_history(skill):
    skill.execute("emit", {"event_type": "saved", "data": {}})
    skill.execute("emit", {"event_type": "restored", "data": {}})
    result = skill.execute("history", {"limit": 10})
    assert result.success
    assert result.data["total_events"] >= 2


def test_status(skill):
    skill.execute("wire", {"trigger_ids": ["pre_self_modify"]})
    result = skill.execute("status")
    assert result.success
    assert result.data["active_trigger_count"] == 1
    assert result.data["total_triggers"] == len(REACTIVE_TRIGGERS)
    assert len(result.data["events"]) == len(CHECKPOINT_EVENTS)


def test_unknown_action(skill):
    result = skill.execute("nonexistent")
    assert not result.success
