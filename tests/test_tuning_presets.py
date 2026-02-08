"""Tests for TuningPresetsSkill."""

import json
import asyncio
import pytest
from pathlib import Path
from singularity.skills.tuning_presets import (
    TuningPresetsSkill, PRESETS_FILE, BUILTIN_PRESETS, BUILTIN_BUNDLES,
)
from singularity.skills.self_tuning import SelfTuningSkill, TUNING_FILE


@pytest.fixture(autouse=True)
def clean_data():
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    if TUNING_FILE.exists():
        TUNING_FILE.unlink()
    yield
    if PRESETS_FILE.exists():
        PRESETS_FILE.unlink()
    if TUNING_FILE.exists():
        TUNING_FILE.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_skill():
    return TuningPresetsSkill()


def test_manifest():
    s = make_skill()
    m = s.manifest
    assert m.skill_id == "tuning_presets"
    actions = [a.name for a in m.actions]
    assert "list_presets" in actions
    assert "preview" in actions
    assert "deploy" in actions
    assert "deploy_bundle" in actions
    assert "customize" in actions
    assert "list_bundles" in actions
    assert "status" in actions


def test_list_presets_all():
    s = make_skill()
    r = run(s.execute("list_presets", {}))
    assert r.success
    assert r.data["total"] == len(BUILTIN_PRESETS)
    ids = [p["preset_id"] for p in r.data["presets"]]
    assert "latency_batch_reduce" in ids
    assert "error_circuit_breaker" in ids
    assert "cost_model_downgrade" in ids


def test_list_presets_filter_category():
    s = make_skill()
    r = run(s.execute("list_presets", {"category": "latency"}))
    assert r.success
    for p in r.data["presets"]:
        assert p["category"] == "latency"
    assert r.data["total"] >= 2


def test_preview():
    s = make_skill()
    r = run(s.execute("preview", {"preset_id": "error_circuit_breaker"}))
    assert r.success
    assert "rule_config" in r.data
    cfg = r.data["rule_config"]
    assert cfg["metric_name"] == "skill_error_rate"
    assert cfg["condition"] == "above"
    assert cfg["target_skill"] == "error_recovery"


def test_preview_not_found():
    s = make_skill()
    r = run(s.execute("preview", {"preset_id": "nonexistent"}))
    assert not r.success


def test_deploy_preset():
    s = make_skill()
    r = run(s.execute("deploy", {"preset_id": "latency_batch_reduce"}))
    assert r.success
    assert "deployment" in r.data
    assert r.data["deployment"]["preset_id"] == "latency_batch_reduce"
    # Check it shows as deployed in status
    st = run(s.execute("status", {}))
    assert st.success
    deployed_ids = [d["preset_id"] for d in st.data["deployed"]]
    assert "latency_batch_reduce" in deployed_ids


def test_deploy_duplicate_rejected():
    s = make_skill()
    r1 = run(s.execute("deploy", {"preset_id": "latency_batch_reduce"}))
    assert r1.success
    r2 = run(s.execute("deploy", {"preset_id": "latency_batch_reduce"}))
    assert not r2.success
    assert "already deployed" in r2.message


def test_deploy_with_overrides():
    s = make_skill()
    r = run(s.execute("deploy", {
        "preset_id": "error_circuit_breaker",
        "overrides": {"threshold": 0.3, "cooldown_minutes": 20},
    }))
    assert r.success
    cfg = r.data["rule_config"]
    assert cfg["threshold"] == 0.3
    assert cfg["cooldown_minutes"] == 20


def test_deploy_bundle():
    s = make_skill()
    r = run(s.execute("deploy_bundle", {"bundle_id": "stability"}))
    assert r.success
    assert len(r.data["deployed"]) == len(BUILTIN_BUNDLES["stability"]["presets"])
    assert len(r.data["skipped"]) == 0


def test_deploy_bundle_skips_duplicates():
    s = make_skill()
    run(s.execute("deploy", {"preset_id": "error_circuit_breaker"}))
    r = run(s.execute("deploy_bundle", {"bundle_id": "stability"}))
    assert r.success
    assert len(r.data["skipped"]) == 1
    assert r.data["skipped"][0]["preset_id"] == "error_circuit_breaker"


def test_customize():
    s = make_skill()
    r = run(s.execute("customize", {
        "base_preset_id": "latency_batch_reduce",
        "custom_name": "My Latency Rule",
        "overrides": {"threshold": 3000, "window_minutes": 10},
    }))
    assert r.success
    cid = r.data["custom_id"]
    # Should appear in list
    lr = run(s.execute("list_presets", {}))
    ids = [p["preset_id"] for p in lr.data["presets"]]
    assert cid in ids


def test_list_bundles():
    s = make_skill()
    r = run(s.execute("list_bundles", {}))
    assert r.success
    ids = [b["bundle_id"] for b in r.data["bundles"]]
    assert "stability" in ids
    assert "performance" in ids
    assert "cost_aware" in ids
    assert "revenue_focused" in ids
    assert "full_auto" in ids


def test_status_empty():
    s = make_skill()
    r = run(s.execute("status", {}))
    assert r.success
    assert len(r.data["deployed"]) == 0
    assert r.data["available_count"] == len(BUILTIN_PRESETS)
