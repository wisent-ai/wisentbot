"""Tests for BillingSchedulerBridgeSkill."""
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.billing_scheduler_bridge import (
    BillingSchedulerBridgeSkill, BRIDGE_DATA_FILE
)


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    bf = tmp_path / "billing_scheduler_bridge.json"
    monkeypatch.setattr(
        "singularity.skills.billing_scheduler_bridge.BRIDGE_DATA_FILE", bf
    )
    return bf


@pytest.fixture
def skill():
    return BillingSchedulerBridgeSkill()


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "billing_scheduler_bridge"
    assert m.category == "revenue"
    assert len(m.actions) == 8


@pytest.mark.asyncio
async def test_setup_default(skill):
    result = await skill.execute("setup", {})
    assert result.success
    assert "daily" in result.message
    assert result.data["interval"] == "daily"
    assert result.data["interval_seconds"] == 86400


@pytest.mark.asyncio
async def test_setup_weekly(skill):
    result = await skill.execute("setup", {"interval": "weekly"})
    assert result.success
    assert result.data["interval"] == "weekly"
    assert result.data["interval_seconds"] == 604800


@pytest.mark.asyncio
async def test_setup_invalid_interval(skill):
    result = await skill.execute("setup", {"interval": "biweekly"})
    assert not result.success
    assert "Invalid interval" in result.message


@pytest.mark.asyncio
async def test_status_inactive(skill):
    result = await skill.execute("status", {})
    assert result.success
    assert "INACTIVE" in result.message


@pytest.mark.asyncio
async def test_status_after_setup(skill):
    await skill.execute("setup", {"interval": "daily"})
    result = await skill.execute("status", {})
    assert result.success
    assert "ACTIVE" in result.message
    assert result.data["active"] is True


@pytest.mark.asyncio
async def test_pause_and_resume(skill):
    await skill.execute("setup", {})
    r = await skill.execute("pause", {"reason": "maintenance"})
    assert r.success
    assert "paused" in r.message.lower()

    status = await skill.execute("status", {})
    assert "PAUSED" in status.message

    r2 = await skill.execute("resume", {})
    assert r2.success

    status2 = await skill.execute("status", {})
    assert "ACTIVE" in status2.message


@pytest.mark.asyncio
async def test_pause_inactive_fails(skill):
    r = await skill.execute("pause", {})
    assert not r.success


@pytest.mark.asyncio
async def test_resume_not_paused_fails(skill):
    await skill.execute("setup", {})
    r = await skill.execute("resume", {})
    assert not r.success


@pytest.mark.asyncio
async def test_configure_webhook(skill):
    r = await skill.execute("configure_webhook", {
        "url": "https://example.com/billing-hook",
        "secret": "s3cret",
    })
    assert r.success
    assert "example.com" in r.message
    assert r.data["enabled"] is True


@pytest.mark.asyncio
async def test_configure_webhook_missing_url(skill):
    r = await skill.execute("configure_webhook", {})
    assert not r.success


@pytest.mark.asyncio
async def test_run_now_with_mock(skill):
    """Test run_now using a mock billing pipeline."""
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = {"customers_billed": 2, "total_revenue": 1.50}
    mock_result.message = "Billed 2 customers"

    mock_billing = MagicMock()
    mock_billing.execute = AsyncMock(return_value=mock_result)

    with patch(
        "singularity.skills.billing_scheduler_bridge.BillingSchedulerBridgeSkill._execute_billing_cycle"
    ) as mock_exec:
        from singularity.skills.billing_scheduler_bridge import BillingRun
        mock_exec.return_value = BillingRun(
            run_id="BRUN-test123",
            triggered_at="2026-01-01T00:00:00Z",
            trigger_type="manual",
            completed_at="2026-01-01T00:00:01Z",
            success=True,
            customers_billed=2,
            total_revenue=1.50,
            invoices_generated=2,
            dry_run=False,
        )
        result = await skill.execute("run_now", {})
        assert result.success
        assert result.data["customers_billed"] == 2
        assert result.data["total_revenue"] == 1.50


@pytest.mark.asyncio
async def test_history_empty(skill):
    r = await skill.execute("history", {})
    assert r.success
    assert r.data["total_runs"] == 0


@pytest.mark.asyncio
async def test_health_no_runs(skill):
    r = await skill.execute("health", {})
    assert r.success
    assert r.data["health_score"] <= 100
    assert r.data["total_runs"] == 0


@pytest.mark.asyncio
async def test_health_after_setup(skill):
    await skill.execute("setup", {})
    r = await skill.execute("health", {})
    assert r.success
    assert r.data["active"] is True


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_billing_preset_in_scheduler_presets():
    from singularity.skills.scheduler_presets import BUILTIN_PRESETS
    assert "billing_automation" in BUILTIN_PRESETS
    preset = BUILTIN_PRESETS["billing_automation"]
    assert preset.pillar == "revenue"
    assert len(preset.schedules) == 3
    assert any("billing_scheduler_bridge" in s.skill_id for s in preset.schedules)
