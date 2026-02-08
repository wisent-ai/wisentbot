#!/usr/bin/env python3
"""Tests for WebhookDeliverySkill - reliable webhook delivery with retries."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.webhook_delivery import (
    WebhookDeliverySkill, DELIVERY_LOG_FILE, _sign_payload,
)
from singularity.skills.base import SkillResult


@pytest.fixture
def skill(tmp_path):
    test_file = tmp_path / "delivery_log.json"
    with patch("singularity.skills.webhook_delivery.DELIVERY_LOG_FILE", test_file):
        with patch("singularity.skills.webhook_delivery.DATA_DIR", tmp_path):
            s = WebhookDeliverySkill()
            yield s


@pytest.fixture
def mock_http_skill():
    """Mock HTTPClientSkill that returns success."""
    http = MagicMock()
    http.execute = AsyncMock(return_value=SkillResult(
        success=True, message="OK", data={"status_code": 200, "body": "ok"}
    ))
    return http


def _make_context(http_skill=None):
    ctx = MagicMock()
    ctx.get_skill = MagicMock(return_value=http_skill)
    return ctx


# ── Signing ──────────────────────────────────────

def test_sign_payload():
    sig = _sign_payload('{"key":"value"}', "secret123")
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA256 hex digest


# ── Deliver ──────────────────────────────────────

@pytest.mark.asyncio
async def test_deliver_via_http_skill(skill, mock_http_skill):
    """Deliver webhook using HTTPClientSkill."""
    skill.context = _make_context(mock_http_skill)
    result = await skill.execute("deliver", {
        "url": "https://example.com/webhook",
        "payload": {"task_id": "t1", "status": "completed"},
        "event_type": "task.completed",
    })
    assert result.success
    assert "delivered" in result.message
    assert result.data["status"] == "delivered"
    mock_http_skill.execute.assert_called()


@pytest.mark.asyncio
async def test_deliver_with_retries(skill):
    """Deliver retries on failure."""
    call_count = 0
    async def mock_send(url, body, headers, timeout):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return 500, "Server Error"
        return 200, "OK"

    skill._send_request = mock_send
    # Use 0 base delay for fast tests
    skill._state["config"]["base_delay_seconds"] = 0.01
    skill._state["config"]["max_delay_seconds"] = 0.01

    result = await skill.execute("deliver", {
        "url": "https://example.com/hook",
        "payload": {"data": 1},
        "max_retries": 3,
    })
    assert result.success
    assert call_count == 3
    assert result.data["attempts"] == 3


@pytest.mark.asyncio
async def test_deliver_all_retries_fail(skill):
    """Returns failure after all retries exhausted."""
    async def mock_send(url, body, headers, timeout):
        return 500, "Server Error"

    skill._send_request = mock_send
    skill._state["config"]["base_delay_seconds"] = 0.01

    result = await skill.execute("deliver", {
        "url": "https://example.com/hook",
        "payload": {"data": 1},
        "max_retries": 1,
    })
    assert not result.success
    assert result.data["status"] == "failed"
    assert result.data["attempts"] == 2  # initial + 1 retry


@pytest.mark.asyncio
async def test_deliver_with_signing(skill):
    """Webhook includes HMAC signature header when secret provided."""
    captured_headers = {}
    async def mock_send(url, body, headers, timeout):
        captured_headers.update(headers)
        return 200, "OK"

    skill._send_request = mock_send
    result = await skill.execute("deliver", {
        "url": "https://example.com/hook",
        "payload": {"data": 1},
        "signing_secret": "mysecret",
    })
    assert result.success
    assert "X-Webhook-Signature" in captured_headers
    assert captured_headers["X-Webhook-Signature"].startswith("sha256=")


@pytest.mark.asyncio
async def test_deliver_idempotency(skill):
    """Duplicate delivery with same idempotency key returns cached result."""
    async def mock_send(url, body, headers, timeout):
        return 200, "OK"

    skill._send_request = mock_send
    result1 = await skill.execute("deliver", {
        "url": "https://example.com/hook",
        "payload": {"data": 1},
        "idempotency_key": "idem_123",
    })
    assert result1.success

    result2 = await skill.execute("deliver", {
        "url": "https://example.com/hook",
        "payload": {"data": 1},
        "idempotency_key": "idem_123",
    })
    assert result2.success
    assert "Already delivered" in result2.message


@pytest.mark.asyncio
async def test_deliver_missing_params(skill):
    result = await skill.execute("deliver", {"url": "https://x.com"})
    assert not result.success
    assert "required" in result.message


# ── Status ───────────────────────────────────────

@pytest.mark.asyncio
async def test_status(skill):
    async def mock_send(url, body, headers, timeout):
        return 200, "OK"
    skill._send_request = mock_send

    r = await skill.execute("deliver", {
        "url": "https://example.com/hook",
        "payload": {"x": 1},
    })
    delivery_id = r.data["delivery_id"]

    status = await skill.execute("status", {"delivery_id": delivery_id})
    assert status.success
    assert status.data["status"] == "delivered"


# ── History ──────────────────────────────────────

@pytest.mark.asyncio
async def test_history(skill):
    async def mock_send(url, body, headers, timeout):
        return 200, "OK"
    skill._send_request = mock_send

    for i in range(3):
        await skill.execute("deliver", {
            "url": f"https://example.com/hook{i}",
            "payload": {"i": i},
        })

    result = await skill.execute("history", {"limit": 10})
    assert result.success
    assert len(result.data["deliveries"]) == 3


@pytest.mark.asyncio
async def test_history_filter_by_status(skill):
    async def mock_send(url, body, headers, timeout):
        return 500, "fail"
    skill._send_request = mock_send
    skill._state["config"]["base_delay_seconds"] = 0.01
    skill._state["config"]["max_retries"] = 0

    await skill.execute("deliver", {
        "url": "https://example.com/hook",
        "payload": {"x": 1},
        "max_retries": 0,
    })

    result = await skill.execute("history", {"status": "failed"})
    assert result.success
    assert len(result.data["deliveries"]) == 1


# ── Stats ────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats(skill):
    async def mock_send(url, body, headers, timeout):
        return 200, "OK"
    skill._send_request = mock_send

    await skill.execute("deliver", {"url": "https://a.com/h", "payload": {}})
    await skill.execute("deliver", {"url": "https://a.com/h", "payload": {}})

    result = await skill.execute("stats", {})
    assert result.success
    assert result.data["total_attempted"] == 2
    assert result.data["total_delivered"] == 2
    assert result.data["success_rate_pct"] == 100.0


# ── Configure ────────────────────────────────────

@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {"key": "max_retries", "value": 5})
    assert result.success
    assert skill._state["config"]["max_retries"] == 5


@pytest.mark.asyncio
async def test_configure_invalid_key(skill):
    result = await skill.execute("configure", {"key": "invalid_key", "value": 1})
    assert not result.success


# ── Pending ──────────────────────────────────────

@pytest.mark.asyncio
async def test_pending(skill):
    async def mock_send(url, body, headers, timeout):
        return 500, "fail"
    skill._send_request = mock_send
    skill._state["config"]["base_delay_seconds"] = 0.01

    await skill.execute("deliver", {
        "url": "https://fail.com/h", "payload": {}, "max_retries": 0,
    })

    result = await skill.execute("pending", {})
    assert result.success
    assert len(result.data["deliveries"]) == 1


# ── Unknown action ───────────────────────────────

@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
