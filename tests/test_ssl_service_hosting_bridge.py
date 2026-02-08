#!/usr/bin/env python3
"""Tests for SSLServiceHostingBridgeSkill."""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch
from singularity.skills.ssl_service_hosting_bridge import (
    SSLServiceHostingBridgeSkill,
    BRIDGE_STATE_FILE, SSL_DATA_FILE, SERVICES_DATA_FILE,
    DEFAULT_POLICY,
)


@pytest.fixture
def skill():
    s = SSLServiceHostingBridgeSkill()
    s._state = s._default_state()
    return s


@pytest.fixture
def sample_services():
    return {
        "services": {
            "eve-code-review": {
                "service_name": "Code Review",
                "agent_name": "eve",
                "domain": "eve-code-review.singularity.wisent.ai",
                "status": "active",
            },
            "adam-api": {
                "service_name": "General API",
                "agent_name": "adam",
                "domain": "adam-api.singularity.wisent.ai",
                "status": "active",
            },
            "bob-internal": {
                "service_name": "Internal Tool",
                "agent_name": "bob",
                "status": "active",
                # No domain assigned
            },
        },
        "routing_rules": {},
        "domain_assignments": {},
    }


@pytest.fixture
def sample_ssl_data():
    now = datetime.utcnow()
    return {
        "certificates": {
            "cert_abc123": {
                "cert_id": "cert_abc123",
                "domain": "eve-code-review.singularity.wisent.ai",
                "provider": "letsencrypt",
                "status": "active",
                "expires_at": (now + timedelta(days=60)).isoformat(),
            },
        },
        "domains": {
            "eve-code-review.singularity.wisent.ai": {
                "active_cert_id": "cert_abc123",
            },
        },
    }


@pytest.mark.asyncio
async def test_auto_provision_new_service(skill, sample_services):
    """Auto-provision SSL for a service without a certificate."""
    with patch.object(skill, '_load_services_data', return_value=sample_services), \
         patch.object(skill, '_load_ssl_data', return_value={"certificates": {}, "domains": {}}), \
         patch.object(skill, '_provision_cert_record') as mock_prov, \
         patch.object(skill, '_save_state'):
        mock_prov.return_value = {"cert_id": "cert_new123", "domain": "adam-api.singularity.wisent.ai", "status": "active"}
        result = await skill.execute("auto_provision", {"service_id": "adam-api"})
    assert result.success
    assert "provisioned" in result.message.lower() or "provisioned" in result.data.get("action", "")
    mock_prov.assert_called_once()


@pytest.mark.asyncio
async def test_auto_provision_already_covered(skill, sample_services, sample_ssl_data):
    """Skip provisioning when service already has active SSL."""
    with patch.object(skill, '_load_services_data', return_value=sample_services), \
         patch.object(skill, '_load_ssl_data', return_value=sample_ssl_data):
        result = await skill.execute("auto_provision", {"service_id": "eve-code-review"})
    assert result.success
    assert result.data["action"] == "already_covered"


@pytest.mark.asyncio
async def test_auto_provision_no_domain(skill, sample_services):
    """Fail gracefully when service has no domain."""
    with patch.object(skill, '_load_services_data', return_value=sample_services):
        result = await skill.execute("auto_provision", {"service_id": "bob-internal"})
    assert not result.success
    assert "no domain" in result.message.lower()


@pytest.mark.asyncio
async def test_auto_provision_disabled(skill, sample_services):
    """Respect disabled auto_provision policy."""
    skill._state["policy"]["auto_provision"] = False
    result = await skill.execute("auto_provision", {"service_id": "adam-api"})
    assert not result.success
    assert "disabled" in result.message.lower()


@pytest.mark.asyncio
async def test_bulk_secure(skill, sample_services, sample_ssl_data):
    """Bulk secure provisions missing certs and skips covered ones."""
    with patch.object(skill, '_load_services_data', return_value=sample_services), \
         patch.object(skill, '_load_ssl_data', return_value=sample_ssl_data), \
         patch.object(skill, '_provision_cert_record') as mock_prov, \
         patch.object(skill, '_save_state'):
        mock_prov.return_value = {"cert_id": "cert_bulk1", "domain": "adam-api.singularity.wisent.ai", "status": "active"}
        result = await skill.execute("bulk_secure", {})
    assert result.success
    assert result.data["already_covered"] == 1  # eve
    assert result.data["provisioned"] == 1       # adam
    assert result.data["no_domain"] == 1          # bob


@pytest.mark.asyncio
async def test_coverage_report(skill, sample_services, sample_ssl_data):
    """Coverage report shows correct percentages."""
    with patch.object(skill, '_load_services_data', return_value=sample_services), \
         patch.object(skill, '_load_ssl_data', return_value=sample_ssl_data), \
         patch.object(skill, '_save_state'):
        result = await skill.execute("coverage_report", {})
    assert result.success
    assert result.data["total_services"] == 3
    assert result.data["covered"] == 1   # eve has cert
    assert result.data["uncovered"] == 2  # adam + bob


@pytest.mark.asyncio
async def test_renewal_check_healthy(skill, sample_services, sample_ssl_data):
    """Renewal check finds healthy certs (not expiring soon)."""
    with patch.object(skill, '_load_services_data', return_value=sample_services), \
         patch.object(skill, '_load_ssl_data', return_value=sample_ssl_data), \
         patch.object(skill, '_save_state'):
        result = await skill.execute("renewal_check", {"threshold_days": 30})
    assert result.success
    assert len(result.data["healthy"]) == 1
    assert len(result.data["uncovered"]) == 2


@pytest.mark.asyncio
async def test_renewal_check_expiring(skill, sample_services):
    """Renewal check triggers renewal for expiring certs."""
    soon = (datetime.utcnow() + timedelta(days=5)).isoformat()
    ssl_data = {
        "certificates": {"cert_exp1": {"cert_id": "cert_exp1", "domain": "eve-code-review.singularity.wisent.ai",
                                        "provider": "letsencrypt", "status": "active", "expires_at": soon}},
        "domains": {"eve-code-review.singularity.wisent.ai": {"active_cert_id": "cert_exp1"}},
    }
    with patch.object(skill, '_load_services_data', return_value=sample_services), \
         patch.object(skill, '_load_ssl_data', return_value=ssl_data), \
         patch.object(skill, '_provision_cert_record') as mock_prov, \
         patch.object(skill, '_save_state'):
        mock_prov.return_value = {"cert_id": "cert_renewed1", "status": "active"}
        result = await skill.execute("renewal_check", {"threshold_days": 30})
    assert result.success
    assert len(result.data["renewed"]) == 1
    assert result.data["renewed"][0]["old_cert_id"] == "cert_exp1"


@pytest.mark.asyncio
async def test_configure_policy(skill):
    """Update SSL policy settings."""
    with patch.object(skill, '_save_state'):
        result = await skill.execute("configure_policy", {
            "preferred_provider": "self_signed",
            "use_wildcard": True,
            "wildcard_base_domain": "singularity.wisent.ai",
        })
    assert result.success
    assert skill._state["policy"]["preferred_provider"] == "self_signed"
    assert skill._state["policy"]["use_wildcard"] is True


@pytest.mark.asyncio
async def test_configure_policy_invalid_provider(skill):
    """Reject invalid provider."""
    result = await skill.execute("configure_policy", {"preferred_provider": "invalid"})
    assert not result.success


@pytest.mark.asyncio
async def test_service_ssl_status(skill, sample_services, sample_ssl_data):
    """Get SSL status for a specific service."""
    with patch.object(skill, '_load_services_data', return_value=sample_services), \
         patch.object(skill, '_load_ssl_data', return_value=sample_ssl_data):
        result = await skill.execute("service_ssl_status", {"service_id": "eve-code-review"})
    assert result.success
    assert result.data["ssl_status"] == "active"


@pytest.mark.asyncio
async def test_provisions_log(skill):
    """View provisioning history."""
    skill._state["provisions"] = [
        {"service_id": f"svc-{i}", "cert_id": f"cert-{i}", "timestamp": "2026-01-01"}
        for i in range(5)
    ]
    result = await skill.execute("provisions_log", {"limit": 3})
    assert result.success
    assert result.data["showing"] == 3
    assert result.data["total"] == 5


@pytest.mark.asyncio
async def test_status(skill, sample_services, sample_ssl_data):
    """Get bridge status overview."""
    with patch.object(skill, '_load_services_data', return_value=sample_services), \
         patch.object(skill, '_load_ssl_data', return_value=sample_ssl_data):
        result = await skill.execute("status", {})
    assert result.success
    assert "stats" in result.data
    assert "policy" in result.data
