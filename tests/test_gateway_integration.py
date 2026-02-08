"""Tests for API Gateway integration with ServiceAPI."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.service_api import (
    ServiceAPI, create_app, HAS_FASTAPI,
)
from singularity.skills.base import SkillResult

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


def make_mock_gateway(allowed=True, key_id="key_abc", owner="test_owner"):
    """Create a mock APIGatewaySkill."""
    gateway = AsyncMock()

    async def mock_execute(action, params):
        if action == "check_access":
            if allowed:
                return SkillResult(
                    success=True, message="Access granted",
                    data={"allowed": True, "key_id": key_id, "owner": owner,
                           "scopes": ["skills:read", "tasks:write"],
                           "remaining_rate": 59, "remaining_daily": 999},
                )
            else:
                return SkillResult(
                    success=False, message="Invalid API key",
                    data={"allowed": False, "reason": "invalid_key"},
                )
        elif action == "record_usage":
            return SkillResult(success=True, message="Usage recorded",
                               data={"key_id": params.get("key_id"), "total_requests": 1})
        elif action == "get_billing":
            return SkillResult(success=True, message="Billing",
                               data={"total_revenue": 10.0, "total_cost": 2.0,
                                     "total_profit": 8.0, "active_keys": 1})
        elif action == "get_usage":
            return SkillResult(success=True, message="Usage",
                               data={"key_id": params.get("key_id"), "total_requests": 42})
        elif action == "list_keys":
            return SkillResult(success=True, message="Keys",
                               data={"keys": [{"key_id": key_id, "name": "test"}], "total": 1})
        elif action == "create_key":
            return SkillResult(success=True, message="Key created",
                               data={"key_id": "key_new", "api_key": "sg_test123"})
        elif action == "revoke_key":
            return SkillResult(success=True, message="Key revoked",
                               data={"key_id": params.get("key_id"), "revoked_at": "2026-01-01"})
        return SkillResult(success=False, message=f"Unknown action: {action}")

    gateway.execute = mock_execute
    return gateway


def make_mock_skill():
    """Create a mock skill."""
    action = MagicMock()
    action.name = "bash"
    action.description = "Run bash"
    action.parameters = {"command": {"type": "string"}}
    manifest = MagicMock()
    manifest.skill_id = "shell"
    manifest.name = "Shell"
    manifest.description = "Shell skill"
    manifest.actions = [action]
    skill = AsyncMock()
    skill.manifest = manifest
    skill.execute = AsyncMock(return_value=SkillResult(
        success=True, data={"output": "hello"}, message="ok"
    ))
    return skill


def make_mock_agent(include_gateway=False):
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.ticker = "TEST"
    agent.agent_type = "general"
    agent.balance = 50.0
    agent.cycle = 10
    agent.running = True
    skill = make_mock_skill()
    skills_dict = {"shell": skill}
    if include_gateway:
        skills_dict["api_gateway"] = make_mock_gateway()
    agent.skills = MagicMock()
    agent.skills.skills = skills_dict
    agent.skills.get = lambda sid: skills_dict.get(sid)
    return agent


class TestServiceAPIGatewayInit:
    """Test ServiceAPI initialization with gateway."""

    def test_init_with_gateway(self):
        gw = make_mock_gateway()
        svc = ServiceAPI(api_gateway=gw)
        assert svc.api_gateway is gw
        assert svc.require_auth is True

    def test_init_without_gateway(self):
        svc = ServiceAPI()
        assert svc.api_gateway is None
        assert svc.require_auth is False

    def test_gateway_forces_auth(self):
        svc = ServiceAPI(api_gateway=make_mock_gateway(), require_auth=False)
        assert svc.require_auth is True


class TestGatewayValidation:
    """Test validate_via_gateway method."""

    @pytest.mark.asyncio
    async def test_validate_allowed(self):
        gw = make_mock_gateway(allowed=True)
        svc = ServiceAPI(api_gateway=gw)
        result = await svc.validate_via_gateway("sg_testkey")
        assert result["allowed"] is True
        assert result["key_id"] == "key_abc"
        assert result["owner"] == "test_owner"

    @pytest.mark.asyncio
    async def test_validate_denied(self):
        gw = make_mock_gateway(allowed=False)
        svc = ServiceAPI(api_gateway=gw)
        result = await svc.validate_via_gateway("bad_key")
        assert result["allowed"] is False
        assert result["reason"] == "invalid_key"

    @pytest.mark.asyncio
    async def test_validate_no_gateway(self):
        svc = ServiceAPI()
        result = await svc.validate_via_gateway("any_key")
        assert result["allowed"] is False
        assert result["reason"] == "no_gateway"


class TestGatewayUsageTracking:
    """Test record_gateway_usage method."""

    @pytest.mark.asyncio
    async def test_record_usage(self):
        gw = make_mock_gateway()
        svc = ServiceAPI(api_gateway=gw)
        await svc.record_gateway_usage("key_abc", "tasks/shell/bash")
        # Should not raise

    @pytest.mark.asyncio
    async def test_record_usage_no_gateway(self):
        svc = ServiceAPI()
        await svc.record_gateway_usage("key_abc", "tasks/shell/bash")
        # Should not raise even without gateway


class TestHealthEndpointGateway:
    """Test health endpoint includes gateway status."""

    def test_health_with_gateway(self):
        gw = make_mock_gateway()
        svc = ServiceAPI(api_gateway=gw)
        h = svc.health()
        assert h["api_gateway"]["enabled"] is True

    def test_health_without_gateway(self):
        svc = ServiceAPI()
        h = svc.health()
        assert h["api_gateway"]["enabled"] is False


class TestCreateAppAutoDetect:
    """Test that create_app auto-detects APIGatewaySkill from agent."""

    def test_auto_detect_gateway(self):
        agent = make_mock_agent(include_gateway=True)
        app = create_app(agent=agent)
        assert app.state.service.api_gateway is not None

    def test_no_auto_detect_without_skill(self):
        agent = make_mock_agent(include_gateway=False)
        app = create_app(agent=agent)
        assert app.state.service.api_gateway is None

    def test_explicit_gateway_overrides(self):
        agent = make_mock_agent(include_gateway=False)
        gw = make_mock_gateway()
        app = create_app(agent=agent, api_gateway=gw)
        assert app.state.service.api_gateway is gw


class TestHTTPGatewayEndpoints:
    """Test HTTP endpoints with gateway integration using TestClient."""

    @pytest.fixture
    def client_with_gateway(self):
        from httpx import AsyncClient, ASGITransport
        gw = make_mock_gateway()
        agent = make_mock_agent()
        app = create_app(agent=agent, api_gateway=gw)
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.fixture
    def client_no_gateway(self):
        from httpx import AsyncClient, ASGITransport
        agent = make_mock_agent()
        app = create_app(agent=agent, api_keys=["test123"])
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_gateway_auth_success(self, client_with_gateway):
        async with client_with_gateway as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["api_gateway"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_gateway_auth_required(self, client_with_gateway):
        async with client_with_gateway as client:
            resp = await client.get("/capabilities")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_gateway_auth_valid_key(self, client_with_gateway):
        async with client_with_gateway as client:
            resp = await client.get("/capabilities",
                                     headers={"Authorization": "Bearer sg_validkey"})
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_gateway_auth_invalid_key(self, client_with_gateway):
        from httpx import AsyncClient, ASGITransport
        gw = make_mock_gateway(allowed=False)
        agent = make_mock_agent()
        app = create_app(agent=agent, api_gateway=gw)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/capabilities",
                                     headers={"Authorization": "Bearer bad_key"})
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_billing_endpoint(self, client_with_gateway):
        async with client_with_gateway as client:
            resp = await client.get("/billing",
                                     headers={"Authorization": "Bearer sg_validkey"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_revenue"] == 10.0

    @pytest.mark.asyncio
    async def test_usage_endpoint(self, client_with_gateway):
        async with client_with_gateway as client:
            resp = await client.get("/usage/key_abc",
                                     headers={"Authorization": "Bearer sg_validkey"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_requests"] == 42

    @pytest.mark.asyncio
    async def test_keys_endpoint(self, client_with_gateway):
        async with client_with_gateway as client:
            resp = await client.get("/keys",
                                     headers={"Authorization": "Bearer sg_validkey"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_billing_without_gateway(self, client_no_gateway):
        async with client_no_gateway as client:
            resp = await client.get("/billing",
                                     headers={"Authorization": "Bearer test123"})
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_simple_auth_still_works(self, client_no_gateway):
        async with client_no_gateway as client:
            resp = await client.get("/capabilities",
                                     headers={"Authorization": "Bearer test123"})
            assert resp.status_code == 200
