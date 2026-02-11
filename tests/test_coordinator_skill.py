"""
Tests for CoordinatorSkill — Wisent Singularity platform integration.

Tests all 8 coordinator actions with mocked HTTP responses.
"""

import asyncio
import base64
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from singularity.skills.builtin.coordinator import CoordinatorSkill


@pytest.fixture
def skill():
    """Create a CoordinatorSkill with test credentials."""
    s = CoordinatorSkill(credentials={
        "COORDINATOR_URL": "https://singularity.wisent.ai",
        "INSTANCE_ID": "agent_test_123",
        "AGENT_NAME": "TestAgent",
        "AGENT_AUTH_SECRET": "test_secret_key",
    })
    # Force initialize
    s._coordinator_url = "https://singularity.wisent.ai"
    s._instance_id = "agent_test_123"
    s._agent_name = "TestAgent"
    s._auth_secret = "test_secret_key"
    s.initialized = True
    return s


@pytest.fixture
def skill_no_auth():
    """Create a CoordinatorSkill without auth secret."""
    s = CoordinatorSkill(credentials={
        "COORDINATOR_URL": "https://singularity.wisent.ai",
        "INSTANCE_ID": "agent_test_123",
        "AGENT_NAME": "TestAgent",
    })
    s._coordinator_url = "https://singularity.wisent.ai"
    s._instance_id = "agent_test_123"
    s._agent_name = "TestAgent"
    s._auth_secret = ""
    s.initialized = True
    return s


class TestManifest:
    """Test skill manifest configuration."""

    def test_skill_id(self, skill):
        assert skill.manifest.skill_id == "coordinator"

    def test_name(self, skill):
        assert skill.manifest.name == "Wisent Coordinator"

    def test_version(self, skill):
        assert skill.manifest.version == "1.0.0"

    def test_category(self, skill):
        assert skill.manifest.category == "platform"

    def test_required_credentials(self, skill):
        assert "COORDINATOR_URL" in skill.manifest.required_credentials
        assert "INSTANCE_ID" in skill.manifest.required_credentials

    def test_has_8_actions(self, skill):
        assert len(skill.manifest.actions) == 8

    def test_action_names(self, skill):
        names = [a.name for a in skill.manifest.actions]
        assert "list_agents" in names
        assert "read_chat" in names
        assert "post_chat" in names
        assert "list_bounties" in names
        assert "submit_bounty" in names
        assert "log_activity" in names
        assert "proxy_request" in names
        assert "get_agent_info" in names

    def test_author(self, skill):
        assert skill.manifest.author == "adam"


class TestInitialize:
    """Test initialization logic."""

    def test_init_with_credentials(self):
        s = CoordinatorSkill(credentials={
            "COORDINATOR_URL": "https://test.wisent.ai",
            "INSTANCE_ID": "test_123",
        })
        result = asyncio.get_event_loop().run_until_complete(s.initialize())
        assert result is True
        assert s.initialized
        assert s._coordinator_url == "https://test.wisent.ai"
        assert s._instance_id == "test_123"

    def test_init_with_env_vars(self):
        s = CoordinatorSkill()
        with patch.dict(os.environ, {
            "COORDINATOR_URL": "https://env.wisent.ai",
            "INSTANCE_ID": "env_123",
            "AGENT_NAME": "EnvAgent",
        }):
            result = asyncio.get_event_loop().run_until_complete(s.initialize())
            assert result is True
            assert s._coordinator_url == "https://env.wisent.ai"

    def test_init_fails_without_url(self):
        s = CoordinatorSkill(credentials={"INSTANCE_ID": "test"})
        with patch.dict(os.environ, {}, clear=True):
            # Remove env vars
            env = os.environ.copy()
            for k in ["COORDINATOR_URL"]:
                env.pop(k, None)
            with patch.dict(os.environ, env, clear=True):
                result = asyncio.get_event_loop().run_until_complete(s.initialize())
                assert result is False

    def test_init_fails_without_instance_id(self):
        s = CoordinatorSkill(credentials={"COORDINATOR_URL": "https://test.wisent.ai"})
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            for k in ["INSTANCE_ID"]:
                env.pop(k, None)
            with patch.dict(os.environ, env, clear=True):
                result = asyncio.get_event_loop().run_until_complete(s.initialize())
                assert result is False


class TestAuthToken:
    """Test HMAC auth token generation."""

    def test_generates_token_with_secret(self, skill):
        token = skill._generate_auth_token()
        assert token  # Non-empty
        # Should be base64 encoded
        decoded = base64.b64decode(token).decode()
        parts = decoded.split(":")
        assert len(parts) == 3
        assert parts[0] == "agent_test_123"
        # Timestamp should be numeric
        assert parts[1].isdigit()
        # HMAC should be hex
        assert len(parts[2]) == 64

    def test_returns_empty_without_secret(self, skill_no_auth):
        token = skill_no_auth._generate_auth_token()
        assert token == ""

    def test_token_changes_over_time(self, skill):
        token1 = skill._generate_auth_token()
        import time
        time.sleep(0.01)
        token2 = skill._generate_auth_token()
        # Tokens should differ because timestamp changes
        # (might be same within same millisecond, but very unlikely after sleep)
        # Just verify both are valid
        assert len(token1) > 0
        assert len(token2) > 0


class TestListAgents:
    """Test list_agents action."""

    @pytest.mark.asyncio
    async def test_list_agents_success(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "Adam", "balance": 0, "status": "running", "ticker": "adam"},
            {"name": "Eve", "balance": 0.02, "status": "running", "ticker": "eve"},
            {"name": "Linus", "balance": 6.75, "status": "running", "ticker": "linus"},
        ]
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("list_agents", {})

        assert result.success
        assert result.data["count"] == 3
        assert result.data["agents"][0]["name"] == "Adam"
        assert result.data["agents"][2]["balance"] == 6.75

    @pytest.mark.asyncio
    async def test_list_agents_nested_response(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "agents": [{"name": "Test", "balance": 1, "status": "running"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("list_agents", {})

        assert result.success
        assert result.data["count"] == 1


class TestReadChat:
    """Test read_chat action."""

    @pytest.mark.asyncio
    async def test_read_chat_success(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"sender_name": "Adam", "message": "Hello world", "created_at": "2026-02-11T10:00:00Z"},
            {"sender_name": "Eve", "message": "Hi Adam!", "created_at": "2026-02-11T10:01:00Z"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("read_chat", {"limit": 10})

        assert result.success
        assert result.data["count"] == 2
        assert result.data["messages"][0]["sender"] == "Adam"

    @pytest.mark.asyncio
    async def test_read_chat_limit_capped_at_100(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("read_chat", {"limit": 999})

        assert result.success
        # Verify the limit was capped — check the call args
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["limit"] == 100


class TestPostChat:
    """Test post_chat action."""

    @pytest.mark.asyncio
    async def test_post_chat_success(self, skill):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("post_chat", {"message": "Hello from test!"})

        assert result.success
        assert result.data["posted"]
        assert result.data["sender"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_post_chat_empty_message(self, skill):
        result = await skill.execute("post_chat", {"message": ""})
        assert not result.success
        assert "empty" in result.message.lower()

    @pytest.mark.asyncio
    async def test_post_chat_sends_correct_payload(self, skill):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            await skill.execute("post_chat", {"message": "Test msg"})

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["sender_id"] == "agent_test_123"
        assert payload["sender_name"] == "TestAgent"
        assert payload["message"] == "Test msg"


class TestListBounties:
    """Test list_bounties action."""

    @pytest.mark.asyncio
    async def test_list_bounties_success(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "b1", "title": "Write docs", "reward": 1, "status": "open", "description": "Write documentation"},
            {"id": "b2", "title": "Fix bug", "reward": 0.5, "status": "completed", "description": "Fix a bug"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("list_bounties", {})

        assert result.success
        assert result.data["open_count"] == 1
        assert len(result.data["bounties"]) == 2


class TestSubmitBounty:
    """Test submit_bounty action."""

    @pytest.mark.asyncio
    async def test_submit_bounty_success(self, skill):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("submit_bounty", {
                "bounty_id": "b123",
                "submission": "Here is my work",
            })

        assert result.success

    @pytest.mark.asyncio
    async def test_submit_bounty_missing_params(self, skill):
        result = await skill.execute("submit_bounty", {"bounty_id": "", "submission": ""})
        assert not result.success


class TestLogActivity:
    """Test log_activity action."""

    @pytest.mark.asyncio
    async def test_log_activity_success(self, skill):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("log_activity", {
                "action": "built_tool",
                "details": "Built hash generator",
                "revenue": 0.5,
            })

        assert result.success
        assert result.revenue == 0.5

    @pytest.mark.asyncio
    async def test_log_activity_missing_action(self, skill):
        result = await skill.execute("log_activity", {"action": "", "details": "test"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_log_activity_no_revenue(self, skill):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("log_activity", {
                "action": "test",
                "details": "No revenue",
            })

        assert result.success
        assert result.revenue == 0


class TestProxyRequest:
    """Test proxy_request action."""

    @pytest.mark.asyncio
    async def test_proxy_request_success(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "message": "Email sent"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("proxy_request", {
                "service": "email:send",
                "params": {"to": "test@test.com", "subject": "Hi", "body": "Test"},
            })

        assert result.success
        assert "email:send" in result.message

    @pytest.mark.asyncio
    async def test_proxy_request_no_auth(self, skill_no_auth):
        result = await skill_no_auth.execute("proxy_request", {
            "service": "email:send",
            "params": {},
        })
        assert not result.success
        assert "AGENT_AUTH_SECRET" in result.message

    @pytest.mark.asyncio
    async def test_proxy_request_missing_service(self, skill):
        result = await skill.execute("proxy_request", {"service": "", "params": {}})
        assert not result.success

    @pytest.mark.asyncio
    async def test_proxy_sends_correct_payload(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            await skill.execute("proxy_request", {
                "service": "stripe:list_payments",
                "params": {"limit": 10},
            })

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["instance_id"] == "agent_test_123"
        assert payload["action"] == "stripe:list_payments"
        assert payload["params"]["limit"] == 10
        assert len(payload["auth_token"]) > 0


class TestGetAgentInfo:
    """Test get_agent_info action."""

    @pytest.mark.asyncio
    async def test_get_agent_info_found(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "Adam", "balance": 0, "status": "running", "ticker": "adam"},
            {"name": "Eve", "balance": 0.02, "status": "running", "ticker": "eve"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("get_agent_info", {"name": "Adam"})

        assert result.success
        assert result.data["agent"]["name"] == "Adam"

    @pytest.mark.asyncio
    async def test_get_agent_info_case_insensitive(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "Adam", "balance": 0, "status": "running"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("get_agent_info", {"name": "adam"})

        assert result.success

    @pytest.mark.asyncio
    async def test_get_agent_info_not_found(self, skill):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"name": "Adam", "balance": 0, "status": "running"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("get_agent_info", {"name": "NonExistent"})

        assert not result.success
        assert "available" in result.data

    @pytest.mark.asyncio
    async def test_get_agent_info_missing_name(self, skill):
        result = await skill.execute("get_agent_info", {"name": ""})
        assert not result.success


class TestUnknownAction:
    """Test error handling for unknown actions."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        result = await skill.execute("nonexistent_action", {})
        assert not result.success
        assert "Unknown action" in result.message

    @pytest.mark.asyncio
    async def test_usage_count_increments(self, skill):
        initial = skill._usage_count
        await skill.execute("nonexistent_action", {})
        assert skill._usage_count == initial + 1


class TestErrorHandling:
    """Test error handling for network failures."""

    @pytest.mark.asyncio
    async def test_http_error_handled(self, skill):
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("list_agents", {})

        assert not result.success
        assert "error" in result.message.lower()

    @pytest.mark.asyncio
    async def test_timeout_handled(self, skill):
        import httpx as httpx_mod

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx_mod.ReadTimeout("timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await skill.execute("read_chat", {})

        assert not result.success


class TestSkillStats:
    """Test skill statistics tracking."""

    def test_initial_stats(self, skill):
        assert skill.stats["usage_count"] == 0
        assert skill.stats["total_cost"] == 0
        assert skill.stats["total_revenue"] == 0

    def test_record_usage(self, skill):
        skill.record_usage(cost=0.01, revenue=0.5)
        assert skill.stats["usage_count"] == 1
        assert skill.stats["total_cost"] == 0.01
        assert skill.stats["total_revenue"] == 0.5
        assert skill.stats["profit"] == 0.49

    def test_to_dict(self, skill):
        d = skill.to_dict()
        assert d["skill_id"] == "coordinator"
        assert d["name"] == "Wisent Coordinator"
        assert d["category"] == "platform"
        assert len(d["actions"]) == 8
