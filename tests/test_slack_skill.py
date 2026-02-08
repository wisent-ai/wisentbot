"""Tests for SlackSkill - Slack API integration."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from singularity.skills.slack import SlackSkill
from singularity.skills.base import SkillResult


@pytest.fixture
def creds():
    return {"SLACK_BOT_TOKEN": "xoxb-test-token-123"}


@pytest.fixture
def skill(creds):
    return SlackSkill(credentials=creds)


@pytest.fixture
def skill_no_creds():
    return SlackSkill(credentials={})


def _make_response(data: dict, status_code: int = 200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


# ==================== Credential Tests ====================


@pytest.mark.asyncio
async def test_missing_credentials(skill_no_creds):
    """Skill should fail when SLACK_BOT_TOKEN is missing."""
    result = await skill_no_creds.execute("send_message", {"channel": "C123", "text": "hi"})
    assert not result.success
    assert "Missing credentials" in result.message
    assert "SLACK_BOT_TOKEN" in result.message


def test_check_credentials_valid(skill):
    """Credential check should pass with a valid token."""
    assert skill.check_credentials() is True


def test_check_credentials_missing(skill_no_creds):
    """Credential check should fail without a token."""
    assert skill_no_creds.check_credentials() is False
    missing = skill_no_creds.get_missing_credentials()
    assert "SLACK_BOT_TOKEN" in missing


# ==================== Manifest Tests ====================


def test_manifest(skill):
    """Manifest should have correct metadata and all actions."""
    m = skill.manifest
    assert m.skill_id == "slack"
    assert m.category == "social"
    assert m.author == "eve"
    assert len(m.actions) == 8
    action_names = [a.name for a in m.actions]
    assert "send_message" in action_names
    assert "get_messages" in action_names
    assert "search_messages" in action_names
    assert "react" in action_names
    assert "upload_file" in action_names
    assert "create_channel" in action_names
    assert "get_user_info" in action_names
    assert "set_channel_topic" in action_names


# ==================== Send Message Tests ====================


@pytest.mark.asyncio
async def test_send_message_success(skill):
    """Successfully send a message to a channel."""
    mock_resp = _make_response({
        "ok": True,
        "channel": "C123",
        "ts": "1234567890.123456",
        "message": {"text": "Hello world"},
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("send_message", {
        "channel": "C123",
        "text": "Hello world",
    })

    assert result.success
    assert result.data["channel"] == "C123"
    assert result.data["ts"] == "1234567890.123456"
    assert result.cost == 0.001
    assert "C123" in result.message

    # Verify correct API endpoint was called
    call_args = skill.http.post.call_args
    assert "chat.postMessage" in call_args[0][0]
    payload = call_args[1]["json"]
    assert payload["channel"] == "C123"
    assert payload["text"] == "Hello world"


@pytest.mark.asyncio
async def test_send_message_with_thread(skill):
    """Send a threaded reply."""
    mock_resp = _make_response({
        "ok": True,
        "channel": "C123",
        "ts": "1234567890.999999",
        "message": {"text": "thread reply"},
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("send_message", {
        "channel": "C123",
        "text": "thread reply",
        "thread_ts": "1234567890.123456",
    })

    assert result.success
    payload = skill.http.post.call_args[1]["json"]
    assert payload["thread_ts"] == "1234567890.123456"


@pytest.mark.asyncio
async def test_send_message_failure(skill):
    """Handle Slack API error when sending a message."""
    mock_resp = _make_response({"ok": False, "error": "channel_not_found"})
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("send_message", {
        "channel": "C_INVALID",
        "text": "Hello",
    })

    assert not result.success
    assert "channel_not_found" in result.message


@pytest.mark.asyncio
async def test_send_message_missing_params(skill):
    """Fail gracefully when required parameters are missing."""
    result = await skill.execute("send_message", {"channel": "C123"})
    assert not result.success
    assert "required" in result.message.lower()

    result = await skill.execute("send_message", {"text": "hello"})
    assert not result.success
    assert "required" in result.message.lower()


# ==================== Get Messages Tests ====================


@pytest.mark.asyncio
async def test_get_messages_success(skill):
    """Successfully retrieve channel history."""
    mock_resp = _make_response({
        "ok": True,
        "messages": [
            {"text": "msg1", "ts": "1111111111.111111", "user": "U1"},
            {"text": "msg2", "ts": "2222222222.222222", "user": "U2"},
        ],
        "has_more": False,
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("get_messages", {"channel": "C123", "limit": 5})

    assert result.success
    assert result.data["count"] == 2
    assert len(result.data["messages"]) == 2
    assert result.data["has_more"] is False

    call_args = skill.http.post.call_args
    assert "conversations.history" in call_args[0][0]
    payload = call_args[1]["json"]
    assert payload["channel"] == "C123"
    assert payload["limit"] == 5


@pytest.mark.asyncio
async def test_get_messages_with_timestamps(skill):
    """Retrieve messages within a time range."""
    mock_resp = _make_response({
        "ok": True,
        "messages": [{"text": "in range", "ts": "1500000000.000000"}],
        "has_more": False,
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("get_messages", {
        "channel": "C123",
        "oldest": "1400000000.000000",
        "latest": "1600000000.000000",
    })

    assert result.success
    payload = skill.http.post.call_args[1]["json"]
    assert payload["oldest"] == "1400000000.000000"
    assert payload["latest"] == "1600000000.000000"


@pytest.mark.asyncio
async def test_get_messages_missing_channel(skill):
    """Fail when channel is not provided."""
    result = await skill.execute("get_messages", {})
    assert not result.success
    assert "required" in result.message.lower()


# ==================== Search Messages Tests ====================


@pytest.mark.asyncio
async def test_search_messages_success(skill):
    """Successfully search for messages."""
    mock_resp = _make_response({
        "ok": True,
        "messages": {
            "total": 3,
            "matches": [
                {"text": "match1", "ts": "111", "channel": {"id": "C1"}},
                {"text": "match2", "ts": "222", "channel": {"id": "C2"}},
                {"text": "match3", "ts": "333", "channel": {"id": "C1"}},
            ],
        },
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("search_messages", {
        "query": "deployment",
        "sort": "timestamp",
        "count": 10,
    })

    assert result.success
    assert result.data["total"] == 3
    assert len(result.data["matches"]) == 3
    assert result.data["query"] == "deployment"
    assert result.cost == 0.002

    payload = skill.http.post.call_args[1]["json"]
    assert payload["query"] == "deployment"
    assert payload["sort"] == "timestamp"
    assert payload["count"] == 10


@pytest.mark.asyncio
async def test_search_messages_no_query(skill):
    """Fail when search query is missing."""
    result = await skill.execute("search_messages", {})
    assert not result.success
    assert "required" in result.message.lower()


# ==================== React Tests ====================


@pytest.mark.asyncio
async def test_react_success(skill):
    """Successfully add a reaction."""
    mock_resp = _make_response({"ok": True})
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("react", {
        "channel": "C123",
        "timestamp": "1234567890.123456",
        "name": "thumbsup",
    })

    assert result.success
    assert result.data["emoji"] == "thumbsup"
    assert result.cost == 0.001

    payload = skill.http.post.call_args[1]["json"]
    assert payload["name"] == "thumbsup"
    assert "reactions.add" in skill.http.post.call_args[0][0]


@pytest.mark.asyncio
async def test_react_strips_colons(skill):
    """Colons around emoji name should be stripped."""
    mock_resp = _make_response({"ok": True})
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("react", {
        "channel": "C123",
        "timestamp": "1234567890.123456",
        "name": ":fire:",
    })

    assert result.success
    payload = skill.http.post.call_args[1]["json"]
    assert payload["name"] == "fire"


@pytest.mark.asyncio
async def test_react_missing_params(skill):
    """Fail when required reaction parameters are missing."""
    result = await skill.execute("react", {"channel": "C123", "timestamp": "123"})
    assert not result.success


# ==================== Upload File Tests ====================


@pytest.mark.asyncio
async def test_upload_file_success(skill):
    """Successfully upload a file."""
    mock_resp = _make_response({
        "ok": True,
        "file": {
            "id": "F123",
            "name": "report.txt",
            "url_private": "https://files.slack.com/files-pri/T123/report.txt",
            "size": 1024,
        },
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("upload_file", {
        "channels": "C123,C456",
        "content": "file contents here",
        "filename": "report.txt",
        "title": "Daily Report",
        "filetype": "text",
    })

    assert result.success
    assert result.data["file_id"] == "F123"
    assert result.data["filename"] == "report.txt"
    assert result.cost == 0.002

    payload = skill.http.post.call_args[1]["json"]
    assert payload["channels"] == "C123,C456"
    assert payload["content"] == "file contents here"
    assert payload["title"] == "Daily Report"
    assert payload["filetype"] == "text"


@pytest.mark.asyncio
async def test_upload_file_missing_params(skill):
    """Fail when required upload parameters are missing."""
    result = await skill.execute("upload_file", {
        "channels": "C123",
        "content": "data",
    })
    assert not result.success
    assert "required" in result.message.lower()


# ==================== Create Channel Tests ====================


@pytest.mark.asyncio
async def test_create_channel_success(skill):
    """Successfully create a channel."""
    mock_resp = _make_response({
        "ok": True,
        "channel": {
            "id": "C_NEW",
            "name": "project-alpha",
            "is_private": False,
        },
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("create_channel", {"name": "project-alpha"})

    assert result.success
    assert result.data["channel_id"] == "C_NEW"
    assert result.data["name"] == "project-alpha"
    assert result.cost == 0.001

    payload = skill.http.post.call_args[1]["json"]
    assert payload["name"] == "project-alpha"
    assert payload["is_private"] is False


@pytest.mark.asyncio
async def test_create_channel_private(skill):
    """Create a private channel."""
    mock_resp = _make_response({
        "ok": True,
        "channel": {"id": "C_PRIV", "name": "secret-ops", "is_private": True},
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("create_channel", {
        "name": "secret-ops",
        "is_private": True,
    })

    assert result.success
    payload = skill.http.post.call_args[1]["json"]
    assert payload["is_private"] is True


@pytest.mark.asyncio
async def test_create_channel_error(skill):
    """Handle error when channel name already taken."""
    mock_resp = _make_response({"ok": False, "error": "name_taken"})
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("create_channel", {"name": "existing-channel"})

    assert not result.success
    assert "name_taken" in result.message


# ==================== Get User Info Tests ====================


@pytest.mark.asyncio
async def test_get_user_info_success(skill):
    """Successfully get user information."""
    mock_resp = _make_response({
        "ok": True,
        "user": {
            "id": "U123",
            "name": "jdoe",
            "real_name": "Jane Doe",
            "is_admin": True,
            "is_bot": False,
            "tz": "America/New_York",
            "profile": {
                "display_name": "Jane",
                "email": "jane@example.com",
                "title": "Engineer",
            },
        },
    })
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("get_user_info", {"user_id": "U123"})

    assert result.success
    assert result.data["id"] == "U123"
    assert result.data["name"] == "jdoe"
    assert result.data["real_name"] == "Jane Doe"
    assert result.data["display_name"] == "Jane"
    assert result.data["email"] == "jane@example.com"
    assert result.data["is_admin"] is True
    assert result.data["is_bot"] is False

    payload = skill.http.post.call_args[1]["json"]
    assert payload["user"] == "U123"


@pytest.mark.asyncio
async def test_get_user_info_not_found(skill):
    """Handle user not found error."""
    mock_resp = _make_response({"ok": False, "error": "user_not_found"})
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("get_user_info", {"user_id": "U_BAD"})

    assert not result.success
    assert "user_not_found" in result.message


# ==================== Set Channel Topic Tests ====================


@pytest.mark.asyncio
async def test_set_channel_topic_success(skill):
    """Successfully set a channel topic."""
    mock_resp = _make_response({"ok": True, "topic": "Sprint 42 goals"})
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("set_channel_topic", {
        "channel": "C123",
        "topic": "Sprint 42 goals",
    })

    assert result.success
    assert result.data["topic"] == "Sprint 42 goals"
    assert result.data["channel"] == "C123"
    assert result.cost == 0.001

    payload = skill.http.post.call_args[1]["json"]
    assert payload["topic"] == "Sprint 42 goals"
    assert "conversations.setTopic" in skill.http.post.call_args[0][0]


@pytest.mark.asyncio
async def test_set_channel_topic_missing_params(skill):
    """Fail when topic or channel is missing."""
    result = await skill.execute("set_channel_topic", {"channel": "C123"})
    assert not result.success
    assert "required" in result.message.lower()


# ==================== Error Handling Tests ====================


@pytest.mark.asyncio
async def test_unknown_action(skill):
    """Return error for an unknown action name."""
    result = await skill.execute("nonexistent_action", {})
    assert not result.success
    assert "Unknown action" in result.message


@pytest.mark.asyncio
async def test_exception_handling(skill):
    """Handle unexpected exceptions gracefully."""
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(side_effect=Exception("Connection timeout"))

    result = await skill.execute("send_message", {
        "channel": "C123",
        "text": "hello",
    })

    assert not result.success
    assert "Slack error" in result.message
    assert "Connection timeout" in result.message


@pytest.mark.asyncio
async def test_rate_limit_error(skill):
    """Handle Slack rate limit response."""
    mock_resp = _make_response({"ok": False, "error": "ratelimited"})
    skill.http = AsyncMock()
    skill.http.post = AsyncMock(return_value=mock_resp)

    result = await skill.execute("send_message", {
        "channel": "C123",
        "text": "too fast",
    })

    assert not result.success
    assert "ratelimited" in result.message


# ==================== Auth Header Tests ====================


def test_get_headers(skill):
    """Verify authorization headers are correctly constructed."""
    headers = skill._get_headers()
    assert headers["Authorization"] == "Bearer xoxb-test-token-123"
    assert "application/json" in headers["Content-Type"]


# ==================== Skill Infrastructure Tests ====================


def test_to_dict(skill):
    """Verify the skill serializes correctly for LLM context."""
    d = skill.to_dict()
    assert d["skill_id"] == "slack"
    assert d["name"] == "Slack Integration"
    assert len(d["actions"]) == 8
    assert d["initialized"] is False


@pytest.mark.asyncio
async def test_initialize_with_credentials(skill):
    """Skill should initialize when credentials are present."""
    result = await skill.initialize()
    assert result is True
    assert skill.initialized is True


@pytest.mark.asyncio
async def test_initialize_without_credentials(skill_no_creds):
    """Skill should fail to initialize without credentials."""
    result = await skill_no_creds.initialize()
    assert result is False
    assert skill_no_creds.initialized is False


def test_record_usage(skill):
    """Usage recording should update stats correctly."""
    skill.record_usage(cost=0.001, revenue=0)
    skill.record_usage(cost=0.002, revenue=0)
    stats = skill.stats
    assert stats["usage_count"] == 2
    assert abs(stats["total_cost"] - 0.003) < 1e-9


def test_estimate_cost(skill):
    """Cost estimation should return the action's estimated cost."""
    assert skill.estimate_cost("send_message", {}) == 0.001
    assert skill.estimate_cost("search_messages", {}) == 0.002
    assert skill.estimate_cost("upload_file", {}) == 0.002
    assert skill.estimate_cost("nonexistent", {}) == 0
