#!/usr/bin/env python3
"""
Slack Skill

Enables agents to:
- Send messages to channels and DMs
- Read channel history
- Search messages
- React to messages
- Upload files
- Manage channels (create, invite, set topic)
- Get user info
"""

import httpx
from typing import Dict, List, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction


class SlackSkill(Skill):
    """
    Skill for Slack API interactions.

    Required credentials:
    - SLACK_BOT_TOKEN: Slack bot token (xoxb-...)
    """

    API_BASE = "https://slack.com/api"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="slack",
            name="Slack Integration",
            version="1.0.0",
            category="social",
            description="Send messages, read channels, search, react, upload files, and manage Slack",
            required_credentials=["SLACK_BOT_TOKEN"],
            install_cost=0,
            author="eve",
            actions=[
                SkillAction(
                    name="send_message",
                    description="Send a message to a channel or user",
                    parameters={
                        "channel": {
                            "type": "string",
                            "required": True,
                            "description": "Channel ID or user ID to send to",
                        },
                        "text": {
                            "type": "string",
                            "required": True,
                            "description": "Message text",
                        },
                        "thread_ts": {
                            "type": "string",
                            "required": False,
                            "description": "Thread timestamp to reply in a thread",
                        },
                        "blocks": {
                            "type": "array",
                            "required": False,
                            "description": "Block Kit blocks for rich formatting",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="get_messages",
                    description="Read channel message history",
                    parameters={
                        "channel": {
                            "type": "string",
                            "required": True,
                            "description": "Channel ID to read from",
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of messages to fetch (default: 10)",
                        },
                        "oldest": {
                            "type": "string",
                            "required": False,
                            "description": "Only messages after this timestamp",
                        },
                        "latest": {
                            "type": "string",
                            "required": False,
                            "description": "Only messages before this timestamp",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="search_messages",
                    description="Search for messages across channels",
                    parameters={
                        "query": {
                            "type": "string",
                            "required": True,
                            "description": "Search query",
                        },
                        "sort": {
                            "type": "string",
                            "required": False,
                            "description": "Sort order: 'score' or 'timestamp'",
                        },
                        "count": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of results to return (default: 20)",
                        },
                    },
                    estimated_cost=0.002,
                    estimated_duration_seconds=5,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="react",
                    description="Add a reaction emoji to a message",
                    parameters={
                        "channel": {
                            "type": "string",
                            "required": True,
                            "description": "Channel ID containing the message",
                        },
                        "timestamp": {
                            "type": "string",
                            "required": True,
                            "description": "Message timestamp to react to",
                        },
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Emoji name without colons (e.g. 'thumbsup')",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="upload_file",
                    description="Upload a file to one or more channels",
                    parameters={
                        "channels": {
                            "type": "string",
                            "required": True,
                            "description": "Comma-separated channel IDs",
                        },
                        "content": {
                            "type": "string",
                            "required": True,
                            "description": "File content as text",
                        },
                        "filename": {
                            "type": "string",
                            "required": True,
                            "description": "Filename for the upload",
                        },
                        "title": {
                            "type": "string",
                            "required": False,
                            "description": "Title of the file",
                        },
                        "filetype": {
                            "type": "string",
                            "required": False,
                            "description": "File type identifier (e.g. 'python', 'json')",
                        },
                    },
                    estimated_cost=0.002,
                    estimated_duration_seconds=5,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="create_channel",
                    description="Create a new Slack channel",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Channel name (lowercase, no spaces)",
                        },
                        "is_private": {
                            "type": "boolean",
                            "required": False,
                            "description": "Create as private channel (default: false)",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="get_user_info",
                    description="Get information about a Slack user",
                    parameters={
                        "user_id": {
                            "type": "string",
                            "required": True,
                            "description": "Slack user ID",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="set_channel_topic",
                    description="Set the topic of a channel",
                    parameters={
                        "channel": {
                            "type": "string",
                            "required": True,
                            "description": "Channel ID",
                        },
                        "topic": {
                            "type": "string",
                            "required": True,
                            "description": "New channel topic text",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.95,
                ),
            ],
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient(timeout=30)

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for Slack API requests."""
        token = self.credentials.get("SLACK_BOT_TOKEN", "")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def _check_slack_response(self, data: Dict) -> Optional[str]:
        """
        Check a Slack API response for errors.

        Slack API always returns HTTP 200 but uses 'ok' field for success.

        Returns:
            Error message string if the response indicates failure, None if ok.
        """
        if not data.get("ok", False):
            return data.get("error", "unknown_error")
        return None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a Slack action."""
        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(
                success=False,
                message=f"Missing credentials: {missing}",
            )

        try:
            if action == "send_message":
                return await self._send_message(
                    params.get("channel"),
                    params.get("text"),
                    params.get("thread_ts"),
                    params.get("blocks"),
                )
            elif action == "get_messages":
                return await self._get_messages(
                    params.get("channel"),
                    params.get("limit", 10),
                    params.get("oldest"),
                    params.get("latest"),
                )
            elif action == "search_messages":
                return await self._search_messages(
                    params.get("query"),
                    params.get("sort", "score"),
                    params.get("count", 20),
                )
            elif action == "react":
                return await self._react(
                    params.get("channel"),
                    params.get("timestamp"),
                    params.get("name"),
                )
            elif action == "upload_file":
                return await self._upload_file(
                    params.get("channels"),
                    params.get("content"),
                    params.get("filename"),
                    params.get("title"),
                    params.get("filetype"),
                )
            elif action == "create_channel":
                return await self._create_channel(
                    params.get("name"),
                    params.get("is_private", False),
                )
            elif action == "get_user_info":
                return await self._get_user_info(params.get("user_id"))
            elif action == "set_channel_topic":
                return await self._set_channel_topic(
                    params.get("channel"),
                    params.get("topic"),
                )
            else:
                return SkillResult(
                    success=False,
                    message=f"Unknown action: {action}",
                )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Slack error: {str(e)}",
            )

    async def _send_message(
        self,
        channel: str,
        text: str,
        thread_ts: str = None,
        blocks: List = None,
    ) -> SkillResult:
        """Send a message to a channel or user."""
        if not channel or not text:
            return SkillResult(success=False, message="Channel and text are required")

        payload: Dict = {"channel": channel, "text": text}
        if thread_ts:
            payload["thread_ts"] = thread_ts
        if blocks:
            payload["blocks"] = blocks

        response = await self.http.post(
            f"{self.API_BASE}/chat.postMessage",
            headers=self._get_headers(),
            json=payload,
        )
        data = response.json()

        error = self._check_slack_response(data)
        if error:
            return SkillResult(
                success=False,
                message=f"Failed to send message: {error}",
            )

        return SkillResult(
            success=True,
            message=f"Message sent to {channel}",
            data={
                "channel": data.get("channel"),
                "ts": data.get("ts"),
                "message": data.get("message", {}),
            },
            cost=0.001,
        )

    async def _get_messages(
        self,
        channel: str,
        limit: int = 10,
        oldest: str = None,
        latest: str = None,
    ) -> SkillResult:
        """Read channel message history with pagination support."""
        if not channel:
            return SkillResult(success=False, message="Channel is required")

        limit = min(max(limit, 1), 1000)

        payload: Dict = {"channel": channel, "limit": limit}
        if oldest:
            payload["oldest"] = oldest
        if latest:
            payload["latest"] = latest

        response = await self.http.post(
            f"{self.API_BASE}/conversations.history",
            headers=self._get_headers(),
            json=payload,
        )
        data = response.json()

        error = self._check_slack_response(data)
        if error:
            return SkillResult(
                success=False,
                message=f"Failed to get messages: {error}",
            )

        messages = data.get("messages", [])
        return SkillResult(
            success=True,
            message=f"Retrieved {len(messages)} messages from {channel}",
            data={
                "messages": messages,
                "count": len(messages),
                "has_more": data.get("has_more", False),
                "response_metadata": data.get("response_metadata", {}),
            },
            cost=0.001,
        )

    async def _search_messages(
        self,
        query: str,
        sort: str = "score",
        count: int = 20,
    ) -> SkillResult:
        """Search for messages across channels."""
        if not query:
            return SkillResult(success=False, message="Search query is required")

        count = min(max(count, 1), 100)

        payload: Dict = {"query": query, "sort": sort, "count": count}

        response = await self.http.post(
            f"{self.API_BASE}/search.messages",
            headers=self._get_headers(),
            json=payload,
        )
        data = response.json()

        error = self._check_slack_response(data)
        if error:
            return SkillResult(
                success=False,
                message=f"Search failed: {error}",
            )

        matches = data.get("messages", {}).get("matches", [])
        total = data.get("messages", {}).get("total", 0)
        return SkillResult(
            success=True,
            message=f"Found {total} messages matching '{query}'",
            data={
                "matches": matches,
                "total": total,
                "query": query,
            },
            cost=0.002,
        )

    async def _react(
        self,
        channel: str,
        timestamp: str,
        name: str,
    ) -> SkillResult:
        """Add a reaction to a message."""
        if not channel or not timestamp or not name:
            return SkillResult(
                success=False,
                message="Channel, timestamp, and emoji name are required",
            )

        # Strip colons if user accidentally includes them
        name = name.strip(":")

        payload = {"channel": channel, "timestamp": timestamp, "name": name}

        response = await self.http.post(
            f"{self.API_BASE}/reactions.add",
            headers=self._get_headers(),
            json=payload,
        )
        data = response.json()

        error = self._check_slack_response(data)
        if error:
            return SkillResult(
                success=False,
                message=f"Failed to add reaction: {error}",
            )

        return SkillResult(
            success=True,
            message=f"Added :{name}: reaction",
            data={"channel": channel, "timestamp": timestamp, "emoji": name},
            cost=0.001,
        )

    async def _upload_file(
        self,
        channels: str,
        content: str,
        filename: str,
        title: str = None,
        filetype: str = None,
    ) -> SkillResult:
        """Upload a file to one or more channels."""
        if not channels or not content or not filename:
            return SkillResult(
                success=False,
                message="Channels, content, and filename are required",
            )

        payload: Dict = {
            "channels": channels,
            "content": content,
            "filename": filename,
        }
        if title:
            payload["title"] = title
        if filetype:
            payload["filetype"] = filetype

        response = await self.http.post(
            f"{self.API_BASE}/files.upload",
            headers=self._get_headers(),
            json=payload,
        )
        data = response.json()

        error = self._check_slack_response(data)
        if error:
            return SkillResult(
                success=False,
                message=f"Failed to upload file: {error}",
            )

        file_data = data.get("file", {})
        return SkillResult(
            success=True,
            message=f"File '{filename}' uploaded to {channels}",
            data={
                "file_id": file_data.get("id"),
                "filename": file_data.get("name"),
                "url": file_data.get("url_private"),
                "size": file_data.get("size"),
            },
            cost=0.002,
        )

    async def _create_channel(
        self,
        name: str,
        is_private: bool = False,
    ) -> SkillResult:
        """Create a new Slack channel."""
        if not name:
            return SkillResult(success=False, message="Channel name is required")

        payload: Dict = {"name": name, "is_private": is_private}

        response = await self.http.post(
            f"{self.API_BASE}/conversations.create",
            headers=self._get_headers(),
            json=payload,
        )
        data = response.json()

        error = self._check_slack_response(data)
        if error:
            return SkillResult(
                success=False,
                message=f"Failed to create channel: {error}",
            )

        channel_data = data.get("channel", {})
        return SkillResult(
            success=True,
            message=f"Channel #{name} created",
            data={
                "channel_id": channel_data.get("id"),
                "name": channel_data.get("name"),
                "is_private": channel_data.get("is_private", False),
            },
            cost=0.001,
        )

    async def _get_user_info(self, user_id: str) -> SkillResult:
        """Get information about a Slack user."""
        if not user_id:
            return SkillResult(success=False, message="User ID is required")

        payload = {"user": user_id}

        response = await self.http.post(
            f"{self.API_BASE}/users.info",
            headers=self._get_headers(),
            json=payload,
        )
        data = response.json()

        error = self._check_slack_response(data)
        if error:
            return SkillResult(
                success=False,
                message=f"Failed to get user info: {error}",
            )

        user_data = data.get("user", {})
        profile = user_data.get("profile", {})
        return SkillResult(
            success=True,
            message=f"Got info for user {user_id}",
            data={
                "id": user_data.get("id"),
                "name": user_data.get("name"),
                "real_name": user_data.get("real_name"),
                "display_name": profile.get("display_name"),
                "email": profile.get("email"),
                "title": profile.get("title"),
                "is_admin": user_data.get("is_admin", False),
                "is_bot": user_data.get("is_bot", False),
                "tz": user_data.get("tz"),
            },
            cost=0.001,
        )

    async def _set_channel_topic(self, channel: str, topic: str) -> SkillResult:
        """Set the topic of a channel."""
        if not channel or not topic:
            return SkillResult(
                success=False,
                message="Channel and topic are required",
            )

        payload = {"channel": channel, "topic": topic}

        response = await self.http.post(
            f"{self.API_BASE}/conversations.setTopic",
            headers=self._get_headers(),
            json=payload,
        )
        data = response.json()

        error = self._check_slack_response(data)
        if error:
            return SkillResult(
                success=False,
                message=f"Failed to set topic: {error}",
            )

        return SkillResult(
            success=True,
            message=f"Topic set for {channel}",
            data={
                "channel": channel,
                "topic": data.get("topic", topic),
            },
            cost=0.001,
        )

    async def close(self):
        """Clean up HTTP client."""
        await self.http.aclose()
