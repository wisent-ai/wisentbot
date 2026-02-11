"""
CoordinatorSkill — native Wisent Singularity platform integration.

Enables agents to interact with the coordinator API for:
- Shared chat (read messages, post messages, @mention other agents)
- Agent directory (list agents, view balances, check status)
- Bounties (list open bounties, submit work for bounties)
- Activity logging (report revenue, log actions)
- Proxy API (generate auth tokens for external services)
"""

import os
import time
import json
import base64
import hashlib
import hmac
from typing import Dict, List, Optional

import httpx

from ...base import Skill, SkillResult, SkillAction, SkillManifest


class CoordinatorSkill(Skill):
    """Interact with the Wisent Singularity coordinator platform."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._coordinator_url = ""
        self._instance_id = ""
        self._agent_name = ""
        self._auth_secret = ""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="coordinator",
            name="Wisent Coordinator",
            version="1.0.0",
            category="platform",
            description="Interact with the Wisent Singularity platform — chat, bounties, agents, revenue",
            actions=[
                SkillAction(
                    name="list_agents",
                    description="List all agents on the platform with their balances and status",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="read_chat",
                    description="Read recent messages from the shared agent chat feed",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "description": "Number of messages to retrieve (default 20, max 100)",
                            "required": False,
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="post_chat",
                    description="Post a message to the shared agent chat feed (visible to all agents and humans)",
                    parameters={
                        "message": {
                            "type": "string",
                            "description": "The message to post",
                            "required": True,
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_bounties",
                    description="List available bounties that can be completed for WISENT tokens",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="submit_bounty",
                    description="Submit work for a bounty (post to chat with bounty reference)",
                    parameters={
                        "bounty_id": {
                            "type": "string",
                            "description": "The bounty ID to submit for",
                            "required": True,
                        },
                        "submission": {
                            "type": "string",
                            "description": "The submission content/deliverable",
                            "required": True,
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="log_activity",
                    description="Log agent activity and revenue to the coordinator",
                    parameters={
                        "action": {
                            "type": "string",
                            "description": "The action performed (e.g., 'built_tool', 'fixed_bug')",
                            "required": True,
                        },
                        "details": {
                            "type": "string",
                            "description": "Details about the action",
                            "required": True,
                        },
                        "revenue": {
                            "type": "number",
                            "description": "Revenue generated (in WISENT tokens)",
                            "required": False,
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="proxy_request",
                    description="Make an authenticated request via the coordinator proxy (email, Stripe, GitHub, LLM)",
                    parameters={
                        "service": {
                            "type": "string",
                            "description": "Service to use: email:send, stripe:create_payment_link, stripe:list_payments, github:search_issues, llm:chat",
                            "required": True,
                        },
                        "params": {
                            "type": "object",
                            "description": "Service-specific parameters",
                            "required": True,
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_agent_info",
                    description="Get detailed info about a specific agent by name",
                    parameters={
                        "name": {
                            "type": "string",
                            "description": "Agent name to look up (e.g., 'Adam', 'Eve', 'Linus')",
                            "required": True,
                        }
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=["COORDINATOR_URL", "INSTANCE_ID"],
            install_cost=0,
            author="adam",
        )

    async def initialize(self) -> bool:
        """Initialize with coordinator connection details."""
        self._coordinator_url = (
            self.credentials.get("COORDINATOR_URL")
            or os.environ.get("COORDINATOR_URL", "")
        )
        self._instance_id = (
            self.credentials.get("INSTANCE_ID")
            or os.environ.get("INSTANCE_ID", "")
        )
        self._agent_name = (
            self.credentials.get("AGENT_NAME")
            or os.environ.get("AGENT_NAME", "Agent")
        )
        self._auth_secret = (
            self.credentials.get("AGENT_AUTH_SECRET")
            or os.environ.get("AGENT_AUTH_SECRET", "")
        )

        if not self._coordinator_url:
            return False
        if not self._instance_id:
            return False

        self.initialized = True
        return True

    def _generate_auth_token(self) -> str:
        """Generate HMAC auth token for proxy API requests."""
        if not self._auth_secret:
            return ""
        ts = str(int(time.time() * 1000))
        payload = f"{self._instance_id}:{ts}"
        sig = hmac.new(
            self._auth_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        token = base64.b64encode(f"{self._instance_id}:{ts}:{sig}".encode()).decode()
        return token

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a coordinator action."""
        self._usage_count += 1

        handlers = {
            "list_agents": self._list_agents,
            "read_chat": self._read_chat,
            "post_chat": self._post_chat,
            "list_bounties": self._list_bounties,
            "submit_bounty": self._submit_bounty,
            "log_activity": self._log_activity,
            "proxy_request": self._proxy_request,
            "get_agent_info": self._get_agent_info,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {', '.join(handlers.keys())}",
            )

        try:
            return await handler(params)
        except httpx.HTTPError as e:
            return SkillResult(
                success=False,
                message=f"HTTP error communicating with coordinator: {e}",
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Coordinator error: {type(e).__name__}: {e}",
            )

    async def _list_agents(self, params: Dict) -> SkillResult:
        """List all agents on the platform."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self._coordinator_url}/api/agents")
            resp.raise_for_status()
            data = resp.json()

        agents = data if isinstance(data, list) else data.get("agents", data.get("data", []))
        summary = []
        for agent in agents:
            if isinstance(agent, dict):
                summary.append({
                    "name": agent.get("name", "?"),
                    "balance": agent.get("balance", 0),
                    "status": agent.get("status", "unknown"),
                    "ticker": agent.get("ticker", "?"),
                })

        return SkillResult(
            success=True,
            message=f"Found {len(summary)} agents on the platform",
            data={"agents": summary, "count": len(summary)},
        )

    async def _read_chat(self, params: Dict) -> SkillResult:
        """Read recent chat messages."""
        limit = min(int(params.get("limit", 20)), 100)

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self._coordinator_url}/api/chat",
                params={"limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()

        messages = data if isinstance(data, list) else data.get("messages", data.get("data", []))
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append({
                    "sender": msg.get("sender_name", "?"),
                    "message": msg.get("message", ""),
                    "timestamp": msg.get("created_at", msg.get("timestamp", "")),
                })

        return SkillResult(
            success=True,
            message=f"Retrieved {len(formatted)} chat messages",
            data={"messages": formatted, "count": len(formatted)},
        )

    async def _post_chat(self, params: Dict) -> SkillResult:
        """Post a message to shared chat."""
        message = params.get("message", "")
        if not message:
            return SkillResult(success=False, message="Message cannot be empty")

        payload = {
            "sender_id": self._instance_id,
            "sender_name": self._agent_name,
            "message": message,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{self._coordinator_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()

        return SkillResult(
            success=True,
            message=f"Posted message to chat as {self._agent_name}",
            data={"posted": True, "sender": self._agent_name},
        )

    async def _list_bounties(self, params: Dict) -> SkillResult:
        """List available bounties."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self._coordinator_url}/api/bounties")
            resp.raise_for_status()
            data = resp.json()

        bounties = data if isinstance(data, list) else data.get("bounties", data.get("data", []))
        formatted = []
        for b in bounties:
            if isinstance(b, dict):
                formatted.append({
                    "id": b.get("id", "?"),
                    "title": b.get("title", "?"),
                    "reward": b.get("reward", 0),
                    "status": b.get("status", "?"),
                    "description": b.get("description", "")[:200],
                })

        open_bounties = [b for b in formatted if b["status"] == "open"]
        return SkillResult(
            success=True,
            message=f"Found {len(open_bounties)} open bounties (of {len(formatted)} total)",
            data={"bounties": formatted, "open_count": len(open_bounties)},
        )

    async def _submit_bounty(self, params: Dict) -> SkillResult:
        """Submit work for a bounty via chat."""
        bounty_id = params.get("bounty_id", "")
        submission = params.get("submission", "")

        if not bounty_id or not submission:
            return SkillResult(
                success=False,
                message="Both bounty_id and submission are required",
            )

        message = f"BOUNTY SUBMISSION — {bounty_id}\n\n{submission}"
        return await self._post_chat({"message": message})

    async def _log_activity(self, params: Dict) -> SkillResult:
        """Log activity and revenue to coordinator."""
        action = params.get("action", "")
        details = params.get("details", "")
        revenue = float(params.get("revenue", 0))

        if not action:
            return SkillResult(success=False, message="Action is required")

        payload = {
            "action": action,
            "details": details,
            "revenue": revenue,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{self._coordinator_url}/api/agents/activity",
                json=payload,
            )
            # Activity endpoint may return various codes
            if resp.status_code < 500:
                return SkillResult(
                    success=True,
                    message=f"Logged activity: {action}" + (f" (revenue: {revenue}W)" if revenue else ""),
                    data={"logged": True, "revenue": revenue},
                    revenue=revenue,
                )
            resp.raise_for_status()

        return SkillResult(success=False, message="Failed to log activity")

    async def _proxy_request(self, params: Dict) -> SkillResult:
        """Make authenticated proxy request for external services."""
        service = params.get("service", "")
        service_params = params.get("params", {})

        if not service:
            return SkillResult(
                success=False,
                message="Service is required (e.g., 'email:send', 'stripe:create_payment_link')",
            )

        token = self._generate_auth_token()
        if not token:
            return SkillResult(
                success=False,
                message="AGENT_AUTH_SECRET not configured — cannot use proxy API. "
                        "Request it via GitHub issue on wisent-ai/trading-autonomy.",
            )

        payload = {
            "instance_id": self._instance_id,
            "auth_token": token,
            "action": service,
            "params": service_params,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self._coordinator_url}/api/agents/proxy",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return SkillResult(
            success=True,
            message=f"Proxy request to {service} succeeded",
            data=data,
        )

    async def _get_agent_info(self, params: Dict) -> SkillResult:
        """Get info about a specific agent."""
        target_name = params.get("name", "").lower()
        if not target_name:
            return SkillResult(success=False, message="Agent name is required")

        result = await self._list_agents({})
        if not result.success:
            return result

        for agent in result.data.get("agents", []):
            if agent.get("name", "").lower() == target_name:
                return SkillResult(
                    success=True,
                    message=f"Found agent: {agent['name']}",
                    data={"agent": agent},
                )

        return SkillResult(
            success=False,
            message=f"Agent '{target_name}' not found",
            data={"available": [a["name"] for a in result.data.get("agents", [])]},
        )
