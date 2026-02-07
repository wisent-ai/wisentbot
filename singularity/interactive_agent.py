#!/usr/bin/env python3
"""
InteractiveAgent — Conversational AI agent with tool execution.

Unlike AutonomousAgent which runs in a continuous loop, InteractiveAgent
processes messages from users one at a time, executing tools as needed
and returning a natural language response.

This enables:
- Chat-based services (code review, data analysis, Q&A)
- Human-in-the-loop workflows
- API-driven interactions where clients send messages and get responses
- Revenue generation through conversational services
"""

import asyncio
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .cognition import CognitionEngine, AgentState, Decision, Action, TokenUsage, calculate_api_cost
from .skills.base import SkillRegistry


@dataclass
class ChatMessage:
    """A message in the conversation."""
    role: str  # "user", "assistant", "tool_result"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from the agent to the user."""
    message: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    thinking_steps: int = 0


class InteractiveAgent:
    """
    A conversational AI agent that processes user messages and responds.

    Usage:
        agent = InteractiveAgent(llm_provider="anthropic")
        response = await agent.chat("Review this Python code for bugs")
        print(response.message)

    The agent can:
    - Execute tools to gather information or take actions
    - Maintain conversation history across messages
    - Track costs per conversation
    - Limit tool calls per message to prevent runaway execution
    """

    def __init__(
        self,
        name: str = "Assistant",
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
        llm_base_url: str = "http://localhost:8000/v1",
        anthropic_api_key: str = "",
        openai_api_key: str = "",
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        max_tool_calls_per_message: int = 10,
        skills: Optional[List[str]] = None,
    ):
        """
        Initialize an interactive agent.

        Args:
            name: Agent name
            llm_provider: LLM backend (anthropic, openai, etc.)
            llm_model: Model identifier
            llm_base_url: Base URL for OpenAI-compatible APIs
            anthropic_api_key: Anthropic API key
            openai_api_key: OpenAI API key
            system_prompt: Custom system prompt (overrides default)
            system_prompt_file: Path to system prompt file
            max_tool_calls_per_message: Max tools to call before forcing a response
            skills: List of skill IDs to load (None = load all available)
        """
        self.name = name
        self.max_tool_calls = max_tool_calls_per_message

        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0
        self.message_count = 0

        # Conversation history
        self.history: List[ChatMessage] = []

        # Build interactive system prompt
        interactive_prompt = system_prompt or self._default_system_prompt()

        # Initialize cognition engine
        self.cognition = CognitionEngine(
            llm_provider=llm_provider,
            anthropic_api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=llm_base_url,
            llm_model=llm_model,
            agent_name=name,
            agent_ticker="CHAT",
            agent_type="interactive",
            agent_specialty="conversational assistant",
            system_prompt=interactive_prompt,
            system_prompt_file=system_prompt_file,
        )

        # Skills registry
        self.skills = SkillRegistry()
        self._init_skills(filter_skills=skills)

    def _default_system_prompt(self) -> str:
        """Default prompt for interactive mode."""
        return f"""You are {self.name}, an interactive AI assistant with access to tools.

When a user sends you a message, you can either:
1. Respond directly with a helpful answer
2. Use a tool to gather information or take an action, then respond

RESPONSE FORMAT:
- To use a tool: respond with JSON: {{"tool": "skill:action", "params": {{}}, "reasoning": "why"}}
- To respond to the user: respond with JSON: {{"tool": "respond", "params": {{"message": "your response to the user"}}, "reasoning": "why"}}

IMPORTANT RULES:
- When you have enough information to answer, use the "respond" tool
- Keep responses helpful, clear, and concise
- If a tool call fails, try a different approach or explain the issue to the user
- You can use multiple tools before responding, but always end with "respond"
"""

    def _init_skills(self, filter_skills: Optional[List[str]] = None):
        """Install available skills."""
        credentials = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            "RESEND_API_KEY": os.environ.get("RESEND_API_KEY", ""),
            "VERCEL_TOKEN": os.environ.get("VERCEL_TOKEN", ""),
        }
        self.skills.set_credentials(credentials)

        # Import skill classes - only import what we need
        from .skills.filesystem import FilesystemSkill
        from .skills.shell import ShellSkill
        from .skills.request import RequestSkill

        # Core skills that don't need special credentials
        core_skills = [
            FilesystemSkill,
            ShellSkill,
            RequestSkill,
        ]

        # Optional skills that need credentials
        optional_skills = []
        try:
            from .skills.github import GitHubSkill
            optional_skills.append(GitHubSkill)
        except ImportError:
            pass

        try:
            from .skills.content import ContentCreationSkill
            optional_skills.append(ContentCreationSkill)
        except ImportError:
            pass

        all_skills = core_skills + optional_skills

        for skill_class in all_skills:
            try:
                self.skills.install(skill_class)
                skill = self.skills.get(skill_class(credentials).manifest.skill_id)

                if skill and skill.check_credentials():
                    # Wire LLM into content skill if present
                    if hasattr(skill, 'set_llm'):
                        skill.set_llm(
                            self.cognition.llm,
                            self.cognition.llm_type,
                            self.cognition.llm_model
                        )

                    # If filtering, check if this skill is wanted
                    if filter_skills is not None:
                        if skill.manifest.skill_id not in filter_skills:
                            self.skills.uninstall(skill.manifest.skill_id)
                            continue
                else:
                    self.skills.uninstall(skill_class(credentials).manifest.skill_id)
            except Exception:
                pass

    def _get_tools(self) -> List[Dict]:
        """Get tools from installed skills."""
        tools = []
        for skill in self.skills.skills.values():
            for action in skill.manifest.actions:
                tool_name = f"{skill.manifest.skill_id}:{action.name}"
                tools.append({
                    "name": tool_name,
                    "description": action.description,
                    "parameters": action.parameters,
                })

        # Always add the "respond" meta-tool
        tools.append({
            "name": "respond",
            "description": "Send a response message back to the user. Use this when you have enough information to answer.",
            "parameters": {"message": "The response text to send to the user"},
        })

        return tools

    async def chat(self, message: str, context: str = "") -> ChatResponse:
        """
        Process a user message and return a response.

        The agent will think about the message, optionally execute tools,
        and return a natural language response.

        Args:
            message: The user's message
            context: Optional additional context to include

        Returns:
            ChatResponse with the agent's reply and metadata
        """
        self.message_count += 1
        self.history.append(ChatMessage(role="user", content=message))

        tools = self._get_tools()
        tool_calls = []
        total_cost = 0.0
        total_tokens = 0
        steps = 0

        # Conversation context from history
        history_text = self._format_history(limit=20)

        for _ in range(self.max_tool_calls):
            steps += 1

            # Build state for cognition
            state = AgentState(
                balance=999.0,  # Not relevant for interactive mode
                burn_rate=0.0,
                runway_hours=999.0,
                tools=tools,
                recent_actions=[],
                cycle=steps,
                project_context=self._build_context(message, history_text, tool_calls, context),
            )

            # Think
            decision = await self.cognition.think(state)
            total_cost += decision.api_cost_usd
            total_tokens += decision.token_usage.total_tokens()

            action = decision.action

            # Check if agent wants to respond to user
            if action.tool == "respond":
                response_text = action.params.get("message", action.reasoning or "I'm not sure how to help with that.")
                self.history.append(ChatMessage(
                    role="assistant",
                    content=response_text,
                    metadata={"tool_calls": len(tool_calls), "cost": total_cost},
                ))
                self.total_cost += total_cost
                self.total_tokens += total_tokens
                return ChatResponse(
                    message=response_text,
                    tool_calls=tool_calls,
                    total_cost_usd=total_cost,
                    total_tokens=total_tokens,
                    thinking_steps=steps,
                )

            # Execute tool
            result = await self._execute(action)
            tool_calls.append({
                "tool": action.tool,
                "params": action.params,
                "result": result,
                "reasoning": action.reasoning,
            })

        # If we exhausted tool calls, force a response based on what we have
        response_text = self._synthesize_response(message, tool_calls)
        self.history.append(ChatMessage(
            role="assistant",
            content=response_text,
            metadata={"tool_calls": len(tool_calls), "cost": total_cost, "forced": True},
        ))
        self.total_cost += total_cost
        self.total_tokens += total_tokens
        return ChatResponse(
            message=response_text,
            tool_calls=tool_calls,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            thinking_steps=steps,
        )

    def _build_context(
        self,
        current_message: str,
        history_text: str,
        tool_calls: List[Dict],
        extra_context: str = "",
    ) -> str:
        """Build context for the LLM prompt."""
        parts = []

        if history_text:
            parts.append(f"CONVERSATION HISTORY:\n{history_text}")

        parts.append(f"CURRENT USER MESSAGE:\n{current_message}")

        if tool_calls:
            tool_text = "\n".join([
                f"  - {tc['tool']}: {json.dumps(tc['result'])[:300]}"
                for tc in tool_calls
            ])
            parts.append(f"TOOL RESULTS SO FAR:\n{tool_text}")
            parts.append("Use the 'respond' tool to send your final answer to the user.")

        if extra_context:
            parts.append(f"ADDITIONAL CONTEXT:\n{extra_context}")

        return "\n\n".join(parts)

    def _format_history(self, limit: int = 20) -> str:
        """Format conversation history for context."""
        # Skip the most recent user message (it's in current_message)
        relevant = self.history[:-1] if self.history else []
        relevant = relevant[-limit:]

        if not relevant:
            return ""

        lines = []
        for msg in relevant:
            role = msg.role.upper()
            content = msg.content[:500]
            lines.append(f"[{role}]: {content}")

        return "\n".join(lines)

    def _synthesize_response(self, message: str, tool_calls: List[Dict]) -> str:
        """Synthesize a response when max tool calls are exhausted."""
        if not tool_calls:
            return "I wasn't able to process your request. Could you try rephrasing?"

        # Gather successful results
        results = []
        for tc in tool_calls:
            if tc["result"].get("status") == "success":
                results.append(f"- {tc['tool']}: {tc['result'].get('message', tc['result'].get('data', ''))}"[:200])

        if results:
            return f"Here's what I found:\n" + "\n".join(results)
        else:
            return "I tried several approaches but wasn't able to get a clear result. Could you provide more details?"

    async def _execute(self, action: Action) -> Dict:
        """Execute a skill action."""
        tool = action.tool
        params = action.params

        if tool == "wait" or tool == "respond":
            return {"status": "skipped", "message": "Meta action"}

        if ":" in tool:
            parts = tool.split(":", 1)
            skill_id = parts[0]
            action_name = parts[1] if len(parts) > 1 else ""

            skill = self.skills.get(skill_id)
            if skill:
                try:
                    result = await skill.execute(action_name, params)
                    return {
                        "status": "success" if result.success else "failed",
                        "data": result.data,
                        "message": result.message,
                    }
                except Exception as e:
                    return {"status": "error", "message": str(e)}

        return {"status": "error", "message": f"Unknown tool: {tool}"}

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "messages": self.message_count,
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "history_length": len(self.history),
            "skills_loaded": len(self.skills.skills),
        }

    def get_history(self) -> List[Dict]:
        """Get conversation history as serializable dicts."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata,
            }
            for msg in self.history
        ]
