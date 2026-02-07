"""Tests for InteractiveAgent."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.interactive_agent import InteractiveAgent, ChatMessage, ChatResponse


class TestChatMessage:
    def test_creation(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp  # auto-populated
        assert msg.metadata == {}

    def test_with_metadata(self):
        msg = ChatMessage(role="assistant", content="hi", metadata={"cost": 0.01})
        assert msg.metadata["cost"] == 0.01


class TestChatResponse:
    def test_creation(self):
        resp = ChatResponse(message="Hello!")
        assert resp.message == "Hello!"
        assert resp.tool_calls == []
        assert resp.total_cost_usd == 0.0
        assert resp.total_tokens == 0
        assert resp.thinking_steps == 0

    def test_with_tool_calls(self):
        resp = ChatResponse(
            message="Done",
            tool_calls=[{"tool": "fs:read", "result": {"status": "success"}}],
            total_cost_usd=0.003,
            total_tokens=500,
            thinking_steps=2,
        )
        assert len(resp.tool_calls) == 1
        assert resp.thinking_steps == 2


class TestInteractiveAgent:
    def test_init_without_llm(self):
        """Test agent initializes with 'none' provider."""
        agent = InteractiveAgent(name="TestBot", llm_provider="none")
        assert agent.name == "TestBot"
        assert agent.message_count == 0
        assert agent.total_cost == 0.0
        assert agent.history == []

    def test_get_tools_has_respond(self):
        agent = InteractiveAgent(llm_provider="none")
        tools = agent._get_tools()
        tool_names = [t["name"] for t in tools]
        assert "respond" in tool_names

    def test_clear_history(self):
        agent = InteractiveAgent(llm_provider="none")
        agent.history.append(ChatMessage(role="user", content="hi"))
        agent.clear_history()
        assert len(agent.history) == 0

    def test_get_stats(self):
        agent = InteractiveAgent(llm_provider="none")
        stats = agent.get_stats()
        assert stats["messages"] == 0
        assert stats["total_cost_usd"] == 0.0
        assert "skills_loaded" in stats

    def test_get_history_empty(self):
        agent = InteractiveAgent(llm_provider="none")
        assert agent.get_history() == []

    def test_format_history(self):
        agent = InteractiveAgent(llm_provider="none")
        agent.history.append(ChatMessage(role="user", content="hello"))
        agent.history.append(ChatMessage(role="assistant", content="hi there"))
        agent.history.append(ChatMessage(role="user", content="new msg"))
        result = agent._format_history()
        assert "[USER]: hello" in result
        assert "[ASSISTANT]: hi there" in result

    def test_synthesize_no_calls(self):
        agent = InteractiveAgent(llm_provider="none")
        result = agent._synthesize_response("test", [])
        assert "rephrasing" in result.lower() or "process" in result.lower()

    def test_synthesize_with_results(self):
        agent = InteractiveAgent(llm_provider="none")
        tool_calls = [
            {"tool": "fs:read", "result": {"status": "success", "message": "file content here"}}
        ]
        result = agent._synthesize_response("read a file", tool_calls)
        assert "found" in result.lower()

    def test_default_system_prompt(self):
        agent = InteractiveAgent(name="TestBot", llm_provider="none")
        prompt = agent._default_system_prompt()
        assert "TestBot" in prompt
        assert "respond" in prompt

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        agent = InteractiveAgent(llm_provider="none")
        result = await agent._execute(
            MagicMock(tool="nonexistent:action", params={})
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_execute_wait(self):
        agent = InteractiveAgent(llm_provider="none")
        result = await agent._execute(MagicMock(tool="wait", params={}))
        assert result["status"] == "skipped"
