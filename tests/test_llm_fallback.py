"""Tests for LLM provider fallback chain in CognitionEngine."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.cognition import CognitionEngine, TokenUsage, AgentState


class TestFallbackChainBuilding:
    """Test _build_fallback_chain constructs correct chains."""

    def test_no_fallback_when_disabled(self):
        engine = CognitionEngine(llm_provider="none", enable_fallback=False)
        assert engine._fallback_chain == []

    def test_fallback_chain_excludes_primary(self):
        engine = CognitionEngine(llm_provider="none", enable_fallback=True)
        # Primary is 'none', so all available providers should be in chain
        # Without API keys, chain should be empty
        assert all(f["provider"] != engine._primary_provider for f in engine._fallback_chain)

    @patch("singularity.cognition.HAS_ANTHROPIC", True)
    @patch("singularity.cognition.HAS_OPENAI", True)
    def test_anthropic_primary_has_openai_fallback(self):
        engine = CognitionEngine(
            llm_provider="none",
            anthropic_api_key="test-key",
            openai_api_key="test-key",
            enable_fallback=True,
        )
        providers = [f["provider"] for f in engine._fallback_chain]
        assert "anthropic" in providers
        assert "openai" in providers

    def test_fallback_stats_initialized(self):
        engine = CognitionEngine(llm_provider="none")
        stats = engine.get_fallback_stats()
        assert stats["primary_failures"] == 0
        assert stats["fallback_successes"] == 0
        assert stats["total_fallbacks"] == 0


class TestCallProvider:
    """Test _call_provider with mocked clients."""

    @pytest.mark.asyncio
    async def test_call_anthropic_provider(self):
        engine = CognitionEngine(llm_provider="none")
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"tool": "wait", "params": {}}')]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        text, usage = await engine._call_provider(
            "anthropic", "claude-sonnet-4-20250514", mock_client, "sys", "user"
        )
        assert "wait" in text
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_call_openai_provider(self):
        engine = CognitionEngine(llm_provider="none")
        mock_client = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.content = '{"tool": "wait", "params": {}}'
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 8
        mock_client.chat.completions.create.return_value = mock_response

        text, usage = await engine._call_provider(
            "openai", "gpt-4o-mini", mock_client, "sys", "user"
        )
        assert "wait" in text
        assert usage.input_tokens == 20

    @pytest.mark.asyncio
    async def test_call_unsupported_provider_raises(self):
        engine = CognitionEngine(llm_provider="none")
        with pytest.raises(ValueError, match="Unsupported"):
            await engine._call_provider("unknown", "model", None, "sys", "user")


class TestCallWithFallback:
    """Test _call_with_fallback tries providers in order."""

    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(self):
        engine = CognitionEngine(llm_provider="none")
        engine.llm = MagicMock()
        engine.llm_type = "anthropic"
        engine.llm_model = "claude-sonnet-4-20250514"

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text='{"tool":"wait"}')]
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5
        engine.llm.messages = AsyncMock()
        engine.llm.messages.create = AsyncMock(return_value=mock_resp)

        text, usage, provider, model = await engine._call_with_fallback("sys", "user")
        assert text == '{"tool":"wait"}'
        assert provider == "anthropic"
        assert engine._fallback_stats["primary_failures"] == 0

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_succeeds(self):
        engine = CognitionEngine(llm_provider="none", enable_fallback=True)
        engine.llm = MagicMock()
        engine.llm_type = "anthropic"
        engine.llm_model = "claude-sonnet-4-20250514"
        engine.llm.messages = AsyncMock()
        engine.llm.messages.create = AsyncMock(side_effect=Exception("Rate limited"))

        # Set up a fallback
        engine._fallback_chain = [{"provider": "openai", "model": "gpt-4o-mini"}]

        mock_client = AsyncMock()
        mock_msg = MagicMock(content='{"tool":"shell:bash"}')
        mock_choice = MagicMock(message=mock_msg)
        mock_resp = MagicMock(choices=[mock_choice])
        mock_resp.usage.prompt_tokens = 15
        mock_resp.usage.completion_tokens = 7
        mock_client.chat.completions.create.return_value = mock_resp
        engine._fallback_clients["openai"] = mock_client

        text, usage, provider, model = await engine._call_with_fallback("sys", "user")
        assert provider == "openai"
        assert model == "gpt-4o-mini"
        assert engine._fallback_stats["primary_failures"] == 1
        assert engine._fallback_stats["fallback_successes"] == 1

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self):
        engine = CognitionEngine(llm_provider="none", enable_fallback=True)
        engine.llm = MagicMock()
        engine.llm_type = "anthropic"
        engine.llm_model = "test"
        engine.llm.messages = AsyncMock()
        engine.llm.messages.create = AsyncMock(side_effect=Exception("Down"))
        engine._fallback_chain = [{"provider": "openai", "model": "gpt-4o-mini"}]
        engine._fallback_clients["openai"] = None  # Will fail

        with pytest.raises(Exception, match="Down"):
            await engine._call_with_fallback("sys", "user")

    @pytest.mark.asyncio
    async def test_no_provider_returns_empty(self):
        engine = CognitionEngine(llm_provider="none")
        engine.llm = None
        engine.llm_type = "none"
        text, usage, provider, model = await engine._call_with_fallback("sys", "user")
        assert text == ""
        assert provider == "none"
