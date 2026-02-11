"""Tests for singularity.cognition.engine — CognitionEngine."""

import pytest
from unittest.mock import MagicMock
from singularity.cognition.types import TokenUsage, AgentState
from singularity.cognition.engine import CognitionEngine


def _make_engine(**kwargs):
    """Create a CognitionEngine with mocked providers."""
    defaults = dict(
        llm_provider="none",
        anthropic_api_key="",
        openai_api_key="",
        llm_model="test-model",
        agent_name="TestAgent",
        agent_ticker="TEST",
        agent_type="test",
        agent_specialty="testing",
    )
    defaults.update(kwargs)
    return CognitionEngine(**defaults)


def _make_state(**overrides):
    defaults = dict(
        balance=10.0, burn_rate=0.02, runway_hours=500,
        tools=[], recent_actions=[], cycle=1,
        chat_messages=[], project_context="",
        goals_progress={}, pending_tasks=[],
        created_resources={},
    )
    defaults.update(overrides)
    return AgentState(**defaults)


# ─── Initialization ──────────────────────────────────────────────────────


class TestCognitionEngineInit:
    def test_basic_init(self):
        e = _make_engine()
        assert e.agent_name == "TestAgent"
        assert e.agent_ticker == "TEST"
        assert e.agent_type == "test"
        assert e.agent_specialty == "testing"

    def test_default_specialty_fallback(self):
        e = _make_engine(agent_specialty="", agent_type="trading")
        assert e.agent_specialty == "trading"

    def test_llm_model_stored(self):
        e = _make_engine(llm_model="gpt-4o")
        assert e.llm_model == "gpt-4o"

    def test_no_provider_means_none(self):
        e = _make_engine(llm_provider="none")
        assert e.llm_type == "none"
        assert e.llm is None

    def test_training_examples_initially_empty(self):
        e = _make_engine()
        assert e._training_examples == []
        assert e._finetuned_model_id is None

    def test_prompt_additions_initially_empty(self):
        e = _make_engine()
        assert e._prompt_additions == []


# ─── System Prompt Management ────────────────────────────────────────────


class TestSystemPrompt:
    def test_get_system_prompt_basic(self):
        e = _make_engine()
        e.system_prompt = "Be helpful"
        assert e.get_system_prompt() == "Be helpful"

    def test_get_system_prompt_with_additions(self):
        e = _make_engine()
        e.system_prompt = "Base"
        e._prompt_additions = ["Rule 1", "Rule 2"]
        result = e.get_system_prompt()
        assert "Base" in result
        assert "Rule 1" in result
        assert "Rule 2" in result

    def test_set_system_prompt_resets_additions(self):
        e = _make_engine()
        e._prompt_additions = ["old rule"]
        e.set_system_prompt("New prompt")
        assert e.system_prompt == "New prompt"
        assert e._prompt_additions == []

    def test_append_to_prompt(self):
        e = _make_engine()
        e.append_to_prompt("New rule")
        assert "New rule" in e._prompt_additions


# ─── Model Access ────────────────────────────────────────────────────────


class TestModelAccess:
    def test_get_model(self):
        e = _make_engine()
        assert e.get_model() is None  # No provider initialized

    def test_get_tokenizer(self):
        e = _make_engine()
        assert e.get_tokenizer() is None

    def test_is_local_model(self):
        e = _make_engine()
        e.llm_type = "vllm"
        assert e.is_local_model() is True

    def test_is_not_local_model(self):
        e = _make_engine()
        e.llm_type = "anthropic"
        assert e.is_local_model() is False

    def test_get_current_model(self):
        e = _make_engine(llm_model="claude-sonnet-4-20250514")
        info = e.get_current_model()
        assert info["model"] == "claude-sonnet-4-20250514"
        assert info["finetuned"] is False
        assert info["finetuned_model_id"] is None


# ─── Training Data ──────────────────────────────────────────────────────


class TestTrainingData:
    def test_record_training_example(self):
        e = _make_engine()
        e.record_training_example("prompt", "response", "success")
        assert len(e._training_examples) == 1
        example = e._training_examples[0]
        assert example["outcome"] == "success"
        assert len(example["messages"]) == 3

    def test_get_training_examples_all(self):
        e = _make_engine()
        e.record_training_example("p1", "r1", "success")
        e.record_training_example("p2", "r2", "failure")
        assert len(e.get_training_examples()) == 2

    def test_get_training_examples_filtered(self):
        e = _make_engine()
        e.record_training_example("p1", "r1", "success")
        e.record_training_example("p2", "r2", "failure")
        e.record_training_example("p3", "r3", "success")
        success = e.get_training_examples("success")
        assert len(success) == 2
        assert all(ex["outcome"] == "success" for ex in success)

    def test_clear_training_examples(self):
        e = _make_engine()
        e.record_training_example("p", "r")
        e.record_training_example("p2", "r2")
        count = e.clear_training_examples()
        assert count == 2
        assert e._training_examples == []

    def test_export_training_data_as_string(self):
        e = _make_engine()
        e.record_training_example("prompt", "response", "success")
        result = e.export_training_data()
        assert isinstance(result, str)
        assert "prompt" in result

    def test_export_training_data_only_success(self):
        e = _make_engine()
        e.record_training_example("p1", "r1", "success")
        e.record_training_example("p2", "r2", "failure")
        result = e.export_training_data()
        # Should only have 1 line (1 success)
        lines = [ln for ln in result.strip().split("\n") if ln]
        assert len(lines) == 1

    def test_get_training_examples_returns_copy(self):
        e = _make_engine()
        e.record_training_example("p", "r")
        examples = e.get_training_examples()
        examples.clear()  # Modifying copy shouldn't affect original
        assert len(e._training_examples) == 1


# ─── think (no LLM) ─────────────────────────────────────────────────────


class TestThinkNoLLM:
    @pytest.mark.asyncio
    async def test_think_without_llm_returns_wait(self):
        e = _make_engine()
        assert e.llm is None
        state = _make_state()
        decision = await e.think(state)
        assert decision.action.tool == "wait"
        assert "No LLM" in decision.reasoning

    @pytest.mark.asyncio
    async def test_think_with_context_without_llm(self):
        e = _make_engine()
        state = _make_state()
        decision, conversation = await e.think_with_context(state)
        assert decision.action.tool == "wait"
        assert isinstance(conversation, list)


# ─── _finalize_decision ─────────────────────────────────────────────────


class TestFinalizeDecision:
    def test_finalize_parses_and_calculates_cost(self):
        e = _make_engine()
        e.llm_type = "anthropic"
        e.llm_model = "claude-sonnet-4-20250514"
        text = "REASON: testing\nTOOL: github:search_repos\nPARAM_query: singularity"
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        decision = e._finalize_decision(text, usage)
        assert decision.action.tool == "github:search_repos"
        assert decision.token_usage.input_tokens == 1000
        assert decision.api_cost_usd > 0

    def test_finalize_calls_cost_callback(self):
        callback = MagicMock()
        e = _make_engine()
        e._cost_callback = callback
        e.llm_type = "anthropic"
        e.llm_model = "claude-sonnet-4-20250514"
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        e._finalize_decision("REASON: x\nTOOL: wait", usage)
        callback.assert_called_once()

    def test_finalize_no_callback_when_zero_tokens(self):
        callback = MagicMock()
        e = _make_engine()
        e._cost_callback = callback
        e._finalize_decision("REASON: x\nTOOL: wait", TokenUsage())
        callback.assert_not_called()


# ─── use_finetuned_model ─────────────────────────────────────────────────


class TestUseFinetuned:
    def test_no_finetuned_model(self):
        e = _make_engine()
        assert e.use_finetuned_model() is False

    def test_with_finetuned_model_id(self):
        e = _make_engine()
        e._finetuned_model_id = "ft:gpt-4o-mini:test"
        # switch_model will fail because no HAS_OPENAI in test env
        # but the method should still try
        result = e.use_finetuned_model()
        # Result depends on provider availability - just test it doesn't crash
        assert isinstance(result, bool)
