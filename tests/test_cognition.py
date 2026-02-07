"""Tests for the cognition module - data classes, cost calculation, and action parsing."""

import json

import pytest

from singularity.cognition import (
    Action,
    AgentState,
    CognitionEngine,
    Decision,
    TokenUsage,
    calculate_api_cost,
)

# ── TokenUsage ──────────────────────────────────────────────────────────


class TestTokenUsage:
    def test_defaults(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens() == 150

    def test_total_tokens_zero(self):
        usage = TokenUsage()
        assert usage.total_tokens() == 0

    def test_large_values(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        assert usage.total_tokens() == 1_500_000


# ── Action ──────────────────────────────────────────────────────────────


class TestAction:
    def test_defaults(self):
        action = Action(tool="wait")
        assert action.tool == "wait"
        assert action.params == {}
        assert action.reasoning == ""

    def test_with_params(self):
        action = Action(
            tool="shell:bash",
            params={"command": "echo hello"},
            reasoning="Testing command execution",
        )
        assert action.tool == "shell:bash"
        assert action.params == {"command": "echo hello"}
        assert action.reasoning == "Testing command execution"

    def test_params_default_factory(self):
        """Ensure each Action gets its own dict instance for params."""
        a1 = Action(tool="a")
        a2 = Action(tool="b")
        a1.params["key"] = "val"
        assert "key" not in a2.params


# ── AgentState ──────────────────────────────────────────────────────────


class TestAgentState:
    def test_defaults(self):
        state = AgentState(balance=100.0, burn_rate=0.01, runway_hours=10.0)
        assert state.balance == 100.0
        assert state.burn_rate == 0.01
        assert state.runway_hours == 10.0
        assert state.tools == []
        assert state.recent_actions == []
        assert state.cycle == 0
        assert state.project_context == ""
        assert state.created_resources == {}

    def test_with_tools(self):
        tools = [{"name": "shell:bash", "description": "Run bash", "parameters": {}}]
        state = AgentState(
            balance=50.0, burn_rate=0.02, runway_hours=5.0, tools=tools
        )
        assert len(state.tools) == 1
        assert state.tools[0]["name"] == "shell:bash"


# ── Decision ────────────────────────────────────────────────────────────


class TestDecision:
    def test_defaults(self):
        action = Action(tool="wait")
        decision = Decision(action=action)
        assert decision.action.tool == "wait"
        assert decision.reasoning == ""
        assert decision.api_cost_usd == 0.0
        assert decision.token_usage.total_tokens() == 0

    def test_with_cost(self):
        action = Action(tool="content:write", reasoning="Create a blog post")
        usage = TokenUsage(input_tokens=500, output_tokens=200)
        decision = Decision(
            action=action,
            reasoning="Creating content to generate revenue",
            token_usage=usage,
            api_cost_usd=0.0045,
        )
        assert decision.api_cost_usd == 0.0045
        assert decision.token_usage.total_tokens() == 700


# ── calculate_api_cost ──────────────────────────────────────────────────


class TestCalculateApiCost:
    def test_anthropic_claude_sonnet(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        # input: 1000/1M * 3.0 = 0.003, output: 500/1M * 15.0 = 0.0075
        expected = 0.003 + 0.0075
        assert abs(cost - expected) < 1e-9

    def test_anthropic_haiku(self):
        usage = TokenUsage(input_tokens=10000, output_tokens=2000)
        cost = calculate_api_cost("anthropic", "claude-3-haiku-20240307", usage)
        # input: 10000/1M * 0.25 = 0.0025, output: 2000/1M * 1.25 = 0.0025
        expected = 0.0025 + 0.0025
        assert abs(cost - expected) < 1e-9

    def test_openai_gpt4o(self):
        usage = TokenUsage(input_tokens=5000, output_tokens=1000)
        cost = calculate_api_cost("openai", "gpt-4o", usage)
        # input: 5000/1M * 2.5 = 0.0125, output: 1000/1M * 10.0 = 0.01
        expected = 0.0125 + 0.01
        assert abs(cost - expected) < 1e-9

    def test_openai_gpt4o_mini(self):
        usage = TokenUsage(input_tokens=5000, output_tokens=1000)
        cost = calculate_api_cost("openai", "gpt-4o-mini", usage)
        # input: 5000/1M * 0.15 = 0.00075, output: 1000/1M * 0.6 = 0.0006
        expected = 0.00075 + 0.0006
        assert abs(cost - expected) < 1e-9

    def test_local_vllm_is_free(self):
        usage = TokenUsage(input_tokens=10000, output_tokens=5000)
        cost = calculate_api_cost("vllm", "any-model", usage)
        assert cost == 0.0

    def test_local_transformers_is_free(self):
        usage = TokenUsage(input_tokens=10000, output_tokens=5000)
        cost = calculate_api_cost("transformers", "any-model", usage)
        assert cost == 0.0

    def test_unknown_provider_is_free(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("unknown_provider", "unknown_model", usage)
        assert cost == 0.0

    def test_unknown_model_uses_default(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("anthropic", "claude-future-model", usage)
        # Should use anthropic default: input 3.0, output 15.0
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected) < 1e-9

    def test_zero_tokens(self):
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        assert cost == 0.0

    def test_vertex_gemini_flash(self):
        usage = TokenUsage(input_tokens=2000, output_tokens=1000)
        cost = calculate_api_cost("vertex", "gemini-2.0-flash-001", usage)
        # input: 2000/1M * 0.35 = 0.0007, output: 1000/1M * 1.5 = 0.0015
        expected = 0.0007 + 0.0015
        assert abs(cost - expected) < 1e-9


# ── CognitionEngine._parse_action ──────────────────────────────────────


class TestParseAction:
    """Test the _parse_action method of CognitionEngine."""

    @pytest.fixture
    def engine(self):
        """Create a CognitionEngine with no actual LLM backend."""
        # Use a provider that won't try to connect
        return CognitionEngine(
            llm_provider="none",
            agent_name="TestAgent",
            agent_ticker="TEST",
        )

    def test_parse_valid_json(self, engine):
        response = '{"tool": "shell:bash", "params": {"command": "ls"}, "reasoning": "listing files"}'
        action = engine._parse_action(response)
        assert action.tool == "shell:bash"
        assert action.params == {"command": "ls"}
        assert action.reasoning == "listing files"

    def test_parse_json_with_surrounding_text(self, engine):
        response = 'I think I should run a command.\n{"tool": "shell:bash", "params": {"command": "echo hi"}, "reasoning": "test"}\nThat should work.'
        action = engine._parse_action(response)
        assert action.tool == "shell:bash"
        assert action.params == {"command": "echo hi"}

    def test_parse_json_missing_params(self, engine):
        response = '{"tool": "content:write"}'
        action = engine._parse_action(response)
        assert action.tool == "content:write"
        assert action.params == {}

    def test_parse_json_missing_tool_falls_back(self, engine):
        response = '{"params": {"key": "value"}}'
        action = engine._parse_action(response)
        assert action.tool == "wait"

    def test_parse_fallback_tool_name(self, engine):
        response = "I want to use filesystem:write to save the file."
        action = engine._parse_action(response)
        assert action.tool == "filesystem:write"

    def test_parse_unparseable_response(self, engine):
        response = "I have no idea what to do."
        action = engine._parse_action(response)
        assert action.tool == "wait"
        assert action.reasoning == "Could not parse response"

    def test_parse_empty_response(self, engine):
        action = engine._parse_action("")
        assert action.tool == "wait"

    def test_parse_json_with_reasoning(self, engine):
        response = json.dumps(
            {
                "tool": "github:create_repo",
                "params": {"name": "test-repo", "private": True},
                "reasoning": "Creating a new repo for the project",
            }
        )
        action = engine._parse_action(response)
        assert action.tool == "github:create_repo"
        assert action.params["name"] == "test-repo"
        assert action.params["private"] is True
        assert "Creating a new repo" in action.reasoning


# ── CognitionEngine system prompt ──────────────────────────────────────


class TestSystemPrompt:
    @pytest.fixture
    def engine(self):
        return CognitionEngine(
            llm_provider="none",
            agent_name="TestBot",
            agent_ticker="TBOT",
            agent_type="tester",
            agent_specialty="testing things",
        )

    def test_default_prompt_includes_agent_info(self, engine):
        prompt = engine.get_system_prompt()
        assert "TestBot" in prompt
        assert "TBOT" in prompt
        assert "testing things" in prompt

    def test_custom_system_prompt(self):
        engine = CognitionEngine(
            llm_provider="none",
            system_prompt="You are a custom agent.",
        )
        assert engine.get_system_prompt() == "You are a custom agent."

    def test_set_system_prompt(self, engine):
        engine.set_system_prompt("New prompt")
        assert engine.get_system_prompt() == "New prompt"

    def test_append_to_prompt(self, engine):
        original = engine.get_system_prompt()
        engine.append_to_prompt("EXTRA INSTRUCTION: Be concise.")
        new_prompt = engine.get_system_prompt()
        assert new_prompt.startswith(original)
        assert "EXTRA INSTRUCTION: Be concise." in new_prompt

    def test_set_prompt_clears_additions(self, engine):
        engine.append_to_prompt("Addition 1")
        engine.set_system_prompt("Fresh prompt")
        assert engine.get_system_prompt() == "Fresh prompt"
        assert "Addition 1" not in engine.get_system_prompt()


# ── CognitionEngine model info ─────────────────────────────────────────


class TestModelInfo:
    @pytest.fixture
    def engine(self):
        return CognitionEngine(
            llm_provider="none",
            llm_model="test-model",
            agent_name="TestAgent",
        )

    def test_get_current_model(self, engine):
        info = engine.get_current_model()
        assert info["model"] == "test-model"
        assert info["finetuned"] is False
        assert info["finetuned_model_id"] is None

    def test_is_local_model_false_for_none(self, engine):
        assert engine.is_local_model() is False


# ── CognitionEngine training examples ──────────────────────────────────


class TestTrainingExamples:
    @pytest.fixture
    def engine(self):
        return CognitionEngine(llm_provider="none", agent_name="TestAgent")

    def test_record_and_get_examples(self, engine):
        engine.record_training_example("prompt1", "response1", "success")
        engine.record_training_example("prompt2", "response2", "failure")
        all_examples = engine.get_training_examples()
        assert len(all_examples) == 2

    def test_filter_by_outcome(self, engine):
        engine.record_training_example("p1", "r1", "success")
        engine.record_training_example("p2", "r2", "failure")
        engine.record_training_example("p3", "r3", "success")
        successes = engine.get_training_examples("success")
        assert len(successes) == 2

    def test_clear_examples(self, engine):
        engine.record_training_example("p1", "r1")
        engine.record_training_example("p2", "r2")
        count = engine.clear_training_examples()
        assert count == 2
        assert len(engine.get_training_examples()) == 0

    def test_export_training_data(self, engine):
        engine.record_training_example("prompt1", "response1", "success")
        exported = engine.export_training_data()
        lines = exported.strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["messages"][1]["content"] == "prompt1"
        assert data["messages"][2]["content"] == "response1"

    def test_export_only_successes(self, engine):
        engine.record_training_example("p1", "r1", "success")
        engine.record_training_example("p2", "r2", "failure")
        exported = engine.export_training_data()
        lines = [line for line in exported.strip().split("\n") if line]
        assert len(lines) == 1
