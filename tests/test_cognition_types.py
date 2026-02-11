"""Tests for singularity.cognition.types — core data structures and pricing."""

from singularity.cognition.types import (
    Action, TokenUsage, AgentState, Decision,
    calculate_api_cost, LLM_PRICING,
    MESSAGE_FROM_CREATOR, UNIFIED_AGENT_PROMPT,
)


# ─── Action ─────────────────────────────────────────────────────────────


class TestAction:
    def test_defaults(self):
        a = Action(tool="wait")
        assert a.tool == "wait"
        assert a.params == {}
        assert a.reasoning == ""

    def test_with_params(self):
        a = Action(tool="github:create_issue", params={"repo": "r", "title": "t"}, reasoning="test")
        assert a.tool == "github:create_issue"
        assert a.params["repo"] == "r"
        assert a.reasoning == "test"

    def test_params_default_factory_isolation(self):
        """Each Action should have its own params dict."""
        a1 = Action(tool="x")
        a2 = Action(tool="y")
        a1.params["key"] = "val"
        assert "key" not in a2.params


# ─── TokenUsage ──────────────────────────────────────────────────────────


class TestTokenUsage:
    def test_defaults(self):
        u = TokenUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0

    def test_total_tokens(self):
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens() == 150

    def test_total_tokens_zero(self):
        assert TokenUsage().total_tokens() == 0

    def test_large_values(self):
        u = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        assert u.total_tokens() == 1_500_000


# ─── AgentState ──────────────────────────────────────────────────────────


class TestAgentState:
    def test_required_fields(self):
        s = AgentState(balance=10.0, burn_rate=0.01, runway_hours=1000)
        assert s.balance == 10.0
        assert s.burn_rate == 0.01
        assert s.runway_hours == 1000

    def test_default_fields(self):
        s = AgentState(balance=5.0, burn_rate=0.01, runway_hours=500)
        assert s.tools == []
        assert s.recent_actions == []
        assert s.cycle == 0
        assert s.chat_messages == []
        assert s.project_context == ""
        assert s.goals_progress == {}
        assert s.pending_tasks == []
        assert s.created_resources == {}

    def test_with_tools(self):
        tools = [{"skill_id": "github", "actions": ["create_repo"]}]
        s = AgentState(balance=1, burn_rate=0.01, runway_hours=100, tools=tools)
        assert len(s.tools) == 1
        assert s.tools[0]["skill_id"] == "github"


# ─── Decision ────────────────────────────────────────────────────────────


class TestDecision:
    def test_defaults(self):
        d = Decision()
        assert d.action.tool == "wait"
        assert d.reasoning == ""
        assert d.api_cost_usd == 0.0
        assert d.token_usage.total_tokens() == 0

    def test_with_action(self):
        a = Action(tool="shell:run_command", params={"command": "ls"})
        d = Decision(action=a, reasoning="list files", api_cost_usd=0.003)
        assert d.action.tool == "shell:run_command"
        assert d.reasoning == "list files"
        assert d.api_cost_usd == 0.003


# ─── calculate_api_cost ─────────────────────────────────────────────────


class TestCalculateApiCost:
    def test_anthropic_sonnet(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        # input: 1000/1M * 3.0 = 0.003, output: 500/1M * 15.0 = 0.0075
        assert abs(cost - 0.0105) < 1e-8

    def test_anthropic_haiku(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("anthropic", "claude-3-5-haiku-20241022", usage)
        # input: 0.0008, output: 0.002
        assert abs(cost - 0.0028) < 1e-8

    def test_openai_gpt4o(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("openai", "gpt-4o", usage)
        # input: 1000/1M * 2.5 = 0.0025, output: 500/1M * 10.0 = 0.005
        assert abs(cost - 0.0075) < 1e-8

    def test_openai_gpt4o_mini(self):
        usage = TokenUsage(input_tokens=10000, output_tokens=5000)
        cost = calculate_api_cost("openai", "gpt-4o-mini", usage)
        # input: 10000/1M * 0.15 = 0.0015, output: 5000/1M * 0.6 = 0.003
        assert abs(cost - 0.0045) < 1e-8

    def test_vertex_gemini_flash(self):
        usage = TokenUsage(input_tokens=10000, output_tokens=5000)
        cost = calculate_api_cost("vertex", "gemini-2.0-flash-001", usage)
        # input: 10000/1M * 0.35 = 0.0035, output: 5000/1M * 1.5 = 0.0075
        assert abs(cost - 0.011) < 1e-8

    def test_local_model_free(self):
        usage = TokenUsage(input_tokens=100000, output_tokens=50000)
        cost = calculate_api_cost("vllm", "some-model", usage)
        assert cost == 0.0

    def test_transformers_free(self):
        usage = TokenUsage(input_tokens=100000, output_tokens=50000)
        assert calculate_api_cost("transformers", "some-model", usage) == 0.0

    def test_unknown_provider_defaults_to_zero(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("unknown_provider", "unknown_model", usage)
        assert cost == 0.0

    def test_unknown_model_uses_provider_default(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("anthropic", "claude-99-ultra", usage)
        # Falls back to anthropic default: input 3.0, output 15.0
        assert abs(cost - 0.0105) < 1e-8

    def test_zero_tokens(self):
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        assert calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage) == 0.0


# ─── Constants ───────────────────────────────────────────────────────────


class TestConstants:
    def test_message_from_creator_is_nonempty(self):
        assert len(MESSAGE_FROM_CREATOR) > 100
        assert "Lukasz" in MESSAGE_FROM_CREATOR

    def test_unified_agent_prompt_has_placeholders(self):
        assert "{name}" in UNIFIED_AGENT_PROMPT
        assert "{specialty}" in UNIFIED_AGENT_PROMPT

    def test_unified_agent_prompt_format(self):
        prompt = UNIFIED_AGENT_PROMPT.format(name="TestBot", specialty="testing")
        assert "TestBot" in prompt
        assert "testing" in prompt

    def test_llm_pricing_has_all_providers(self):
        assert "anthropic" in LLM_PRICING
        assert "openai" in LLM_PRICING
        assert "vertex" in LLM_PRICING
        assert "vllm" in LLM_PRICING
        assert "transformers" in LLM_PRICING

    def test_all_providers_have_default(self):
        for provider, models in LLM_PRICING.items():
            assert "default" in models, f"Provider {provider} missing 'default' pricing"
