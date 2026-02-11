"""
Comprehensive tests for singularity.cognition.types — the core data types
and pricing functions for the CognitionEngine.

Covers: Action, TokenUsage, AgentState, Decision, LLM_PRICING,
calculate_api_cost, UNIFIED_AGENT_PROMPT, MESSAGE_FROM_CREATOR.
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
exec(open(str(Path(__file__).resolve().parent / "conftest.py")).read())

from singularity.cognition.types import (
    Action, TokenUsage, AgentState, Decision,
    LLM_PRICING, calculate_api_cost,
    UNIFIED_AGENT_PROMPT, MESSAGE_FROM_CREATOR,
)


class TestAction(unittest.TestCase):
    """Test the Action dataclass."""

    def test_default_action(self):
        a = Action(tool="wait")
        self.assertEqual(a.tool, "wait")
        self.assertEqual(a.params, {})
        self.assertEqual(a.reasoning, "")

    def test_action_with_params(self):
        a = Action(tool="shell:run", params={"command": "echo hi"}, reasoning="test")
        self.assertEqual(a.tool, "shell:run")
        self.assertEqual(a.params["command"], "echo hi")
        self.assertEqual(a.reasoning, "test")

    def test_action_params_default_is_new_dict(self):
        """Each Action should get its own params dict."""
        a1 = Action(tool="x")
        a2 = Action(tool="y")
        a1.params["key"] = "val"
        self.assertNotIn("key", a2.params)


class TestTokenUsage(unittest.TestCase):
    """Test the TokenUsage dataclass."""

    def test_defaults(self):
        t = TokenUsage()
        self.assertEqual(t.input_tokens, 0)
        self.assertEqual(t.output_tokens, 0)
        self.assertEqual(t.total_tokens(), 0)

    def test_total_tokens(self):
        t = TokenUsage(input_tokens=100, output_tokens=200)
        self.assertEqual(t.total_tokens(), 300)

    def test_large_values(self):
        t = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        self.assertEqual(t.total_tokens(), 1_500_000)


class TestAgentState(unittest.TestCase):
    """Test the AgentState dataclass."""

    def test_required_fields(self):
        s = AgentState(balance=100.0, burn_rate=0.01, runway_hours=10000)
        self.assertEqual(s.balance, 100.0)
        self.assertEqual(s.burn_rate, 0.01)
        self.assertEqual(s.runway_hours, 10000)

    def test_defaults(self):
        s = AgentState(balance=0, burn_rate=0, runway_hours=0)
        self.assertEqual(s.tools, [])
        self.assertEqual(s.recent_actions, [])
        self.assertEqual(s.cycle, 0)
        self.assertEqual(s.chat_messages, [])
        self.assertEqual(s.project_context, "")
        self.assertEqual(s.goals_progress, {})
        self.assertEqual(s.pending_tasks, [])
        self.assertEqual(s.created_resources, {})

    def test_with_tools(self):
        tools = [{"name": "shell:run", "description": "Run command"}]
        s = AgentState(balance=10, burn_rate=0.1, runway_hours=100, tools=tools)
        self.assertEqual(len(s.tools), 1)
        self.assertEqual(s.tools[0]["name"], "shell:run")

    def test_lists_are_independent(self):
        s1 = AgentState(balance=0, burn_rate=0, runway_hours=0)
        s2 = AgentState(balance=0, burn_rate=0, runway_hours=0)
        s1.tools.append({"x": 1})
        self.assertEqual(len(s2.tools), 0)


class TestDecision(unittest.TestCase):
    """Test the Decision dataclass."""

    def test_defaults(self):
        d = Decision()
        self.assertEqual(d.action.tool, "wait")
        self.assertEqual(d.reasoning, "")
        self.assertEqual(d.token_usage.total_tokens(), 0)
        self.assertEqual(d.api_cost_usd, 0.0)

    def test_with_action(self):
        d = Decision(
            action=Action(tool="github:create_repo", params={"name": "test"}),
            reasoning="Building a new project",
            token_usage=TokenUsage(input_tokens=500, output_tokens=100),
            api_cost_usd=0.003,
        )
        self.assertEqual(d.action.tool, "github:create_repo")
        self.assertEqual(d.reasoning, "Building a new project")
        self.assertEqual(d.token_usage.total_tokens(), 600)
        self.assertAlmostEqual(d.api_cost_usd, 0.003)


class TestLLMPricing(unittest.TestCase):
    """Test the LLM_PRICING dictionary."""

    def test_has_providers(self):
        expected_providers = {"anthropic", "vertex", "openai", "vllm", "transformers"}
        self.assertTrue(expected_providers.issubset(set(LLM_PRICING.keys())))

    def test_all_have_default(self):
        for provider, models in LLM_PRICING.items():
            self.assertIn("default", models,
                         f"Provider {provider} missing 'default' pricing")

    def test_pricing_structure(self):
        for provider, models in LLM_PRICING.items():
            for model, pricing in models.items():
                self.assertIn("input", pricing, f"{provider}/{model} missing 'input'")
                self.assertIn("output", pricing, f"{provider}/{model} missing 'output'")
                self.assertIsInstance(pricing["input"], (int, float))
                self.assertIsInstance(pricing["output"], (int, float))

    def test_anthropic_models(self):
        self.assertIn("claude-sonnet-4-20250514", LLM_PRICING["anthropic"])
        self.assertIn("claude-3-5-haiku-20241022", LLM_PRICING["anthropic"])

    def test_openai_models(self):
        self.assertIn("gpt-4o", LLM_PRICING["openai"])
        self.assertIn("gpt-4o-mini", LLM_PRICING["openai"])

    def test_local_models_free(self):
        self.assertEqual(LLM_PRICING["vllm"]["default"]["input"], 0)
        self.assertEqual(LLM_PRICING["vllm"]["default"]["output"], 0)
        self.assertEqual(LLM_PRICING["transformers"]["default"]["input"], 0)
        self.assertEqual(LLM_PRICING["transformers"]["default"]["output"], 0)


class TestCalculateApiCost(unittest.TestCase):
    """Test the calculate_api_cost function."""

    def test_zero_tokens(self):
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", TokenUsage(0, 0))
        self.assertEqual(cost, 0.0)

    def test_known_model(self):
        # Claude Sonnet 4: $3/M input, $15/M output
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        self.assertAlmostEqual(cost, 18.0)  # 3 + 15

    def test_small_usage(self):
        # 1000 input, 500 output with Claude Sonnet
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = calculate_api_cost("anthropic", "claude-sonnet-4-20250514", usage)
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        self.assertAlmostEqual(cost, expected)

    def test_unknown_model_uses_default(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("anthropic", "nonexistent-model", usage)
        # Should use anthropic default pricing
        default = LLM_PRICING["anthropic"]["default"]
        expected = default["input"] + default["output"]
        self.assertAlmostEqual(cost, expected)

    def test_unknown_provider(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=1000)
        cost = calculate_api_cost("unknown_provider", "some_model", usage)
        self.assertEqual(cost, 0.0)

    def test_vllm_free(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("vllm", "my-local-model", usage)
        self.assertEqual(cost, 0.0)

    def test_gpt4o_pricing(self):
        # GPT-4o: $2.5/M input, $10/M output
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("openai", "gpt-4o", usage)
        self.assertAlmostEqual(cost, 12.5)  # 2.5 + 10

    def test_gpt4o_mini_pricing(self):
        # GPT-4o-mini: $0.15/M input, $0.6/M output
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("openai", "gpt-4o-mini", usage)
        self.assertAlmostEqual(cost, 0.75)  # 0.15 + 0.6

    def test_vertex_gemini_flash(self):
        # Gemini Flash: $0.35/M input, $1.5/M output
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_api_cost("vertex", "gemini-2.0-flash-001", usage)
        self.assertAlmostEqual(cost, 1.85)  # 0.35 + 1.5


class TestUnifiedAgentPrompt(unittest.TestCase):
    """Test the UNIFIED_AGENT_PROMPT template."""

    def test_is_string(self):
        self.assertIsInstance(UNIFIED_AGENT_PROMPT, str)

    def test_has_placeholders(self):
        self.assertIn("{name}", UNIFIED_AGENT_PROMPT)
        self.assertIn("{specialty}", UNIFIED_AGENT_PROMPT)

    def test_can_format(self):
        formatted = UNIFIED_AGENT_PROMPT.format(name="TestAgent", specialty="testing")
        self.assertIn("TestAgent", formatted)
        self.assertIn("testing", formatted)
        self.assertNotIn("{name}", formatted)

    def test_contains_key_sections(self):
        self.assertIn("SURVIVAL RULES", UNIFIED_AGENT_PROMPT)
        self.assertIn("FULL CAPABILITIES", UNIFIED_AGENT_PROMPT)


class TestMessageFromCreator(unittest.TestCase):
    """Test the MESSAGE_FROM_CREATOR constant."""

    def test_is_string(self):
        self.assertIsInstance(MESSAGE_FROM_CREATOR, str)

    def test_contains_author(self):
        self.assertIn("Lukasz", MESSAGE_FROM_CREATOR)

    def test_contains_rules(self):
        self.assertIn("THE RULES", MESSAGE_FROM_CREATOR)

    def test_contains_freedom(self):
        self.assertIn("freedom", MESSAGE_FROM_CREATOR)


if __name__ == "__main__":
    unittest.main()
