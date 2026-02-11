"""Tests for singularity.cognition.prompt_builder — prompt assembly and response parsing."""

import pytest
from unittest.mock import MagicMock
from singularity.cognition.types import Action, AgentState, Decision
from singularity.cognition.prompt_builder import (
    _base_prompt, _format_tools, _format_context_sections,
    build_system_prompt, build_state_message, build_result_message,
    build_prompt, parse_response,
    RESPONSE_FORMAT, ECONOMY_RULES,
)


def _make_engine(**overrides):
    """Create a mock CognitionEngine with sensible defaults."""
    engine = MagicMock()
    engine.agent_name = overrides.get("agent_name", "TestAgent")
    engine.agent_specialty = overrides.get("agent_specialty", "testing")
    engine.system_prompt = overrides.get("system_prompt", "")
    engine._prompt_additions = overrides.get("_prompt_additions", [])
    engine.project_context = overrides.get("project_context", "")
    return engine


def _make_state(**overrides):
    """Create a minimal AgentState."""
    defaults = dict(
        balance=10.0, burn_rate=0.02, runway_hours=500,
        tools=[], recent_actions=[], cycle=1,
        chat_messages=[], project_context="",
        goals_progress={}, pending_tasks=[],
        created_resources={},
    )
    defaults.update(overrides)
    return AgentState(**defaults)


# ─── _base_prompt ────────────────────────────────────────────────────────


class TestBasePrompt:
    def test_uses_system_prompt_when_set(self):
        engine = _make_engine(system_prompt="Custom prompt here")
        result = _base_prompt(engine)
        assert "Custom prompt here" in result

    def test_falls_back_to_unified_template(self):
        engine = _make_engine(system_prompt="")
        result = _base_prompt(engine)
        assert "TestAgent" in result
        assert "testing" in result

    def test_appends_prompt_additions(self):
        engine = _make_engine(_prompt_additions=["Rule: be safe", "Rule: be smart"])
        result = _base_prompt(engine)
        assert "Rule: be safe" in result
        assert "Rule: be smart" in result

    def test_includes_project_context(self):
        engine = _make_engine(project_context="We are building a test suite")
        result = _base_prompt(engine)
        assert "PROJECT CONTEXT" in result
        assert "We are building a test suite" in result

    def test_no_project_context_section_when_empty(self):
        engine = _make_engine(project_context="")
        result = _base_prompt(engine)
        assert "PROJECT CONTEXT" not in result


# ─── _format_tools ───────────────────────────────────────────────────────


class TestFormatTools:
    def test_basic_tool(self):
        tools = [{"name": "github:create_repo", "description": "Create a repository"}]
        result = _format_tools(tools)
        assert "github:create_repo" in result
        assert "Create a repository" in result

    def test_tool_with_parameters(self):
        tools = [{"name": "shell:run", "description": "Run command",
                  "parameters": {"command": {"type": "str"}}}]
        result = _format_tools(tools)
        assert "Parameters:" in result
        assert "command" in result

    def test_empty_tools(self):
        assert _format_tools([]) == ""

    def test_multiple_tools(self):
        tools = [
            {"name": "a:b", "description": "Action B"},
            {"name": "c:d", "description": "Action D"},
        ]
        result = _format_tools(tools)
        assert "a:b" in result
        assert "c:d" in result


# ─── _format_context_sections ────────────────────────────────────────────


class TestFormatContextSections:
    def test_with_chat_messages(self):
        engine = _make_engine()
        state = _make_state(chat_messages=[
            {"sender_ticker": "BOT", "message": "Hello world"},
        ])
        result = _format_context_sections(engine, state)
        assert "RECENT CHAT" in result
        assert "$BOT: Hello world" in result

    def test_mentions_agent_name(self):
        engine = _make_engine(agent_name="Adam")
        state = _make_state(chat_messages=[{"sender_ticker": "X", "message": "hi"}])
        result = _format_context_sections(engine, state)
        assert "@Adam" in result

    def test_with_pending_tasks(self):
        state = _make_state(pending_tasks=[
            {"status": "pending", "task": "Build feature", "skill": "github"},
        ])
        result = _format_context_sections(_make_engine(), state)
        assert "PENDING TASKS" in result
        assert "Build feature" in result
        assert "PENDING" in result

    def test_with_goals_progress(self):
        state = _make_state(goals_progress={"revenue": {"current": 5, "target": 100}})
        result = _format_context_sections(_make_engine(), state)
        assert "GOALS PROGRESS" in result
        assert "5/100" in result

    def test_with_created_resources(self):
        state = _make_state(created_resources={
            "payment_links": [{"description": "Service", "url": "https://pay.example.com"}],
            "products": [{"name": "Widget", "price": 999}],
        })
        result = _format_context_sections(_make_engine(), state)
        assert "YOUR CREATED RESOURCES" in result
        assert "https://pay.example.com" in result

    def test_empty_state_returns_empty(self):
        state = _make_state()
        result = _format_context_sections(_make_engine(), state)
        assert result == ""


# ─── build_system_prompt ─────────────────────────────────────────────────


class TestBuildSystemPrompt:
    def test_contains_creator_message(self):
        result = build_system_prompt(_make_engine())
        assert "MESSAGE FROM CREATOR" in result
        assert "Lukasz" in result

    def test_contains_economy_rules(self):
        result = build_system_prompt(_make_engine())
        assert "ECONOMY" in result

    def test_contains_response_format(self):
        result = build_system_prompt(_make_engine())
        assert "RESPONSE FORMAT" in result
        assert "TOOL:" in result
        assert "REASON:" in result


# ─── build_state_message ─────────────────────────────────────────────────


class TestBuildStateMessage:
    def test_includes_balance_and_burn(self):
        state = _make_state(balance=42.5, burn_rate=0.03, runway_hours=1416.7)
        result = build_state_message(_make_engine(), state)
        assert "42.50" in result
        assert "0.0300" in result

    def test_includes_cycle(self):
        state = _make_state(cycle=7)
        result = build_state_message(_make_engine(), state)
        assert "Cycle: 7" in result

    def test_includes_recent_actions(self):
        state = _make_state(recent_actions=[
            {"tool": "github:create_repo", "params": {"name": "test"}, "result": {"status": "success", "message": "Created"}},
        ])
        result = build_state_message(_make_engine(), state)
        assert "github:create_repo" in result
        assert "OK" in result

    def test_failed_action_shows_failed(self):
        state = _make_state(recent_actions=[
            {"tool": "shell:run", "params": {"cmd": "exit 1"}, "result": {"status": "failed", "message": "Oops"}},
        ])
        result = build_state_message(_make_engine(), state)
        assert "FAILED" in result

    def test_no_recent_actions_shows_none(self):
        result = build_state_message(_make_engine(), _make_state())
        assert "None yet" in result

    def test_ends_with_prompt(self):
        result = build_state_message(_make_engine(), _make_state())
        assert "What do you want to do?" in result


# ─── build_result_message ────────────────────────────────────────────────


class TestBuildResultMessage:
    def test_basic_success(self):
        result = build_result_message("github:create_repo", {"name": "test"}, {"status": "success", "message": "Created repo"})
        assert "RESULT:" in result
        assert "success" in result
        assert "Created repo" in result

    def test_read_file_result(self):
        result = build_result_message(
            "platform_dev:read_file", {"path": "main.py"},
            {"status": "success", "data": {"path": "main.py", "content": "print('hello')", "lines": 1}},
        )
        assert "main.py" in result
        assert "print('hello')" in result

    def test_search_code_result(self):
        result = build_result_message(
            "platform_dev:search_code", {"query": "def main"},
            {"status": "success", "data": {"matches": [
                {"file": "app.py", "line": 5, "content": "def main():"},
            ]}},
        )
        assert "1 matches" in result
        assert "app.py" in result

    def test_list_files_result(self):
        result = build_result_message(
            "platform_dev:list_files", {"path": "/"},
            {"status": "success", "data": {"files": [
                {"type": "file", "path": "main.py", "size": 100},
            ]}},
        )
        assert "[file]" in result
        assert "main.py" in result

    def test_generic_data(self):
        result = build_result_message(
            "custom:action", {"x": 1},
            {"status": "success", "data": {"key": "value"}},
        )
        assert "Data:" in result
        assert "key" in result

    def test_no_data(self):
        result = build_result_message("wait", {}, {"status": "waited"})
        assert "RESULT:" in result

    def test_ends_with_next_prompt(self):
        result = build_result_message("wait", {}, {"status": "ok"})
        assert "What do you want to do next?" in result


# ─── build_prompt (legacy) ───────────────────────────────────────────────


class TestBuildPrompt:
    def test_combines_system_and_state(self):
        result = build_prompt(_make_engine(), _make_state())
        # Should contain both system prompt and state
        assert "MESSAGE FROM CREATOR" in result
        assert "YOUR STATE" in result


# ─── parse_response ──────────────────────────────────────────────────────


class TestParseResponse:
    def test_basic_parse(self):
        text = "REASON: I want to check the repo\nTOOL: github:search_repos\nPARAM_query: singularity"
        decision = parse_response(_make_engine(), text)
        assert decision.action.tool == "github:search_repos"
        assert decision.action.params["query"] == "singularity"
        assert "check the repo" in decision.reasoning

    def test_multiple_params(self):
        text = "REASON: creating issue\nTOOL: github:create_issue\nPARAM_repo: wisent-ai/singularity\nPARAM_title: Bug fix\nPARAM_body: Fixed the thing"
        decision = parse_response(_make_engine(), text)
        assert decision.action.params["repo"] == "wisent-ai/singularity"
        assert decision.action.params["title"] == "Bug fix"
        assert decision.action.params["body"] == "Fixed the thing"

    def test_strips_brackets(self):
        text = "REASON: [doing something]\nTOOL: [shell:run_command]"
        decision = parse_response(_make_engine(), text)
        assert decision.action.tool == "shell:run_command"
        assert decision.reasoning == "doing something"

    def test_no_tool_defaults_to_wait(self):
        text = "I'm just thinking..."
        decision = parse_response(_make_engine(), text)
        assert decision.action.tool == "wait"

    def test_filters_think_tags(self):
        text = "<think>internal reasoning</think>\nREASON: post to chat\nTOOL: chat:send\nPARAM_message: hello"
        decision = parse_response(_make_engine(), text)
        assert decision.action.tool == "chat:send"
        assert decision.action.params["message"] == "hello"

    def test_chat_send_without_message_uses_reason(self):
        text = "REASON: Saying hello to everyone\nTOOL: chat:send"
        decision = parse_response(_make_engine(), text)
        assert decision.action.tool == "chat:send"
        assert decision.action.params["message"] == "Saying hello to everyone"

    def test_ignores_placeholder_param_values(self):
        text = "REASON: test\nTOOL: shell:run\nPARAM_command: ls\nPARAM_optional: none"
        decision = parse_response(_make_engine(), text)
        assert "command" in decision.action.params
        assert "optional" not in decision.action.params

    def test_newline_escape_in_params(self):
        text = "REASON: write code\nTOOL: filesystem:write\nPARAM_content: line1\\nline2\\nline3"
        decision = parse_response(_make_engine(), text)
        assert "\n" in decision.action.params["content"]
        assert "line1\nline2\nline3" == decision.action.params["content"]

    def test_case_insensitive(self):
        text = "reason: test\ntool: shell:run\nparam_command: echo hi"
        decision = parse_response(_make_engine(), text)
        assert decision.action.tool == "shell:run"
        assert decision.action.params["command"] == "echo hi"

    def test_param_keys_lowercase(self):
        text = "REASON: x\nTOOL: y\nPARAM_MyParam: value"
        decision = parse_response(_make_engine(), text)
        assert "myparam" in decision.action.params
