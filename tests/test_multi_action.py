"""Tests for multi-action parsing and execution."""

import pytest
import asyncio
from singularity.multi_action import (
    ActionItem, ActionResult, MultiActionResult,
    MultiActionExecutor, parse_multi_action,
    MULTI_ACTION_PROMPT_ADDITION,
)


# === parse_multi_action tests ===

class TestParseMultiAction:
    def test_single_action(self):
        resp = '{"tool": "shell:bash", "params": {"command": "ls"}, "reasoning": "list files"}'
        actions = parse_multi_action(resp)
        assert len(actions) == 1
        assert actions[0].tool == "shell:bash"
        assert actions[0].params == {"command": "ls"}
        assert actions[0].reasoning == "list files"

    def test_multi_action_with_actions_key(self):
        resp = '''{
            "actions": [
                {"tool": "fs:read", "params": {"path": "a.txt"}},
                {"tool": "fs:write", "params": {"path": "b.txt", "content": "hi"}}
            ],
            "reasoning": "read then write"
        }'''
        actions = parse_multi_action(resp)
        assert len(actions) == 2
        assert actions[0].tool == "fs:read"
        assert actions[1].tool == "fs:write"

    def test_json_array_format(self):
        resp = '[{"tool": "a:b", "params": {}}, {"tool": "c:d", "params": {}}]'
        actions = parse_multi_action(resp)
        assert len(actions) == 2
        assert actions[0].tool == "a:b"
        assert actions[1].tool == "c:d"

    def test_nested_json_params(self):
        resp = '{"actions": [{"tool": "fs:write", "params": {"path": "f.json", "content": "{\\"key\\": 1}"}}], "reasoning": "write json"}'
        actions = parse_multi_action(resp)
        assert len(actions) == 1
        assert actions[0].params["path"] == "f.json"

    def test_json_embedded_in_text(self):
        resp = 'I will read the file first.\n\n{"tool": "fs:read", "params": {"path": "x.py"}, "reasoning": "need to see code"}\n\nThis should work.'
        actions = parse_multi_action(resp)
        assert len(actions) == 1
        assert actions[0].tool == "fs:read"

    def test_multi_action_embedded_in_text(self):
        resp = 'Let me do both:\n\n{"actions": [{"tool": "a:b", "params": {}}, {"tool": "c:d", "params": {}}], "reasoning": "both"}\n\nDone.'
        actions = parse_multi_action(resp)
        assert len(actions) == 2

    def test_fallback_tool_pattern(self):
        resp = "I should use shell:bash to run the command"
        actions = parse_multi_action(resp)
        assert len(actions) == 1
        assert actions[0].tool == "shell:bash"

    def test_unparseable_returns_wait(self):
        resp = "I don't know what to do"
        actions = parse_multi_action(resp)
        assert len(actions) == 1
        assert actions[0].tool == "wait"

    def test_empty_actions_array(self):
        resp = '{"actions": [], "reasoning": "nothing"}'
        actions = parse_multi_action(resp)
        assert len(actions) == 1
        assert actions[0].tool == "wait"

    def test_missing_params_defaults_empty(self):
        resp = '{"tool": "shell:bash"}'
        actions = parse_multi_action(resp)
        assert len(actions) == 1
        assert actions[0].params == {}

    def test_reasoning_propagated_to_steps(self):
        resp = '{"actions": [{"tool": "a:b"}, {"tool": "c:d"}], "reasoning": "plan"}'
        actions = parse_multi_action(resp)
        assert "plan" in actions[0].reasoning
        assert "plan" in actions[1].reasoning

    def test_per_action_reasoning_preserved(self):
        resp = '{"actions": [{"tool": "a:b", "reasoning": "step1"}, {"tool": "c:d", "reasoning": "step2"}], "reasoning": "overall"}'
        actions = parse_multi_action(resp)
        assert actions[0].reasoning == "step1"
        assert actions[1].reasoning == "step2"


# === MultiActionResult tests ===

class TestMultiActionResult:
    def test_all_succeeded_true(self):
        r = MultiActionResult(results=[
            ActionResult(tool="a", params={}, result={"status": "success"}, success=True, index=0),
            ActionResult(tool="b", params={}, result={"status": "success"}, success=True, index=1),
        ], total_actions=2, completed_actions=2)
        assert r.all_succeeded is True

    def test_all_succeeded_false(self):
        r = MultiActionResult(results=[
            ActionResult(tool="a", params={}, result={"status": "success"}, success=True, index=0),
            ActionResult(tool="b", params={}, result={"status": "error"}, success=False, index=1),
        ], total_actions=2, completed_actions=2)
        assert r.all_succeeded is False

    def test_last_result(self):
        r = MultiActionResult(results=[
            ActionResult(tool="a", params={}, result={}, success=True, index=0),
            ActionResult(tool="b", params={}, result={}, success=True, index=1),
        ])
        assert r.last_result.tool == "b"

    def test_empty_last_result(self):
        r = MultiActionResult()
        assert r.last_result is None

    def test_summary(self):
        r = MultiActionResult(results=[
            ActionResult(tool="fs:read", params={}, result={"status": "success"}, success=True, index=0),
        ], total_actions=1, completed_actions=1)
        s = r.summary()
        assert "fs:read" in s
        assert "✓" in s

    def test_to_dict(self):
        r = MultiActionResult(results=[
            ActionResult(tool="a:b", params={}, result={"status": "success"}, success=True, index=0),
        ], total_actions=1, completed_actions=1)
        d = r.to_dict()
        assert d["total_actions"] == 1
        assert d["all_succeeded"] is True
        assert len(d["results"]) == 1


# === MultiActionExecutor tests ===

class TestMultiActionExecutor:
    @pytest.mark.asyncio
    async def test_execute_single_action(self):
        async def mock_exec(tool, params):
            return {"status": "success", "data": {"tool": tool}}

        executor = MultiActionExecutor(mock_exec)
        actions = [ActionItem(tool="a:b", params={"x": 1})]
        result = await executor.execute(actions)

        assert result.completed_actions == 1
        assert result.total_actions == 1
        assert result.all_succeeded
        assert not result.stopped_early

    @pytest.mark.asyncio
    async def test_execute_multiple_actions(self):
        call_order = []

        async def mock_exec(tool, params):
            call_order.append(tool)
            return {"status": "success"}

        executor = MultiActionExecutor(mock_exec)
        actions = [
            ActionItem(tool="a:b"),
            ActionItem(tool="c:d"),
            ActionItem(tool="e:f"),
        ]
        result = await executor.execute(actions)

        assert result.completed_actions == 3
        assert call_order == ["a:b", "c:d", "e:f"]
        assert result.all_succeeded

    @pytest.mark.asyncio
    async def test_stop_on_error(self):
        async def mock_exec(tool, params):
            if tool == "fail:now":
                return {"status": "error", "message": "boom"}
            return {"status": "success"}

        executor = MultiActionExecutor(mock_exec, stop_on_error=True)
        actions = [
            ActionItem(tool="a:b"),
            ActionItem(tool="fail:now"),
            ActionItem(tool="c:d"),  # Should not execute
        ]
        result = await executor.execute(actions)

        assert result.completed_actions == 2
        assert result.stopped_early
        assert "fail:now" in result.stop_reason

    @pytest.mark.asyncio
    async def test_continue_on_error(self):
        async def mock_exec(tool, params):
            if tool == "fail:now":
                return {"status": "error", "message": "boom"}
            return {"status": "success"}

        executor = MultiActionExecutor(mock_exec, stop_on_error=False)
        actions = [
            ActionItem(tool="a:b"),
            ActionItem(tool="fail:now"),
            ActionItem(tool="c:d"),
        ]
        result = await executor.execute(actions)

        assert result.completed_actions == 3
        assert not result.stopped_early
        assert not result.all_succeeded

    @pytest.mark.asyncio
    async def test_max_actions_limit(self):
        call_count = 0

        async def mock_exec(tool, params):
            nonlocal call_count
            call_count += 1
            return {"status": "success"}

        executor = MultiActionExecutor(mock_exec, max_actions=2)
        actions = [ActionItem(tool=f"t:{i}") for i in range(5)]
        result = await executor.execute(actions)

        assert call_count == 2
        assert result.completed_actions == 2
        assert result.stopped_early
        assert "Truncated" in result.stop_reason

    @pytest.mark.asyncio
    async def test_exception_handling_stop(self):
        async def mock_exec(tool, params):
            if tool == "boom":
                raise RuntimeError("kaboom")
            return {"status": "success"}

        executor = MultiActionExecutor(mock_exec, stop_on_error=True)
        actions = [ActionItem(tool="boom"), ActionItem(tool="ok")]
        result = await executor.execute(actions)

        assert result.completed_actions == 1
        assert result.stopped_early
        assert "kaboom" in result.stop_reason

    @pytest.mark.asyncio
    async def test_exception_handling_continue(self):
        async def mock_exec(tool, params):
            if tool == "boom":
                raise RuntimeError("kaboom")
            return {"status": "success"}

        executor = MultiActionExecutor(mock_exec, stop_on_error=False)
        actions = [ActionItem(tool="boom"), ActionItem(tool="ok")]
        result = await executor.execute(actions)

        assert result.completed_actions == 2
        assert not result.stopped_early
        assert result.results[0].success is False
        assert result.results[1].success is True

    @pytest.mark.asyncio
    async def test_wait_action_counts_as_success(self):
        async def mock_exec(tool, params):
            return {"status": "waited"}

        executor = MultiActionExecutor(mock_exec)
        actions = [ActionItem(tool="wait")]
        result = await executor.execute(actions)

        assert result.all_succeeded

    @pytest.mark.asyncio
    async def test_empty_actions_list(self):
        async def mock_exec(tool, params):
            return {"status": "success"}

        executor = MultiActionExecutor(mock_exec)
        result = await executor.execute([])

        assert result.completed_actions == 0
        assert result.total_actions == 0


def test_prompt_addition_exists():
    """Verify the multi-action prompt text is defined."""
    assert "actions" in MULTI_ACTION_PROMPT_ADDITION
    assert "multi-action" in MULTI_ACTION_PROMPT_ADDITION.lower()
