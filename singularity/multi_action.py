#!/usr/bin/env python3
"""
Multi-Action Support for Singularity Agent

Allows the LLM to return multiple actions in a single response,
reducing API calls and enabling more efficient agent execution.

The LLM can respond with either:
1. A single action: {"tool": "...", "params": {}, "reasoning": "..."}
2. Multiple actions: {"actions": [{"tool": "...", "params": {}}, ...], "reasoning": "..."}

Actions execute sequentially. Previous results are available to inform
whether to continue or stop execution.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable


@dataclass
class ActionItem:
    """A single action to execute."""
    tool: str
    params: Dict = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class ActionResult:
    """Result from executing a single action."""
    tool: str
    params: Dict
    result: Dict
    success: bool
    index: int  # Position in the action sequence


@dataclass
class MultiActionResult:
    """Combined result from executing multiple actions."""
    results: List[ActionResult] = field(default_factory=list)
    total_actions: int = 0
    completed_actions: int = 0
    stopped_early: bool = False
    stop_reason: str = ""

    @property
    def all_succeeded(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def last_result(self) -> Optional[ActionResult]:
        return self.results[-1] if self.results else None

    def summary(self) -> str:
        """Human-readable summary of execution."""
        parts = [f"Executed {self.completed_actions}/{self.total_actions} actions"]
        for r in self.results:
            status = "✓" if r.success else "✗"
            parts.append(f"  {status} [{r.index}] {r.tool}: {r.result.get('status', 'unknown')}")
        if self.stopped_early:
            parts.append(f"  ⚠ Stopped: {self.stop_reason}")
        return "\n".join(parts)

    def to_dict(self) -> Dict:
        """Convert to dict for agent state tracking."""
        return {
            "total_actions": self.total_actions,
            "completed_actions": self.completed_actions,
            "all_succeeded": self.all_succeeded,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "results": [
                {
                    "tool": r.tool,
                    "status": r.result.get("status", "unknown"),
                    "success": r.success,
                    "index": r.index,
                }
                for r in self.results
            ],
        }


def parse_multi_action(response_text: str) -> List[ActionItem]:
    """
    Parse LLM response that may contain one or multiple actions.

    Supported formats:
    1. Single action: {"tool": "skill:action", "params": {}, "reasoning": "why"}
    2. Multi-action:  {"actions": [...], "reasoning": "overall plan"}
    3. JSON array:    [{"tool": "...", "params": {}}, ...]

    Returns a list of ActionItem (always a list, even for single action).
    """
    response_text = response_text.strip()

    # Try to find JSON in the response
    json_obj = _extract_json(response_text)

    if json_obj is None:
        # Fallback - try to find tool name
        tool_match = re.search(r'(\w+:\w+)', response_text)
        if tool_match:
            return [ActionItem(
                tool=tool_match.group(1),
                params={},
                reasoning=response_text[:200]
            )]
        return [ActionItem(tool="wait", params={}, reasoning="Could not parse response")]

    # Case 1: JSON array of actions
    if isinstance(json_obj, list):
        return _parse_action_list(json_obj, "")

    # Case 2: Object with "actions" array
    if isinstance(json_obj, dict) and "actions" in json_obj:
        actions_data = json_obj["actions"]
        overall_reasoning = json_obj.get("reasoning", "")
        if isinstance(actions_data, list):
            return _parse_action_list(actions_data, overall_reasoning)

    # Case 3: Single action object with "tool"
    if isinstance(json_obj, dict) and "tool" in json_obj:
        return [ActionItem(
            tool=json_obj.get("tool", "wait"),
            params=json_obj.get("params", {}),
            reasoning=json_obj.get("reasoning", "")
        )]

    return [ActionItem(tool="wait", params={}, reasoning="Could not parse response")]


def _extract_json(text: str) -> Any:
    """Extract JSON object or array from text, handling nested braces/brackets."""
    # Try to find a JSON array first
    array_start = text.find('[')
    # Try to find a JSON object
    obj_start = text.find('{')

    # Determine which comes first
    starts = []
    if array_start >= 0:
        starts.append(('array', array_start, '[', ']'))
    if obj_start >= 0:
        starts.append(('object', obj_start, '{', '}'))

    if not starts:
        return None

    # Try the first JSON structure found
    starts.sort(key=lambda x: x[1])

    for kind, start, open_char, close_char in starts:
        end = _find_matching_close(text, start, open_char, close_char)
        if end is not None:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    return None


def _find_matching_close(text: str, start: int, open_char: str, close_char: str) -> Optional[int]:
    """Find the matching closing bracket/brace, handling nesting."""
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]

        if escape_next:
            escape_next = False
            continue

        if c == '\\' and in_string:
            escape_next = True
            continue

        if c == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return i

    return None


def _parse_action_list(actions_data: list, overall_reasoning: str) -> List[ActionItem]:
    """Parse a list of action dicts into ActionItems."""
    actions = []
    for i, item in enumerate(actions_data):
        if not isinstance(item, dict):
            continue
        tool = item.get("tool", "wait")
        params = item.get("params", {})
        reasoning = item.get("reasoning", overall_reasoning)
        if not reasoning and overall_reasoning:
            reasoning = f"Step {i + 1}: {overall_reasoning}"
        actions.append(ActionItem(tool=tool, params=params, reasoning=reasoning))

    if not actions:
        return [ActionItem(tool="wait", params={}, reasoning="Empty action list")]

    return actions


class MultiActionExecutor:
    """
    Executes a sequence of actions, handling errors and providing results.

    Supports two modes:
    - stop_on_error=True: Stop executing after first failed action
    - stop_on_error=False: Continue executing even if some actions fail

    Also supports a max_actions limit to prevent unbounded execution.
    """

    def __init__(
        self,
        execute_fn: Callable[[str, Dict], Awaitable[Dict]],
        stop_on_error: bool = True,
        max_actions: int = 5,
    ):
        """
        Args:
            execute_fn: Async function that executes a single action.
                       Signature: (tool: str, params: dict) -> dict
            stop_on_error: Whether to stop on first error
            max_actions: Maximum number of actions to execute per batch
        """
        self.execute_fn = execute_fn
        self.stop_on_error = stop_on_error
        self.max_actions = max_actions

    async def execute(self, actions: List[ActionItem]) -> MultiActionResult:
        """
        Execute a list of actions sequentially.

        Args:
            actions: List of ActionItem to execute

        Returns:
            MultiActionResult with all results
        """
        # Enforce max_actions limit
        effective_actions = actions[:self.max_actions]
        truncated = len(actions) > self.max_actions

        result = MultiActionResult(
            total_actions=len(effective_actions),
        )

        for i, action in enumerate(effective_actions):
            try:
                action_result = await self.execute_fn(action.tool, action.params)
                success = action_result.get("status") in ("success", "waited")

                result.results.append(ActionResult(
                    tool=action.tool,
                    params=action.params,
                    result=action_result,
                    success=success,
                    index=i,
                ))
                result.completed_actions += 1

                if not success and self.stop_on_error:
                    result.stopped_early = True
                    result.stop_reason = f"Action {i} ({action.tool}) failed: {action_result.get('message', 'unknown error')}"
                    break

            except Exception as e:
                result.results.append(ActionResult(
                    tool=action.tool,
                    params=action.params,
                    result={"status": "error", "message": str(e)},
                    success=False,
                    index=i,
                ))
                result.completed_actions += 1

                if self.stop_on_error:
                    result.stopped_early = True
                    result.stop_reason = f"Action {i} ({action.tool}) raised exception: {str(e)}"
                    break

        if truncated and not result.stopped_early:
            result.stopped_early = True
            result.stop_reason = f"Truncated: {len(actions)} actions requested, max {self.max_actions} allowed"

        return result


# Convenience: format multi-action instructions for the LLM prompt
MULTI_ACTION_PROMPT_ADDITION = """
MULTI-ACTION MODE:
You can execute multiple actions in sequence by responding with:
{"actions": [
  {"tool": "skill:action", "params": {}},
  {"tool": "skill:action", "params": {}}
], "reasoning": "why these actions in this order"}

Actions execute in order. If one fails, the rest are skipped.
Use multi-action when steps are independent or form a clear sequence.
Maximum 5 actions per response.

For a single action, continue using:
{"tool": "skill:action", "params": {}, "reasoning": "why"}
"""
