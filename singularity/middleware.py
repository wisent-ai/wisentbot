#!/usr/bin/env python3
"""
Action Middleware System

Provides a pipeline for intercepting, modifying, and monitoring agent actions
before and after execution. Middlewares can:
- Validate and modify action parameters before execution
- Block dangerous or costly actions
- Retry failed actions with backoff
- Log and track action performance
- Transform results after execution

Usage:
    agent = AutonomousAgent(...)
    agent.add_middleware(RetryMiddleware(max_retries=2))
    agent.add_middleware(CostGuardMiddleware(min_balance=1.0))
    agent.add_middleware(ActionLogMiddleware())
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ActionContext:
    """Context passed to middleware with action and agent state info."""
    tool: str
    params: Dict[str, Any]
    cycle: int = 0
    balance: float = 0.0
    agent_name: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActionMiddleware(ABC):
    """
    Base class for action middleware.

    Middleware intercepts the agent's action execution pipeline at three points:
    - before_action: Before execution (can modify params or block)
    - after_action: After successful execution (can modify result)
    - on_error: When execution fails (can provide recovery result)
    """

    @property
    def name(self) -> str:
        """Middleware name for logging."""
        return self.__class__.__name__

    async def before_action(
        self, context: ActionContext
    ) -> Optional[ActionContext]:
        """
        Called before action execution.

        Args:
            context: Action context with tool, params, and state

        Returns:
            Modified ActionContext to proceed, or None to block execution.
            When blocked, the pipeline returns a blocked result.
        """
        return context

    async def after_action(
        self, context: ActionContext, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Called after successful action execution.

        Args:
            context: Original action context
            result: Execution result dict

        Returns:
            Modified result dict
        """
        return result

    async def on_error(
        self, context: ActionContext, error: Exception
    ) -> Optional[Dict[str, Any]]:
        """
        Called when action execution raises an exception.

        Args:
            context: Original action context
            error: The exception that was raised

        Returns:
            Recovery result dict, or None to let the error propagate
        """
        return None


class MiddlewareChain:
    """
    Manages an ordered chain of middleware that wraps action execution.

    Middleware is executed in order for before_action, and in reverse
    order for after_action (like layers wrapping the execution).
    """

    def __init__(self):
        self._middlewares: List[ActionMiddleware] = []

    def add(self, middleware: ActionMiddleware) -> None:
        """Add middleware to the chain."""
        self._middlewares.append(middleware)

    def remove(self, name: str) -> bool:
        """Remove middleware by name. Returns True if found."""
        for i, mw in enumerate(self._middlewares):
            if mw.name == name:
                self._middlewares.pop(i)
                return True
        return False

    def list(self) -> List[str]:
        """List middleware names in order."""
        return [mw.name for mw in self._middlewares]

    def clear(self) -> None:
        """Remove all middleware."""
        self._middlewares.clear()

    async def run_before(self, context: ActionContext) -> Optional[ActionContext]:
        """
        Run all before_action middleware in order.
        Returns None if any middleware blocks the action.
        """
        for mw in self._middlewares:
            result = await mw.before_action(context)
            if result is None:
                return None
            context = result
        return context

    async def run_after(
        self, context: ActionContext, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run all after_action middleware in reverse order."""
        for mw in reversed(self._middlewares):
            result = await mw.after_action(context, result)
        return result

    async def run_on_error(
        self, context: ActionContext, error: Exception
    ) -> Optional[Dict[str, Any]]:
        """
        Run on_error middleware until one provides a recovery result.
        Returns None if no middleware handles the error.
        """
        for mw in self._middlewares:
            recovery = await mw.on_error(context, error)
            if recovery is not None:
                return recovery
        return None

    @property
    def is_empty(self) -> bool:
        return len(self._middlewares) == 0


# === Built-in Middleware Implementations ===


class RetryMiddleware(ActionMiddleware):
    """
    Retries failed actions with exponential backoff.

    Wraps action execution to catch failures and retry up to max_retries
    times with configurable delay between attempts.
    """

    def __init__(
        self,
        max_retries: int = 2,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_errors: Optional[List[str]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.retryable_errors = retryable_errors or [
            "timeout", "rate_limit", "connection", "temporary",
            "503", "429", "502", "500",
        ]
        self._retry_counts: Dict[str, int] = {}

    async def on_error(
        self, context: ActionContext, error: Exception
    ) -> Optional[Dict[str, Any]]:
        """Mark error as retryable if it matches patterns."""
        error_str = str(error).lower()
        is_retryable = any(pat in error_str for pat in self.retryable_errors)

        if is_retryable:
            key = f"{context.tool}:{context.cycle}"
            count = self._retry_counts.get(key, 0)
            if count < self.max_retries:
                self._retry_counts[key] = count + 1
                delay = self.base_delay * (self.backoff_factor ** count)
                context.metadata["_retry"] = {
                    "attempt": count + 1,
                    "max": self.max_retries,
                    "delay": delay,
                }
                return {
                    "status": "retry",
                    "message": f"Retryable error (attempt {count + 1}/{self.max_retries}): {error}",
                    "retry_delay": delay,
                }
        return None

    async def after_action(
        self, context: ActionContext, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clear retry count on success."""
        key = f"{context.tool}:{context.cycle}"
        self._retry_counts.pop(key, None)
        return result


class CostGuardMiddleware(ActionMiddleware):
    """
    Blocks actions when agent balance is too low.

    Prevents the agent from executing expensive actions when its
    remaining balance is below a safety threshold.
    """

    def __init__(
        self,
        min_balance: float = 0.50,
        blocked_when_low: Optional[List[str]] = None,
        always_allowed: Optional[List[str]] = None,
    ):
        self.min_balance = min_balance
        self.blocked_when_low = blocked_when_low or []
        self.always_allowed = always_allowed or [
            "wait", "filesystem:view", "filesystem:ls",
            "filesystem:glob", "filesystem:grep",
        ]
        self._blocks: List[Dict] = []

    async def before_action(
        self, context: ActionContext
    ) -> Optional[ActionContext]:
        """Block costly actions when balance is low."""
        if context.tool in self.always_allowed:
            return context

        if context.balance < self.min_balance:
            self._blocks.append({
                "tool": context.tool,
                "balance": context.balance,
                "timestamp": datetime.now().isoformat(),
            })
            return None

        if self.blocked_when_low and context.tool in self.blocked_when_low:
            if context.balance < self.min_balance * 2:
                self._blocks.append({
                    "tool": context.tool,
                    "balance": context.balance,
                    "reason": "blocked_when_low",
                    "timestamp": datetime.now().isoformat(),
                })
                return None

        return context

    @property
    def block_history(self) -> List[Dict]:
        """Get history of blocked actions."""
        return list(self._blocks)


class ActionLogMiddleware(ActionMiddleware):
    """
    Logs all actions with timing and results for analysis.

    Tracks execution time, success rate, and error patterns to
    enable the agent to learn from its action history.
    """

    def __init__(self, max_entries: int = 200):
        self.max_entries = max_entries
        self._log: List[Dict] = []
        self._start_times: Dict[str, float] = {}

    async def before_action(
        self, context: ActionContext
    ) -> Optional[ActionContext]:
        """Record action start time."""
        key = f"{context.tool}:{context.cycle}"
        self._start_times[key] = time.time()
        return context

    async def after_action(
        self, context: ActionContext, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log action completion with timing."""
        key = f"{context.tool}:{context.cycle}"
        start = self._start_times.pop(key, time.time())
        duration = time.time() - start

        entry = {
            "tool": context.tool,
            "cycle": context.cycle,
            "status": result.get("status", "unknown"),
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
        }
        self._log.append(entry)

        # Trim to max size
        if len(self._log) > self.max_entries:
            self._log = self._log[-self.max_entries:]

        return result

    async def on_error(
        self, context: ActionContext, error: Exception
    ) -> Optional[Dict[str, Any]]:
        """Log errors."""
        key = f"{context.tool}:{context.cycle}"
        start = self._start_times.pop(key, time.time())
        duration = time.time() - start

        self._log.append({
            "tool": context.tool,
            "cycle": context.cycle,
            "status": "error",
            "error": str(error)[:200],
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
        })
        return None  # Don't handle the error, just log it

    @property
    def action_log(self) -> List[Dict]:
        """Get action log."""
        return list(self._log)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from action log."""
        if not self._log:
            return {"total": 0}

        total = len(self._log)
        successes = sum(1 for e in self._log if e["status"] == "success")
        errors = sum(1 for e in self._log if e["status"] == "error")
        durations = [e["duration_seconds"] for e in self._log]

        # Per-tool stats
        tool_counts: Dict[str, int] = {}
        tool_errors: Dict[str, int] = {}
        for entry in self._log:
            tool = entry["tool"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            if entry["status"] == "error":
                tool_errors[tool] = tool_errors.get(tool, 0) + 1

        return {
            "total": total,
            "successes": successes,
            "errors": errors,
            "success_rate": round(successes / total, 3) if total else 0,
            "avg_duration": round(sum(durations) / len(durations), 3) if durations else 0,
            "max_duration": round(max(durations), 3) if durations else 0,
            "tool_usage": tool_counts,
            "tool_errors": tool_errors,
        }


class RateLimitMiddleware(ActionMiddleware):
    """
    Rate-limits actions to prevent excessive API calls.

    Enforces a maximum number of actions per time window,
    both globally and per-tool.
    """

    def __init__(
        self,
        max_per_minute: int = 30,
        per_tool_per_minute: int = 10,
    ):
        self.max_per_minute = max_per_minute
        self.per_tool_per_minute = per_tool_per_minute
        self._timestamps: List[float] = []
        self._tool_timestamps: Dict[str, List[float]] = {}

    def _clean_old(self, timestamps: List[float], window: float = 60.0) -> List[float]:
        """Remove timestamps older than window seconds."""
        cutoff = time.time() - window
        return [t for t in timestamps if t > cutoff]

    async def before_action(
        self, context: ActionContext
    ) -> Optional[ActionContext]:
        """Check rate limits before allowing action."""
        now = time.time()

        # Global rate limit
        self._timestamps = self._clean_old(self._timestamps)
        if len(self._timestamps) >= self.max_per_minute:
            context.metadata["_rate_limited"] = True
            return None

        # Per-tool rate limit
        tool = context.tool
        if tool not in self._tool_timestamps:
            self._tool_timestamps[tool] = []
        self._tool_timestamps[tool] = self._clean_old(self._tool_timestamps[tool])
        if len(self._tool_timestamps[tool]) >= self.per_tool_per_minute:
            context.metadata["_rate_limited"] = True
            context.metadata["_rate_limited_tool"] = tool
            return None

        # Record this action
        self._timestamps.append(now)
        self._tool_timestamps[tool].append(now)
        return context
