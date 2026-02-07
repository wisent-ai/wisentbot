#!/usr/bin/env python3
"""
ErrorRecovery - Rich error handling and recovery for agent actions.

Categorizes errors, captures context, suggests recovery strategies,
and tracks error patterns to help the agent learn from failures.
"""

import traceback
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ErrorCategory(Enum):
    """Categories of errors that can occur during skill execution."""
    CREDENTIALS = "credentials"
    INVALID_PARAMS = "invalid_params"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    NETWORK = "network"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Rich error context for a failed action."""
    skill_id: str
    action_name: str
    category: ErrorCategory
    message: str
    traceback_str: str
    params_used: Dict[str, Any]
    suggestions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "action_name": self.action_name,
            "category": self.category.value,
            "message": self.message,
            "traceback": self.traceback_str,
            "params_used": self.params_used,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp,
        }


# Maps exception types to categories
_EXCEPTION_CATEGORY_MAP: Dict[str, ErrorCategory] = {
    "AuthenticationError": ErrorCategory.CREDENTIALS,
    "PermissionError": ErrorCategory.PERMISSION,
    "FileNotFoundError": ErrorCategory.NOT_FOUND,
    "NotFoundError": ErrorCategory.NOT_FOUND,
    "TimeoutError": ErrorCategory.TIMEOUT,
    "asyncio.TimeoutError": ErrorCategory.TIMEOUT,
    "ConnectionError": ErrorCategory.NETWORK,
    "ConnectionRefusedError": ErrorCategory.NETWORK,
    "OSError": ErrorCategory.NETWORK,
    "TypeError": ErrorCategory.INVALID_PARAMS,
    "ValueError": ErrorCategory.INVALID_PARAMS,
    "KeyError": ErrorCategory.INVALID_PARAMS,
    "MemoryError": ErrorCategory.RESOURCE,
    "ResourceWarning": ErrorCategory.RESOURCE,
}

# Keywords in error messages that hint at categories
_MESSAGE_CATEGORY_HINTS: List[Tuple[str, ErrorCategory]] = [
    ("api key", ErrorCategory.CREDENTIALS),
    ("api_key", ErrorCategory.CREDENTIALS),
    ("token", ErrorCategory.CREDENTIALS),
    ("unauthorized", ErrorCategory.CREDENTIALS),
    ("authentication", ErrorCategory.CREDENTIALS),
    ("403", ErrorCategory.PERMISSION),
    ("forbidden", ErrorCategory.PERMISSION),
    ("permission denied", ErrorCategory.PERMISSION),
    ("not found", ErrorCategory.NOT_FOUND),
    ("404", ErrorCategory.NOT_FOUND),
    ("no such file", ErrorCategory.NOT_FOUND),
    ("timeout", ErrorCategory.TIMEOUT),
    ("timed out", ErrorCategory.TIMEOUT),
    ("connection", ErrorCategory.NETWORK),
    ("network", ErrorCategory.NETWORK),
    ("dns", ErrorCategory.NETWORK),
    ("rate limit", ErrorCategory.RATE_LIMIT),
    ("429", ErrorCategory.RATE_LIMIT),
    ("too many requests", ErrorCategory.RATE_LIMIT),
    ("throttl", ErrorCategory.RATE_LIMIT),
    ("invalid", ErrorCategory.INVALID_PARAMS),
    ("missing required", ErrorCategory.INVALID_PARAMS),
    ("expected", ErrorCategory.INVALID_PARAMS),
    ("memory", ErrorCategory.RESOURCE),
    ("disk", ErrorCategory.RESOURCE),
    ("quota", ErrorCategory.RESOURCE),
]

# Recovery suggestions by category
_RECOVERY_SUGGESTIONS: Dict[ErrorCategory, List[str]] = {
    ErrorCategory.CREDENTIALS: [
        "Check that the required API key or token is set in environment variables",
        "Try a different skill that doesn't require authentication",
        "Use skill_info to check which skills have valid credentials",
    ],
    ErrorCategory.INVALID_PARAMS: [
        "Review the parameter names and types expected by this action",
        "Check for typos in parameter names",
        "Ensure required parameters are provided",
        "Try with simplified or default parameters",
    ],
    ErrorCategory.RUNTIME: [
        "Try the action again with different parameters",
        "Check if the target resource is in the expected state",
        "Try a different approach to achieve the same goal",
    ],
    ErrorCategory.TIMEOUT: [
        "Retry the action - it may have been a temporary issue",
        "Try with a smaller input or simpler request",
        "Check if the external service is available",
    ],
    ErrorCategory.NETWORK: [
        "Check network connectivity",
        "Retry the action after a short delay",
        "Try an alternative endpoint or service",
    ],
    ErrorCategory.PERMISSION: [
        "Check file/directory permissions",
        "Try accessing a different resource",
        "Use an action that doesn't require elevated permissions",
    ],
    ErrorCategory.NOT_FOUND: [
        "Verify the file path or resource identifier",
        "List available files/resources first",
        "Check for typos in the path or name",
    ],
    ErrorCategory.RATE_LIMIT: [
        "Wait before retrying this action",
        "Reduce the frequency of API calls",
        "Try a different endpoint or service",
    ],
    ErrorCategory.RESOURCE: [
        "Free up system resources (memory, disk space)",
        "Try with smaller data or fewer items",
        "Check system resource usage",
    ],
    ErrorCategory.UNKNOWN: [
        "Try the action again",
        "Try a different approach",
        "Check the error message for specific guidance",
    ],
}


class ErrorRecoveryEngine:
    """
    Categorizes errors, provides recovery suggestions, and tracks
    error patterns to help the agent learn from failures.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._error_history: List[ErrorContext] = []
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._skill_error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def categorize_error(self, exception: Exception) -> ErrorCategory:
        """Determine the category of an error based on exception type and message."""
        exc_type = type(exception).__name__

        # Check exception type map first
        if exc_type in _EXCEPTION_CATEGORY_MAP:
            return _EXCEPTION_CATEGORY_MAP[exc_type]

        # Check message for hints
        msg_lower = str(exception).lower()
        for hint, category in _MESSAGE_CATEGORY_HINTS:
            if hint in msg_lower:
                return category

        return ErrorCategory.UNKNOWN

    def create_error_context(
        self,
        exception: Exception,
        skill_id: str,
        action_name: str,
        params: Dict[str, Any],
    ) -> ErrorContext:
        """Create a rich error context from an exception."""
        category = self.categorize_error(exception)
        tb = traceback.format_exception(type(exception), exception, exception.__traceback__)
        # Keep only last 5 frames to avoid huge tracebacks
        tb_str = "".join(tb[-5:]) if len(tb) > 5 else "".join(tb)

        # Sanitize params (remove sensitive values)
        safe_params = self._sanitize_params(params)

        suggestions = self.get_recovery_suggestions(category, skill_id, action_name)

        ctx = ErrorContext(
            skill_id=skill_id,
            action_name=action_name,
            category=category,
            message=str(exception),
            traceback_str=tb_str,
            params_used=safe_params,
            suggestions=suggestions,
        )

        self._record_error(ctx)
        return ctx

    def get_recovery_suggestions(
        self, category: ErrorCategory, skill_id: str, action_name: str
    ) -> List[str]:
        """Get recovery suggestions for an error category."""
        suggestions = list(_RECOVERY_SUGGESTIONS.get(category, _RECOVERY_SUGGESTIONS[ErrorCategory.UNKNOWN]))

        # Add context-specific suggestions based on error patterns
        key = f"{skill_id}:{action_name}"
        error_count = self._error_counts.get(key, 0)
        if error_count >= 3:
            suggestions.insert(0, f"This action has failed {error_count} times - consider using a different approach entirely")
        elif error_count >= 2:
            suggestions.insert(0, "This action has failed before - try with significantly different parameters")

        return suggestions

    def format_for_llm(self, error_context: ErrorContext) -> str:
        """Format error context as a concise string for the LLM prompt."""
        lines = [
            f"ERROR [{error_context.category.value}]: {error_context.message}",
        ]
        if error_context.suggestions:
            lines.append(f"Suggestions: {'; '.join(error_context.suggestions[:2])}")
        return " | ".join(lines)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of error patterns for agent self-awareness."""
        if not self._error_history:
            return {"total_errors": 0, "patterns": []}

        # Count by category
        category_counts: Dict[str, int] = defaultdict(int)
        for ctx in self._error_history:
            category_counts[ctx.category.value] += 1

        # Find most-failing actions
        action_failures = sorted(
            self._error_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Find recurring patterns
        patterns = []
        for action_key, count in action_failures:
            if count >= 2:
                patterns.append({
                    "action": action_key,
                    "failures": count,
                    "category": self._get_dominant_category(action_key),
                })

        return {
            "total_errors": len(self._error_history),
            "by_category": dict(category_counts),
            "patterns": patterns,
            "most_failing": action_failures[:3],
        }

    def get_action_failure_count(self, skill_id: str, action_name: str) -> int:
        """Get the number of times a specific action has failed."""
        return self._error_counts.get(f"{skill_id}:{action_name}", 0)

    def clear_history(self):
        """Clear error history."""
        self._error_history.clear()
        self._error_counts.clear()
        self._skill_error_counts.clear()

    def _record_error(self, ctx: ErrorContext):
        """Record an error in the history."""
        self._error_history.append(ctx)
        if len(self._error_history) > self.max_history:
            self._error_history = self._error_history[-self.max_history:]

        key = f"{ctx.skill_id}:{ctx.action_name}"
        self._error_counts[key] += 1
        self._skill_error_counts[ctx.skill_id][ctx.category.value] += 1

    def _get_dominant_category(self, action_key: str) -> str:
        """Find the most common error category for a given action."""
        category_counts: Dict[str, int] = defaultdict(int)
        for ctx in self._error_history:
            if f"{ctx.skill_id}:{ctx.action_name}" == action_key:
                category_counts[ctx.category.value] += 1
        if not category_counts:
            return "unknown"
        return max(category_counts, key=category_counts.get)

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive values from params for safe logging."""
        sensitive_keys = {"key", "token", "secret", "password", "api_key", "auth", "credential"}
        sanitized = {}
        for k, v in params.items():
            if any(s in k.lower() for s in sensitive_keys):
                sanitized[k] = "***REDACTED***"
            elif isinstance(v, str) and len(v) > 200:
                sanitized[k] = v[:200] + "...(truncated)"
            else:
                sanitized[k] = v
        return sanitized
