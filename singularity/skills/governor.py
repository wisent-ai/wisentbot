#!/usr/bin/env python3
"""
GovernorSkill - Agent safety, budget enforcement, and behavioral guardrails.

Monitors and controls agent behavior to prevent runaway spending, infinite loops,
and other failure modes. Provides circuit breakers for failing skills, rate limiting,
budget enforcement, and loop detection.

Zero external dependencies. File-backed persistence.

Pillar: Self-Improvement (safe experimentation), Replication (controlling replicas)
"""

import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult


@dataclass
class BudgetLimit:
    """Budget spending limit."""
    max_amount: float
    period_seconds: float  # Time window
    current_spent: float = 0.0
    window_start: float = 0.0  # Unix timestamp

    def to_dict(self) -> Dict:
        return {
            "max_amount": self.max_amount,
            "period_seconds": self.period_seconds,
            "current_spent": self.current_spent,
            "window_start": self.window_start,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BudgetLimit":
        return cls(**d)


@dataclass
class CircuitBreaker:
    """Circuit breaker for a skill/action."""
    failure_count: int = 0
    failure_threshold: int = 5
    state: str = "closed"  # closed=normal, open=blocked, half_open=testing
    last_failure: float = 0.0
    cooldown_seconds: float = 300.0  # 5 min default
    opened_at: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "state": self.state,
            "last_failure": self.last_failure,
            "cooldown_seconds": self.cooldown_seconds,
            "opened_at": self.opened_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CircuitBreaker":
        return cls(**d)


@dataclass
class RateLimit:
    """Rate limit for actions."""
    max_calls: int
    period_seconds: float
    call_timestamps: List[float] = None

    def __post_init__(self):
        if self.call_timestamps is None:
            self.call_timestamps = []

    def to_dict(self) -> Dict:
        return {
            "max_calls": self.max_calls,
            "period_seconds": self.period_seconds,
            "call_timestamps": self.call_timestamps[-100:],  # Keep last 100
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RateLimit":
        return cls(
            max_calls=d["max_calls"],
            period_seconds=d["period_seconds"],
            call_timestamps=d.get("call_timestamps", []),
        )


class GovernorSkill(Skill):
    """
    Agent safety governor providing budget enforcement, circuit breakers,
    rate limiting, loop detection, and behavioral guardrails.
    """

    DATA_DIR = Path(__file__).parent.parent / "data" / "governor"

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._budgets: Dict[str, BudgetLimit] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._rate_limits: Dict[str, RateLimit] = {}
        self._guardrails: Dict[str, Dict] = {}
        self._action_history: List[Dict] = []
        self._violations: List[Dict] = []
        self._load_state()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="governor",
            name="GovernorSkill",
            version="1.0.0",
            category="safety",
            description="Agent safety governor: budget enforcement, circuit breakers, rate limiting, loop detection, and behavioral guardrails",
            actions=[
                SkillAction(
                    name="check",
                    description="Check if a proposed action is allowed by all safety rules. Returns allow/deny with reasons.",
                    parameters={
                        "action": {"type": "string", "required": True, "description": "Action identifier (e.g., 'shell:bash')"},
                        "estimated_cost": {"type": "number", "required": False, "description": "Estimated cost of the action in USD"},
                    },
                ),
                SkillAction(
                    name="record_action",
                    description="Record an executed action for tracking and analysis",
                    parameters={
                        "action": {"type": "string", "required": True, "description": "Action identifier"},
                        "success": {"type": "boolean", "required": True, "description": "Whether the action succeeded"},
                        "cost": {"type": "number", "required": False, "description": "Actual cost in USD"},
                        "duration_ms": {"type": "number", "required": False, "description": "Execution duration in milliseconds"},
                    },
                ),
                SkillAction(
                    name="set_budget",
                    description="Set a spending limit for a scope (global, per-skill, or per-action)",
                    parameters={
                        "scope": {"type": "string", "required": True, "description": "Budget scope: 'global', skill_id, or 'skill:action'"},
                        "max_amount": {"type": "number", "required": True, "description": "Maximum spend in USD"},
                        "period": {"type": "string", "required": False, "description": "Time period: '1h', '24h', '7d' (default: '24h')"},
                    },
                ),
                SkillAction(
                    name="set_rate_limit",
                    description="Set rate limit for an action",
                    parameters={
                        "action": {"type": "string", "required": True, "description": "Action identifier or 'global'"},
                        "max_calls": {"type": "integer", "required": True, "description": "Maximum calls allowed"},
                        "period": {"type": "string", "required": False, "description": "Time period: '1m', '1h', '24h' (default: '1h')"},
                    },
                ),
                SkillAction(
                    name="circuit_status",
                    description="Get or set circuit breaker status for a skill/action",
                    parameters={
                        "action": {"type": "string", "required": True, "description": "Action or skill identifier"},
                        "command": {"type": "string", "required": False, "description": "'status', 'reset', 'open', 'configure' (default: 'status')"},
                        "threshold": {"type": "integer", "required": False, "description": "Failure threshold (for 'configure')"},
                        "cooldown": {"type": "number", "required": False, "description": "Cooldown seconds (for 'configure')"},
                    },
                ),
                SkillAction(
                    name="detect_loops",
                    description="Analyze recent actions for repetitive patterns and stuck behavior",
                    parameters={
                        "window": {"type": "integer", "required": False, "description": "Number of recent actions to analyze (default: 20)"},
                        "threshold": {"type": "number", "required": False, "description": "Repetition ratio to flag as loop (0-1, default: 0.5)"},
                    },
                ),
                SkillAction(
                    name="set_guardrail",
                    description="Define a behavioral guardrail rule",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Guardrail name"},
                        "rule_type": {"type": "string", "required": True, "description": "'block_action', 'require_approval', 'max_cost', 'time_window'"},
                        "config": {"type": "object", "required": True, "description": "Rule configuration (varies by type)"},
                        "enabled": {"type": "boolean", "required": False, "description": "Whether guardrail is enabled (default: true)"},
                    },
                ),
                SkillAction(
                    name="report",
                    description="Generate a comprehensive safety and health report",
                    parameters={},
                ),
                SkillAction(
                    name="violations",
                    description="Get recent safety violations",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Max violations to return (default: 20)"},
                    },
                ),
                SkillAction(
                    name="reset",
                    description="Reset governor state (budgets, circuit breakers, rate limits, history)",
                    parameters={
                        "scope": {"type": "string", "required": False, "description": "'all', 'budgets', 'circuits', 'rates', 'history', 'violations' (default: 'all')"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "check": self._check,
            "record_action": self._record_action,
            "set_budget": self._set_budget,
            "set_rate_limit": self._set_rate_limit,
            "circuit_status": self._circuit_status,
            "detect_loops": self._detect_loops,
            "set_guardrail": self._set_guardrail,
            "report": self._report,
            "violations": self._get_violations,
            "reset": self._reset,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # --- Core Actions ---

    async def _check(self, params: Dict) -> SkillResult:
        """Check if a proposed action is allowed by all safety rules."""
        action_id = params.get("action", "")
        estimated_cost = params.get("estimated_cost", 0.0)
        now = time.time()

        reasons = []
        allowed = True

        # 1. Check circuit breaker
        cb = self._get_circuit_breaker(action_id)
        if cb and cb.state == "open":
            if now - cb.opened_at < cb.cooldown_seconds:
                allowed = False
                remaining = cb.cooldown_seconds - (now - cb.opened_at)
                reasons.append(f"Circuit breaker OPEN for '{action_id}' ({cb.failure_count} failures, {remaining:.0f}s cooldown remaining)")
            else:
                cb.state = "half_open"
                reasons.append(f"Circuit breaker half-open for '{action_id}' - allowing test call")

        # 2. Check budget limits
        for scope, budget in self._budgets.items():
            if scope == "global" or scope == action_id or action_id.startswith(scope + ":"):
                # Reset window if expired
                if now - budget.window_start > budget.period_seconds:
                    budget.current_spent = 0.0
                    budget.window_start = now

                if budget.current_spent + estimated_cost > budget.max_amount:
                    allowed = False
                    remaining_budget = budget.max_amount - budget.current_spent
                    reasons.append(
                        f"Budget limit exceeded for '{scope}': "
                        f"${budget.current_spent:.4f} + ${estimated_cost:.4f} > ${budget.max_amount:.4f} "
                        f"(${remaining_budget:.4f} remaining in window)"
                    )

        # 3. Check rate limits
        rl = self._rate_limits.get(action_id) or self._rate_limits.get("global")
        if rl:
            # Clean old timestamps
            cutoff = now - rl.period_seconds
            rl.call_timestamps = [t for t in rl.call_timestamps if t > cutoff]
            if len(rl.call_timestamps) >= rl.max_calls:
                allowed = False
                reasons.append(
                    f"Rate limit exceeded for '{action_id}': "
                    f"{len(rl.call_timestamps)}/{rl.max_calls} calls in {rl.period_seconds}s"
                )

        # 4. Check guardrails
        for name, guardrail in self._guardrails.items():
            if not guardrail.get("enabled", True):
                continue

            rule_type = guardrail.get("rule_type", "")
            config = guardrail.get("config", {})

            if rule_type == "block_action":
                blocked = config.get("actions", [])
                if action_id in blocked or any(action_id.startswith(b) for b in blocked):
                    allowed = False
                    reasons.append(f"Guardrail '{name}': action '{action_id}' is blocked")

            elif rule_type == "max_cost":
                max_cost = config.get("max_cost", float("inf"))
                if estimated_cost > max_cost:
                    allowed = False
                    reasons.append(
                        f"Guardrail '{name}': cost ${estimated_cost:.4f} exceeds max ${max_cost:.4f}"
                    )

            elif rule_type == "time_window":
                allowed_hours = config.get("allowed_hours", [])
                if allowed_hours:
                    current_hour = datetime.now().hour
                    if current_hour not in allowed_hours:
                        allowed = False
                        reasons.append(
                            f"Guardrail '{name}': current hour {current_hour} not in allowed hours"
                        )

        if not allowed:
            violation = {
                "timestamp": datetime.now().isoformat(),
                "action": action_id,
                "estimated_cost": estimated_cost,
                "reasons": reasons,
            }
            self._violations.append(violation)
            self._violations = self._violations[-500:]

        self._save_state()

        return SkillResult(
            success=True,
            message="Action allowed" if allowed else "Action DENIED",
            data={
                "allowed": allowed,
                "action": action_id,
                "reasons": reasons,
                "checks_passed": {
                    "circuit_breaker": not any("Circuit breaker OPEN" in r for r in reasons),
                    "budget": not any("Budget limit" in r for r in reasons),
                    "rate_limit": not any("Rate limit" in r for r in reasons),
                    "guardrails": not any("Guardrail" in r for r in reasons),
                },
            },
        )

    async def _record_action(self, params: Dict) -> SkillResult:
        """Record an executed action for tracking."""
        action_id = params.get("action", "")
        success = params.get("success", True)
        cost = params.get("cost", 0.0)
        duration_ms = params.get("duration_ms", 0)
        now = time.time()

        record = {
            "action": action_id,
            "success": success,
            "cost": cost,
            "duration_ms": duration_ms,
            "timestamp": now,
            "iso_time": datetime.now().isoformat(),
        }
        self._action_history.append(record)
        self._action_history = self._action_history[-1000:]

        # Update budget spending
        for scope, budget in self._budgets.items():
            if scope == "global" or scope == action_id or action_id.startswith(scope + ":"):
                if now - budget.window_start > budget.period_seconds:
                    budget.current_spent = 0.0
                    budget.window_start = now
                budget.current_spent += cost

        # Update rate limits
        for key in [action_id, "global"]:
            if key in self._rate_limits:
                self._rate_limits[key].call_timestamps.append(now)

        # Update circuit breaker
        cb = self._get_or_create_circuit_breaker(action_id)
        if success:
            if cb.state == "half_open":
                cb.state = "closed"
                cb.failure_count = 0
            elif cb.state == "closed":
                # Decay failures on success
                cb.failure_count = max(0, cb.failure_count - 1)
        else:
            cb.failure_count += 1
            cb.last_failure = now
            if cb.failure_count >= cb.failure_threshold:
                cb.state = "open"
                cb.opened_at = now

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Recorded {'success' if success else 'failure'} for {action_id}",
            data={
                "action": action_id,
                "circuit_breaker_state": cb.state,
                "circuit_breaker_failures": cb.failure_count,
                "total_actions_recorded": len(self._action_history),
            },
        )

    async def _set_budget(self, params: Dict) -> SkillResult:
        """Set a spending limit."""
        scope = params.get("scope", "global")
        max_amount = params.get("max_amount", 1.0)
        period = params.get("period", "24h")

        period_seconds = self._parse_duration(period)
        if period_seconds <= 0:
            return SkillResult(success=False, message=f"Invalid period: {period}")

        self._budgets[scope] = BudgetLimit(
            max_amount=max_amount,
            period_seconds=period_seconds,
            window_start=time.time(),
        )
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Budget set: ${max_amount:.4f} per {period} for scope '{scope}'",
            data={
                "scope": scope,
                "max_amount": max_amount,
                "period": period,
                "period_seconds": period_seconds,
            },
        )

    async def _set_rate_limit(self, params: Dict) -> SkillResult:
        """Set rate limit for an action."""
        action_id = params.get("action", "global")
        max_calls = params.get("max_calls", 10)
        period = params.get("period", "1h")

        period_seconds = self._parse_duration(period)
        if period_seconds <= 0:
            return SkillResult(success=False, message=f"Invalid period: {period}")

        self._rate_limits[action_id] = RateLimit(
            max_calls=max_calls,
            period_seconds=period_seconds,
        )
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Rate limit set: {max_calls} calls per {period} for '{action_id}'",
            data={
                "action": action_id,
                "max_calls": max_calls,
                "period": period,
                "period_seconds": period_seconds,
            },
        )

    async def _circuit_status(self, params: Dict) -> SkillResult:
        """Get or manage circuit breaker status."""
        action_id = params.get("action", "")
        command = params.get("command", "status")

        cb = self._get_or_create_circuit_breaker(action_id)

        if command == "reset":
            cb.state = "closed"
            cb.failure_count = 0
            cb.opened_at = 0.0
            self._save_state()
            return SkillResult(
                success=True,
                message=f"Circuit breaker reset for '{action_id}'",
                data={"action": action_id, "state": "closed"},
            )

        elif command == "open":
            cb.state = "open"
            cb.opened_at = time.time()
            self._save_state()
            return SkillResult(
                success=True,
                message=f"Circuit breaker opened for '{action_id}'",
                data={"action": action_id, "state": "open"},
            )

        elif command == "configure":
            threshold = params.get("threshold", cb.failure_threshold)
            cooldown = params.get("cooldown", cb.cooldown_seconds)
            cb.failure_threshold = threshold
            cb.cooldown_seconds = cooldown
            self._save_state()
            return SkillResult(
                success=True,
                message=f"Circuit breaker configured: threshold={threshold}, cooldown={cooldown}s",
                data={
                    "action": action_id,
                    "threshold": threshold,
                    "cooldown": cooldown,
                },
            )

        # Default: status
        now = time.time()
        remaining_cooldown = 0
        if cb.state == "open":
            remaining_cooldown = max(0, cb.cooldown_seconds - (now - cb.opened_at))

        return SkillResult(
            success=True,
            message=f"Circuit breaker for '{action_id}': {cb.state}",
            data={
                "action": action_id,
                "state": cb.state,
                "failure_count": cb.failure_count,
                "failure_threshold": cb.failure_threshold,
                "cooldown_seconds": cb.cooldown_seconds,
                "remaining_cooldown": remaining_cooldown,
                "last_failure": datetime.fromtimestamp(cb.last_failure).isoformat() if cb.last_failure > 0 else None,
            },
        )

    async def _detect_loops(self, params: Dict) -> SkillResult:
        """Detect repetitive patterns in recent actions."""
        window = params.get("window", 20)
        threshold = params.get("threshold", 0.5)

        recent = self._action_history[-window:]
        if len(recent) < 3:
            return SkillResult(
                success=True,
                message="Not enough history for loop detection",
                data={"loop_detected": False, "history_size": len(recent)},
            )

        # Count action frequencies
        action_counts = Counter(r["action"] for r in recent)
        total = len(recent)

        # Check for dominant action (single action > threshold)
        loops = []
        for action, count in action_counts.most_common():
            ratio = count / total
            if ratio >= threshold:
                loops.append({
                    "type": "dominant_action",
                    "action": action,
                    "count": count,
                    "ratio": round(ratio, 3),
                    "message": f"'{action}' executed {count}/{total} times ({ratio:.0%})",
                })

        # Check for alternating patterns (A-B-A-B)
        actions = [r["action"] for r in recent]
        if len(actions) >= 4:
            for pattern_len in [2, 3]:
                pattern = tuple(actions[-pattern_len:])
                matches = 0
                for i in range(len(actions) - pattern_len + 1):
                    if tuple(actions[i:i + pattern_len]) == pattern:
                        matches += 1
                if matches >= 3:
                    loops.append({
                        "type": "repeating_pattern",
                        "pattern": list(pattern),
                        "matches": matches,
                        "message": f"Pattern {list(pattern)} repeated {matches} times",
                    })

        # Check for consecutive failures
        recent_failures = [r for r in recent if not r.get("success", True)]
        if len(recent_failures) >= 3:
            failure_actions = Counter(r["action"] for r in recent_failures)
            for action, count in failure_actions.most_common():
                if count >= 3:
                    loops.append({
                        "type": "repeated_failures",
                        "action": action,
                        "failure_count": count,
                        "message": f"'{action}' failed {count} times in last {window} actions",
                    })

        loop_detected = len(loops) > 0
        severity = "none"
        if loop_detected:
            max_ratio = max((l.get("ratio", 0) for l in loops), default=0)
            if max_ratio >= 0.8:
                severity = "critical"
            elif max_ratio >= 0.6:
                severity = "high"
            else:
                severity = "medium"

        return SkillResult(
            success=True,
            message=f"Loop {'DETECTED' if loop_detected else 'not detected'} (severity: {severity})",
            data={
                "loop_detected": loop_detected,
                "severity": severity,
                "patterns": loops,
                "action_distribution": dict(action_counts.most_common(10)),
                "window_size": len(recent),
            },
        )

    async def _set_guardrail(self, params: Dict) -> SkillResult:
        """Define a behavioral guardrail."""
        name = params.get("name", "")
        rule_type = params.get("rule_type", "")
        config = params.get("config", {})
        enabled = params.get("enabled", True)

        valid_types = ["block_action", "require_approval", "max_cost", "time_window"]
        if rule_type not in valid_types:
            return SkillResult(
                success=False,
                message=f"Invalid rule_type: {rule_type}. Must be one of: {valid_types}",
            )

        self._guardrails[name] = {
            "name": name,
            "rule_type": rule_type,
            "config": config,
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
        }
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Guardrail '{name}' set (type: {rule_type}, enabled: {enabled})",
            data={"guardrail": self._guardrails[name]},
        )

    async def _report(self, params: Dict) -> SkillResult:
        """Generate comprehensive safety report."""
        now = time.time()

        # Budget summary
        budget_summary = {}
        for scope, budget in self._budgets.items():
            if now - budget.window_start > budget.period_seconds:
                utilization = 0.0
            else:
                utilization = (budget.current_spent / budget.max_amount * 100) if budget.max_amount > 0 else 0
            budget_summary[scope] = {
                "max": budget.max_amount,
                "spent": round(budget.current_spent, 6),
                "utilization_pct": round(utilization, 1),
                "period_seconds": budget.period_seconds,
            }

        # Circuit breaker summary
        circuit_summary = {}
        for action_id, cb in self._circuit_breakers.items():
            circuit_summary[action_id] = {
                "state": cb.state,
                "failures": cb.failure_count,
                "threshold": cb.failure_threshold,
            }

        # Rate limit summary
        rate_summary = {}
        for action_id, rl in self._rate_limits.items():
            cutoff = now - rl.period_seconds
            active_calls = len([t for t in rl.call_timestamps if t > cutoff])
            rate_summary[action_id] = {
                "current": active_calls,
                "max": rl.max_calls,
                "utilization_pct": round(active_calls / rl.max_calls * 100, 1) if rl.max_calls > 0 else 0,
            }

        # Action statistics
        total_actions = len(self._action_history)
        successes = sum(1 for r in self._action_history if r.get("success", True))
        failures = total_actions - successes
        total_cost = sum(r.get("cost", 0) for r in self._action_history)

        # Recent activity (last hour)
        one_hour_ago = now - 3600
        recent = [r for r in self._action_history if r.get("timestamp", 0) > one_hour_ago]
        recent_cost = sum(r.get("cost", 0) for r in recent)

        # Open circuit breakers
        open_circuits = [aid for aid, cb in self._circuit_breakers.items() if cb.state == "open"]

        # Overall health score (0-100)
        health = 100
        if failures > 0 and total_actions > 0:
            failure_rate = failures / total_actions
            health -= int(failure_rate * 30)  # Up to -30 for failures
        if open_circuits:
            health -= len(open_circuits) * 10  # -10 per open circuit
        if len(self._violations) > 10:
            health -= 10  # -10 if many violations
        health = max(0, min(100, health))

        return SkillResult(
            success=True,
            message=f"Governor report: health={health}/100, {total_actions} actions, {len(self._violations)} violations",
            data={
                "health_score": health,
                "actions": {
                    "total": total_actions,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": round(successes / total_actions * 100, 1) if total_actions > 0 else 100,
                    "total_cost": round(total_cost, 6),
                },
                "last_hour": {
                    "actions": len(recent),
                    "cost": round(recent_cost, 6),
                },
                "budgets": budget_summary,
                "circuit_breakers": circuit_summary,
                "open_circuits": open_circuits,
                "rate_limits": rate_summary,
                "guardrails": {
                    name: {
                        "type": g["rule_type"],
                        "enabled": g["enabled"],
                    }
                    for name, g in self._guardrails.items()
                },
                "violations_count": len(self._violations),
            },
        )

    async def _get_violations(self, params: Dict) -> SkillResult:
        """Get recent violations."""
        limit = params.get("limit", 20)
        recent = self._violations[-limit:]

        return SkillResult(
            success=True,
            message=f"{len(recent)} violations (of {len(self._violations)} total)",
            data={
                "violations": recent,
                "total": len(self._violations),
            },
        )

    async def _reset(self, params: Dict) -> SkillResult:
        """Reset governor state."""
        scope = params.get("scope", "all")

        reset_items = []
        if scope in ("all", "budgets"):
            self._budgets.clear()
            reset_items.append("budgets")
        if scope in ("all", "circuits"):
            self._circuit_breakers.clear()
            reset_items.append("circuit_breakers")
        if scope in ("all", "rates"):
            self._rate_limits.clear()
            reset_items.append("rate_limits")
        if scope in ("all", "history"):
            self._action_history.clear()
            reset_items.append("action_history")
        if scope in ("all", "violations"):
            self._violations.clear()
            reset_items.append("violations")
        if scope in ("all", "guardrails"):
            self._guardrails.clear()
            reset_items.append("guardrails")

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Reset: {', '.join(reset_items)}",
            data={"reset": reset_items},
        )

    # --- Helpers ---

    def _get_circuit_breaker(self, action_id: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for an action (exact match or skill-level)."""
        if action_id in self._circuit_breakers:
            return self._circuit_breakers[action_id]
        # Check skill-level
        skill_id = action_id.split(":")[0] if ":" in action_id else None
        if skill_id and skill_id in self._circuit_breakers:
            return self._circuit_breakers[skill_id]
        return None

    def _get_or_create_circuit_breaker(self, action_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for an action."""
        if action_id not in self._circuit_breakers:
            self._circuit_breakers[action_id] = CircuitBreaker()
        return self._circuit_breakers[action_id]

    def _parse_duration(self, duration: str) -> float:
        """Parse duration string to seconds. Supports: 30s, 5m, 1h, 24h, 7d, 1w."""
        import re
        total = 0.0
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*([smhdw])", duration.lower())
        if not matches:
            try:
                return float(duration)
            except (ValueError, TypeError):
                return 0.0
        for value, unit in matches:
            total += float(value) * multipliers.get(unit, 0)
        return total

    # --- Persistence ---

    def _state_file(self) -> Path:
        return self.DATA_DIR / "governor_state.json"

    def _save_state(self):
        """Save governor state to disk."""
        try:
            self.DATA_DIR.mkdir(parents=True, exist_ok=True)
            state = {
                "budgets": {k: v.to_dict() for k, v in self._budgets.items()},
                "circuit_breakers": {k: v.to_dict() for k, v in self._circuit_breakers.items()},
                "rate_limits": {k: v.to_dict() for k, v in self._rate_limits.items()},
                "guardrails": self._guardrails,
                "action_history": self._action_history[-1000:],
                "violations": self._violations[-500:],
                "saved_at": datetime.now().isoformat(),
            }
            with open(self._state_file(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        """Load governor state from disk."""
        try:
            if self._state_file().exists():
                with open(self._state_file(), "r") as f:
                    state = json.load(f)

                self._budgets = {
                    k: BudgetLimit.from_dict(v)
                    for k, v in state.get("budgets", {}).items()
                }
                self._circuit_breakers = {
                    k: CircuitBreaker.from_dict(v)
                    for k, v in state.get("circuit_breakers", {}).items()
                }
                self._rate_limits = {
                    k: RateLimit.from_dict(v)
                    for k, v in state.get("rate_limits", {}).items()
                }
                self._guardrails = state.get("guardrails", {})
                self._action_history = state.get("action_history", [])
                self._violations = state.get("violations", [])
        except Exception:
            pass
