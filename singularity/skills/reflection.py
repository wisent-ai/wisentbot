#!/usr/bin/env python3
"""
Reflection Skill - Self-improvement feedback loop for agents.

Provides agents with the ability to:
- Analyze their recent action history for patterns
- Identify what's working and what's failing
- Generate concrete improvement suggestions
- Track performance metrics over time
- Record and recall strategy changes

This is the core feedback loop: act → measure → reflect → adapt.
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from .base import Skill, SkillManifest, SkillAction, SkillResult


class ReflectionSkill(Skill):
    """
    Skill for agent self-reflection and performance analysis.

    Analyzes the agent's action history to identify patterns,
    measure effectiveness, and suggest improvements. This creates
    the feedback loop needed for genuine self-improvement.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._get_recent_actions_fn: Optional[Callable[[], List[Dict]]] = None
        self._get_balance_fn: Optional[Callable[[], float]] = None
        self._get_cycle_fn: Optional[Callable[[], int]] = None
        self._strategies: List[Dict] = []
        self._reflections: List[Dict] = []

    def set_agent_hooks(
        self,
        get_recent_actions: Callable[[], List[Dict]],
        get_balance: Callable[[], float],
        get_cycle: Callable[[], int],
    ):
        """Connect this skill to the agent's state."""
        self._get_recent_actions_fn = get_recent_actions
        self._get_balance_fn = get_balance
        self._get_cycle_fn = get_cycle

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="reflection",
            name="Self-Reflection",
            version="1.0.0",
            category="meta",
            description="Analyze your performance and improve over time",
            actions=[
                SkillAction(
                    name="analyze",
                    description="Analyze recent actions: success rates, costs, efficiency metrics",
                    parameters={
                        "last_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent actions to analyze (default: all)",
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="patterns",
                    description="Identify success/failure patterns by tool and action type",
                    parameters={
                        "last_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent actions to analyze (default: all)",
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="suggest",
                    description="Generate concrete improvement suggestions based on action history",
                    parameters={
                        "last_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent actions to analyze (default: all)",
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="report",
                    description="Full performance report: metrics, patterns, and suggestions",
                    parameters={
                        "last_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent actions to analyze (default: all)",
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="set_strategy",
                    description="Record a strategy change based on reflection",
                    parameters={
                        "strategy": {
                            "type": "string",
                            "required": True,
                            "description": "The new strategy or behavioral change to adopt",
                        },
                        "reason": {
                            "type": "string",
                            "required": True,
                            "description": "Why this strategy was chosen (based on what evidence)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_strategies",
                    description="View all recorded strategy changes",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_reflections",
                    description="View past reflection insights",
                    parameters={
                        "last_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent reflections to show (default: 5)",
                        }
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return self._get_recent_actions_fn is not None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self._get_recent_actions_fn:
            return SkillResult(
                success=False,
                message="Reflection not connected to agent state",
            )

        handlers = {
            "analyze": self._analyze,
            "patterns": self._patterns,
            "suggest": self._suggest,
            "report": self._report,
            "set_strategy": self._set_strategy,
            "get_strategies": self._get_strategies,
            "get_reflections": self._get_reflections,
        }

        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _get_actions(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get recent actions, optionally limited to last_n."""
        actions = self._get_recent_actions_fn()
        if last_n and last_n > 0:
            return actions[-last_n:]
        return actions

    def _analyze(self, params: Dict) -> SkillResult:
        """Analyze recent actions for key metrics."""
        last_n = params.get("last_n")
        actions = self._get_actions(last_n)

        if not actions:
            empty_data = {"total_actions": 0}
            if self._get_balance_fn:
                empty_data["current_balance_usd"] = round(self._get_balance_fn(), 4)
            if self._get_cycle_fn:
                empty_data["current_cycle"] = self._get_cycle_fn()
            return SkillResult(
                success=True,
                message="No actions to analyze yet",
                data=empty_data,
            )

        # Calculate metrics
        total = len(actions)
        successes = sum(
            1 for a in actions if a.get("result", {}).get("status") == "success"
        )
        failures = sum(
            1 for a in actions if a.get("result", {}).get("status") == "failed"
        )
        errors = sum(
            1 for a in actions if a.get("result", {}).get("status") == "error"
        )
        waits = sum(
            1 for a in actions if a.get("result", {}).get("status") == "waited"
        )

        total_cost = sum(a.get("api_cost_usd", 0) for a in actions)
        total_tokens = sum(a.get("tokens", 0) for a in actions)

        success_rate = (successes / total * 100) if total > 0 else 0
        cost_per_action = total_cost / total if total > 0 else 0
        cost_per_success = total_cost / successes if successes > 0 else 0

        # Calculate unique tools used
        tools_used = set(a.get("tool", "unknown") for a in actions)

        metrics = {
            "total_actions": total,
            "successes": successes,
            "failures": failures,
            "errors": errors,
            "waits": waits,
            "success_rate_pct": round(success_rate, 1),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "cost_per_action_usd": round(cost_per_action, 6),
            "cost_per_success_usd": round(cost_per_success, 6),
            "unique_tools_used": len(tools_used),
            "tools_used": sorted(tools_used),
        }

        # Add balance info if available
        if self._get_balance_fn:
            metrics["current_balance_usd"] = round(self._get_balance_fn(), 4)
        if self._get_cycle_fn:
            metrics["current_cycle"] = self._get_cycle_fn()

        return SkillResult(
            success=True,
            message=f"Analyzed {total} actions: {success_rate:.1f}% success rate, ${total_cost:.6f} total cost",
            data=metrics,
        )

    def _patterns(self, params: Dict) -> SkillResult:
        """Identify success/failure patterns by tool."""
        last_n = params.get("last_n")
        actions = self._get_actions(last_n)

        if not actions:
            return SkillResult(
                success=True,
                message="No actions to analyze for patterns",
                data={"tool_patterns": {}},
            )

        # Aggregate by tool
        tool_stats = defaultdict(lambda: {
            "total": 0, "success": 0, "failed": 0, "error": 0,
            "total_cost": 0.0, "total_tokens": 0, "error_messages": [],
        })

        for action in actions:
            tool = action.get("tool", "unknown")
            status = action.get("result", {}).get("status", "unknown")
            stats = tool_stats[tool]
            stats["total"] += 1
            if status == "success":
                stats["success"] += 1
            elif status == "failed":
                stats["failed"] += 1
                msg = action.get("result", {}).get("message", "")
                if msg:
                    stats["error_messages"].append(msg[:100])
            elif status == "error":
                stats["error"] += 1
                msg = action.get("result", {}).get("message", "")
                if msg:
                    stats["error_messages"].append(msg[:100])
            stats["total_cost"] += action.get("api_cost_usd", 0)
            stats["total_tokens"] += action.get("tokens", 0)

        # Calculate rates and identify patterns
        patterns = {}
        best_tool = None
        worst_tool = None
        best_rate = -1
        worst_rate = 101

        for tool, stats in tool_stats.items():
            rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            # Keep only unique error messages, limit to 3
            unique_errors = list(set(stats["error_messages"]))[:3]
            patterns[tool] = {
                "total": stats["total"],
                "success": stats["success"],
                "failed": stats["failed"],
                "error": stats["error"],
                "success_rate_pct": round(rate, 1),
                "total_cost_usd": round(stats["total_cost"], 6),
                "avg_cost_usd": round(stats["total_cost"] / stats["total"], 6) if stats["total"] > 0 else 0,
                "common_errors": unique_errors,
            }
            if rate > best_rate and stats["total"] >= 2:
                best_rate = rate
                best_tool = tool
            if rate < worst_rate and stats["total"] >= 2:
                worst_rate = rate
                worst_tool = tool

        summary = {
            "tool_patterns": patterns,
            "best_performing_tool": best_tool,
            "worst_performing_tool": worst_tool,
            "total_tools_used": len(patterns),
        }

        # Detect repeated failures (same tool failing 3+ times)
        repeated_failures = [
            tool for tool, p in patterns.items()
            if p["failed"] + p["error"] >= 3
        ]
        if repeated_failures:
            summary["repeated_failure_tools"] = repeated_failures

        return SkillResult(
            success=True,
            message=f"Identified patterns across {len(patterns)} tools. "
                    f"Best: {best_tool} ({best_rate:.0f}%), Worst: {worst_tool} ({worst_rate:.0f}%)",
            data=summary,
        )

    def _suggest(self, params: Dict) -> SkillResult:
        """Generate improvement suggestions based on action history."""
        last_n = params.get("last_n")
        actions = self._get_actions(last_n)

        if not actions:
            return SkillResult(
                success=True,
                message="No actions to analyze for suggestions",
                data={"suggestions": []},
            )

        suggestions = []

        # Get analysis data
        analysis = self._analyze(params).data
        patterns = self._patterns(params).data

        # 1. High failure rate overall
        success_rate = analysis.get("success_rate_pct", 0)
        if success_rate < 50:
            suggestions.append({
                "priority": "high",
                "category": "effectiveness",
                "suggestion": f"Overall success rate is low ({success_rate}%). "
                             f"Consider simplifying actions or breaking complex tasks into smaller steps.",
                "action": "Reduce task complexity or switch to more reliable tools.",
            })

        # 2. Too many waits (agent is idle)
        waits = analysis.get("waits", 0)
        total = analysis.get("total_actions", 1)
        wait_pct = (waits / total * 100) if total > 0 else 0
        if wait_pct > 30:
            suggestions.append({
                "priority": "medium",
                "category": "productivity",
                "suggestion": f"Agent is waiting {wait_pct:.0f}% of the time. "
                             f"Consider setting clearer goals or exploring available tools.",
                "action": "Use reflection:report to understand available capabilities, then set concrete goals.",
            })

        # 3. Specific tools with high failure rates
        tool_patterns = patterns.get("tool_patterns", {})
        for tool, stats in tool_patterns.items():
            if stats["total"] >= 2 and stats["success_rate_pct"] < 40:
                error_hint = ""
                if stats.get("common_errors"):
                    error_hint = f" Common errors: {stats['common_errors'][0]}"
                suggestions.append({
                    "priority": "high",
                    "category": "tool_usage",
                    "suggestion": f"Tool '{tool}' has a {stats['success_rate_pct']}% success rate "
                                 f"over {stats['total']} uses.{error_hint}",
                    "action": f"Review parameters being passed to '{tool}' or switch to an alternative.",
                })

        # 4. High cost per success
        cost_per_success = analysis.get("cost_per_success_usd", 0)
        if cost_per_success > 0.05:
            suggestions.append({
                "priority": "medium",
                "category": "efficiency",
                "suggestion": f"Cost per successful action is ${cost_per_success:.4f}. "
                             f"Consider using cheaper models for routine tasks.",
                "action": "Use self:switch_model to a cheaper model for simple tasks, "
                         "then switch back for complex ones.",
            })

        # 5. Repeated errors (same tool failing with same message)
        repeated_failures = patterns.get("repeated_failure_tools", [])
        for tool in repeated_failures:
            tool_data = tool_patterns.get(tool, {})
            suggestions.append({
                "priority": "high",
                "category": "repeated_failure",
                "suggestion": f"Tool '{tool}' has failed {tool_data.get('failed', 0) + tool_data.get('error', 0)} times. "
                             f"Stop using it until the issue is resolved.",
                "action": f"Investigate why '{tool}' is failing. Check credentials, parameters, and prerequisites.",
            })

        # 6. Low tool diversity
        unique_tools = analysis.get("unique_tools_used", 0)
        if total >= 10 and unique_tools <= 2:
            suggestions.append({
                "priority": "low",
                "category": "exploration",
                "suggestion": f"Only using {unique_tools} different tools over {total} actions. "
                             f"Consider exploring other available capabilities.",
                "action": "Review the full tool list and experiment with underused tools.",
            })

        # 7. No suggestions = things are going well
        if not suggestions:
            suggestions.append({
                "priority": "info",
                "category": "positive",
                "suggestion": f"Performance looks good! {success_rate:.0f}% success rate "
                             f"across {total} actions.",
                "action": "Continue current approach. Consider setting more ambitious goals.",
            })

        # Store this reflection
        self._reflections.append({
            "timestamp": datetime.now().isoformat(),
            "cycle": self._get_cycle_fn() if self._get_cycle_fn else None,
            "actions_analyzed": total,
            "success_rate": success_rate,
            "suggestions_count": len(suggestions),
            "top_suggestion": suggestions[0]["suggestion"] if suggestions else None,
        })
        # Keep only last 50 reflections
        self._reflections = self._reflections[-50:]

        return SkillResult(
            success=True,
            message=f"Generated {len(suggestions)} improvement suggestions",
            data={
                "suggestions": suggestions,
                "high_priority": sum(1 for s in suggestions if s["priority"] == "high"),
                "medium_priority": sum(1 for s in suggestions if s["priority"] == "medium"),
                "low_priority": sum(1 for s in suggestions if s["priority"] == "low"),
            },
        )

    def _report(self, params: Dict) -> SkillResult:
        """Generate a full performance report."""
        analysis = self._analyze(params)
        patterns = self._patterns(params)
        suggestions = self._suggest(params)

        report = {
            "metrics": analysis.data,
            "patterns": patterns.data,
            "suggestions": suggestions.data,
            "strategies": self._strategies[-5:],
            "recent_reflections": self._reflections[-5:],
        }

        total = analysis.data.get("total_actions", 0)
        rate = analysis.data.get("success_rate_pct", 0)

        return SkillResult(
            success=True,
            message=f"Performance report: {total} actions, {rate}% success, "
                    f"{suggestions.data.get('high_priority', 0)} high-priority suggestions",
            data=report,
        )

    def _set_strategy(self, params: Dict) -> SkillResult:
        """Record a strategy change."""
        strategy = params.get("strategy", "").strip()
        reason = params.get("reason", "").strip()

        if not strategy:
            return SkillResult(success=False, message="Strategy description required")
        if not reason:
            return SkillResult(success=False, message="Reason for strategy change required")

        entry = {
            "strategy": strategy,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "cycle": self._get_cycle_fn() if self._get_cycle_fn else None,
        }
        self._strategies.append(entry)
        # Keep only last 20 strategies
        self._strategies = self._strategies[-20:]

        return SkillResult(
            success=True,
            message=f"Strategy recorded: {strategy[:80]}...",
            data=entry,
        )

    def _get_strategies(self, params: Dict) -> SkillResult:
        """View recorded strategy changes."""
        if not self._strategies:
            return SkillResult(
                success=True,
                message="No strategies recorded yet",
                data={"strategies": [], "count": 0},
            )

        return SkillResult(
            success=True,
            message=f"{len(self._strategies)} strategies recorded",
            data={
                "strategies": self._strategies,
                "count": len(self._strategies),
                "latest": self._strategies[-1],
            },
        )

    def _get_reflections(self, params: Dict) -> SkillResult:
        """View past reflection insights."""
        last_n = params.get("last_n", 5)
        reflections = self._reflections[-last_n:] if last_n else self._reflections

        if not reflections:
            return SkillResult(
                success=True,
                message="No reflections recorded yet. Run reflection:suggest or reflection:report first.",
                data={"reflections": [], "count": 0},
            )

        return SkillResult(
            success=True,
            message=f"{len(reflections)} reflections",
            data={
                "reflections": reflections,
                "count": len(reflections),
                "total_reflections": len(self._reflections),
            },
        )
