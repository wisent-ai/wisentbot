#!/usr/bin/env python3
"""
Spawn Strategy Skill - Intelligent replication decision-making.

Gives agents the ability to make smart decisions about WHEN and HOW
to replicate. Without this, agents can spawn replicas via OrchestratorSkill
but have no framework for deciding if spawning is wise.

This skill evaluates:
- Current workload vs capacity
- Budget constraints and burn rate
- Optimal replica configuration
- ROI tracking of past spawns
- Retirement decisions for underperforming replicas
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .base import Skill, SkillManifest, SkillAction, SkillResult


@dataclass
class SpawnCandidate:
    """A potential spawn configuration."""
    purpose: str
    suggested_budget: float
    suggested_model: str
    priority: float  # 0-1, higher = more urgent
    reasoning: str


@dataclass
class ReplicaRecord:
    """Tracking record for a spawned replica."""
    replica_id: str
    name: str
    purpose: str
    budget_given: float
    spawned_at: str
    last_check: str
    revenue_generated: float = 0.0
    cost_incurred: float = 0.0
    status: str = "alive"  # alive, dead, retired


class SpawnStrategySkill(Skill):
    """
    Intelligent spawn decision-making for autonomous agents.

    Evaluates workload, budget, and strategy to recommend when and how
    an agent should replicate. Tracks ROI of past spawns to improve
    future decisions.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._parent_agent = None
        self._replica_records: Dict[str, ReplicaRecord] = {}
        self._spawn_history: List[Dict] = []
        self._strategy_config = {
            "min_balance_to_spawn": 5.0,       # Don't spawn if balance below this
            "max_spawn_budget_ratio": 0.3,     # Never give more than 30% of balance
            "min_spawn_budget": 1.0,           # Minimum budget for a replica
            "max_active_replicas": 5,          # Don't have more than 5 active
            "min_roi_threshold": -0.5,         # Retire if ROI below -50%
            "cooldown_seconds": 300,           # Wait 5 min between spawns
        }
        self._last_spawn_time: float = 0

    def set_parent_agent(self, agent: Any):
        """Connect to the parent agent for state access."""
        self._parent_agent = agent

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="spawn_strategy",
            name="Spawn Strategy",
            version="1.0.0",
            category="replication",
            description="Intelligent decision-making for when and how to spawn replica agents",
            actions=[
                SkillAction(
                    name="evaluate",
                    description="Evaluate whether spawning a new agent is advisable right now. Returns recommendation with reasoning.",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recommend",
                    description="Get a recommended configuration for a new replica based on current needs and budget.",
                    parameters={
                        "purpose_hint": {
                            "type": "string",
                            "required": False,
                            "description": "Optional hint about what kind of agent to spawn"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="calculate_budget",
                    description="Calculate the safe budget to allocate to a new replica.",
                    parameters={
                        "purpose": {
                            "type": "string",
                            "required": True,
                            "description": "What the replica will do"
                        },
                        "priority": {
                            "type": "string",
                            "required": False,
                            "description": "Priority level: low, medium, high, critical"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="register_replica",
                    description="Register a newly spawned replica for ROI tracking.",
                    parameters={
                        "replica_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the spawned replica"
                        },
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name of the replica"
                        },
                        "purpose": {
                            "type": "string",
                            "required": True,
                            "description": "Purpose of the replica"
                        },
                        "budget_given": {
                            "type": "number",
                            "required": True,
                            "description": "Budget allocated to the replica"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="update_replica",
                    description="Update tracking data for a replica (revenue earned, costs, status).",
                    parameters={
                        "replica_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the replica"
                        },
                        "revenue": {
                            "type": "number",
                            "required": False,
                            "description": "Revenue generated by this replica"
                        },
                        "cost": {
                            "type": "number",
                            "required": False,
                            "description": "Cost incurred by this replica"
                        },
                        "status": {
                            "type": "string",
                            "required": False,
                            "description": "Current status: alive, dead, retired"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="retirement_check",
                    description="Check all replicas and recommend which should be retired based on ROI.",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="fleet_status",
                    description="Get overview of all tracked replicas with ROI analysis.",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update spawn strategy configuration parameters.",
                    parameters={
                        "min_balance_to_spawn": {
                            "type": "number",
                            "required": False,
                            "description": "Minimum balance required to consider spawning"
                        },
                        "max_spawn_budget_ratio": {
                            "type": "number",
                            "required": False,
                            "description": "Maximum fraction of balance to give a replica (0-1)"
                        },
                        "max_active_replicas": {
                            "type": "number",
                            "required": False,
                            "description": "Maximum number of active replicas"
                        },
                        "min_roi_threshold": {
                            "type": "number",
                            "required": False,
                            "description": "Minimum ROI before recommending retirement"
                        },
                        "cooldown_seconds": {
                            "type": "number",
                            "required": False,
                            "description": "Minimum seconds between spawns"
                        }
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        """Always available - no external credentials needed."""
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "evaluate": self._evaluate,
            "recommend": self._recommend,
            "calculate_budget": self._calculate_budget,
            "register_replica": self._register_replica,
            "update_replica": self._update_replica,
            "retirement_check": self._retirement_check,
            "fleet_status": self._fleet_status,
            "configure": self._configure,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    async def _evaluate(self, params: Dict) -> SkillResult:
        """Evaluate whether spawning is advisable right now."""
        reasons_for = []
        reasons_against = []
        score = 0.5  # Start neutral

        # 1. Check balance
        balance = self._get_balance()
        min_balance = self._strategy_config["min_balance_to_spawn"]

        if balance < min_balance:
            reasons_against.append(
                f"Balance ${balance:.2f} is below minimum ${min_balance:.2f}"
            )
            score -= 0.3
        elif balance > min_balance * 3:
            reasons_for.append(
                f"Healthy balance ${balance:.2f} (3x+ minimum)"
            )
            score += 0.1
        else:
            reasons_for.append(f"Balance ${balance:.2f} is above minimum")

        # 2. Check active replica count
        active = self._count_active_replicas()
        max_replicas = self._strategy_config["max_active_replicas"]

        if active >= max_replicas:
            reasons_against.append(
                f"Already at max replicas ({active}/{max_replicas})"
            )
            score -= 0.3
        elif active == 0:
            reasons_for.append("No active replicas - could benefit from help")
            score += 0.1
        else:
            reasons_for.append(f"{active}/{max_replicas} replica slots used")

        # 3. Check cooldown
        elapsed = time.time() - self._last_spawn_time
        cooldown = self._strategy_config["cooldown_seconds"]

        if self._last_spawn_time > 0 and elapsed < cooldown:
            remaining = cooldown - elapsed
            reasons_against.append(
                f"Cooldown active ({remaining:.0f}s remaining)"
            )
            score -= 0.2

        # 4. Check burn rate vs runway
        burn_rate = self._get_burn_rate()
        if burn_rate > 0:
            runway_hours = balance / (burn_rate * 3600) if burn_rate > 0 else float('inf')
            if runway_hours < 1:
                reasons_against.append(
                    f"Low runway ({runway_hours:.1f}h) - focus on survival"
                )
                score -= 0.3
            elif runway_hours > 24:
                reasons_for.append(
                    f"Good runway ({runway_hours:.1f}h) - can afford spawning"
                )
                score += 0.1

        # 5. Check historical ROI of past spawns
        avg_roi = self._average_replica_roi()
        if avg_roi is not None:
            if avg_roi > 0:
                reasons_for.append(
                    f"Past replicas show positive ROI ({avg_roi:.1%})"
                )
                score += 0.15
            elif avg_roi < -0.3:
                reasons_against.append(
                    f"Past replicas show poor ROI ({avg_roi:.1%})"
                )
                score -= 0.15

        # Clamp score
        score = max(0.0, min(1.0, score))
        should_spawn = score >= 0.5

        return SkillResult(
            success=True,
            message=f"{'RECOMMEND SPAWN' if should_spawn else 'DO NOT SPAWN'} (confidence: {score:.0%})",
            data={
                "should_spawn": should_spawn,
                "confidence": round(score, 2),
                "reasons_for": reasons_for,
                "reasons_against": reasons_against,
                "balance": balance,
                "active_replicas": active,
                "max_replicas": max_replicas,
            }
        )

    async def _recommend(self, params: Dict) -> SkillResult:
        """Recommend a spawn configuration."""
        purpose_hint = params.get("purpose_hint", "")
        balance = self._get_balance()
        max_budget_ratio = self._strategy_config["max_spawn_budget_ratio"]
        min_budget = self._strategy_config["min_spawn_budget"]

        available_budget = balance * max_budget_ratio
        if available_budget < min_budget:
            return SkillResult(
                success=False,
                message=f"Insufficient budget for spawning. Available: ${available_budget:.2f}, Minimum: ${min_budget:.2f}",
                data={"balance": balance, "available_for_spawn": available_budget}
            )

        # Analyze what types of replicas have worked well
        successful_purposes = []
        failed_purposes = []
        for record in self._replica_records.values():
            roi = self._calculate_roi(record)
            if roi > 0:
                successful_purposes.append(record.purpose)
            elif roi < -0.3:
                failed_purposes.append(record.purpose)

        # Build recommendation
        if purpose_hint:
            purpose = purpose_hint
            priority = 0.7
        elif successful_purposes:
            purpose = f"Similar to past success: {successful_purposes[0][:100]}"
            priority = 0.8
        else:
            purpose = "General-purpose agent for revenue generation or task assistance"
            priority = 0.5

        # Budget calculation: higher priority = higher budget
        budget = min(available_budget, max(min_budget, available_budget * priority))

        # Model recommendation based on budget
        if budget >= 10:
            model = "claude-sonnet-4-20250514"
            model_reasoning = "High budget allows premium model"
        elif budget >= 3:
            model = "claude-sonnet-4-20250514"
            model_reasoning = "Moderate budget, standard model"
        else:
            model = "claude-haiku-4-20250414"
            model_reasoning = "Low budget, using cost-efficient model"

        candidate = SpawnCandidate(
            purpose=purpose,
            suggested_budget=round(budget, 2),
            suggested_model=model,
            priority=priority,
            reasoning=f"Budget: ${budget:.2f} ({max_budget_ratio:.0%} of ${balance:.2f}). {model_reasoning}."
        )

        return SkillResult(
            success=True,
            message=f"Recommended spawn: '{purpose[:60]}' with ${budget:.2f}",
            data={
                "purpose": candidate.purpose,
                "suggested_budget": candidate.suggested_budget,
                "suggested_model": candidate.suggested_model,
                "priority": candidate.priority,
                "reasoning": candidate.reasoning,
                "successful_past_purposes": successful_purposes[:3],
                "failed_past_purposes": failed_purposes[:3],
            }
        )

    async def _calculate_budget(self, params: Dict) -> SkillResult:
        """Calculate safe budget for a replica."""
        purpose = params.get("purpose", "")
        priority_str = params.get("priority", "medium").lower()

        if not purpose:
            return SkillResult(success=False, message="Purpose is required")

        priority_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
            "critical": 2.0,
        }
        multiplier = priority_multipliers.get(priority_str, 1.0)

        balance = self._get_balance()
        max_ratio = self._strategy_config["max_spawn_budget_ratio"]
        min_budget = self._strategy_config["min_spawn_budget"]
        min_balance = self._strategy_config["min_balance_to_spawn"]

        # Base budget is a fraction of balance
        base_budget = balance * max_ratio * 0.5  # Start conservative at half max
        adjusted_budget = base_budget * multiplier

        # Clamp to constraints
        max_possible = balance - min_balance  # Keep minimum for self
        adjusted_budget = min(adjusted_budget, max_possible, balance * max_ratio)
        adjusted_budget = max(adjusted_budget, 0)

        can_afford = adjusted_budget >= min_budget

        return SkillResult(
            success=True,
            message=f"Budget: ${adjusted_budget:.2f} ({'affordable' if can_afford else 'too low'})",
            data={
                "recommended_budget": round(adjusted_budget, 2),
                "can_afford": can_afford,
                "priority": priority_str,
                "multiplier": multiplier,
                "parent_balance": balance,
                "remaining_after_spawn": round(balance - adjusted_budget, 2),
                "min_budget_required": min_budget,
            }
        )

    async def _register_replica(self, params: Dict) -> SkillResult:
        """Register a newly spawned replica for tracking."""
        replica_id = params.get("replica_id", "")
        name = params.get("name", "")
        purpose = params.get("purpose", "")
        budget_given = params.get("budget_given", 0)

        if not replica_id or not name:
            return SkillResult(success=False, message="replica_id and name are required")

        now = datetime.now().isoformat()
        record = ReplicaRecord(
            replica_id=replica_id,
            name=name,
            purpose=purpose,
            budget_given=budget_given,
            spawned_at=now,
            last_check=now,
        )
        self._replica_records[replica_id] = record
        self._last_spawn_time = time.time()

        self._spawn_history.append({
            "replica_id": replica_id,
            "name": name,
            "purpose": purpose,
            "budget_given": budget_given,
            "spawned_at": now,
        })

        return SkillResult(
            success=True,
            message=f"Registered replica '{name}' (${budget_given:.2f})",
            data={
                "replica_id": replica_id,
                "name": name,
                "budget_given": budget_given,
                "total_tracked": len(self._replica_records),
            }
        )

    async def _update_replica(self, params: Dict) -> SkillResult:
        """Update tracking data for a replica."""
        replica_id = params.get("replica_id", "")

        if not replica_id:
            return SkillResult(success=False, message="replica_id is required")

        record = self._replica_records.get(replica_id)
        if not record:
            return SkillResult(success=False, message=f"No replica found: {replica_id}")

        if "revenue" in params:
            record.revenue_generated = params["revenue"]
        if "cost" in params:
            record.cost_incurred = params["cost"]
        if "status" in params:
            record.status = params["status"]

        record.last_check = datetime.now().isoformat()
        roi = self._calculate_roi(record)

        return SkillResult(
            success=True,
            message=f"Updated '{record.name}': ROI={roi:.1%}",
            data={
                "replica_id": replica_id,
                "name": record.name,
                "revenue": record.revenue_generated,
                "cost": record.cost_incurred,
                "budget_given": record.budget_given,
                "roi": round(roi, 4),
                "status": record.status,
            }
        )

    async def _retirement_check(self, params: Dict) -> SkillResult:
        """Check replicas and recommend retirements."""
        threshold = self._strategy_config["min_roi_threshold"]
        recommendations = []
        healthy = []

        for record in self._replica_records.values():
            if record.status != "alive":
                continue

            roi = self._calculate_roi(record)

            if roi < threshold:
                recommendations.append({
                    "replica_id": record.replica_id,
                    "name": record.name,
                    "roi": round(roi, 4),
                    "budget_given": record.budget_given,
                    "revenue": record.revenue_generated,
                    "cost": record.cost_incurred,
                    "recommendation": "retire",
                    "reason": f"ROI {roi:.1%} below threshold {threshold:.1%}",
                })
            else:
                healthy.append({
                    "replica_id": record.replica_id,
                    "name": record.name,
                    "roi": round(roi, 4),
                    "recommendation": "keep",
                })

        return SkillResult(
            success=True,
            message=f"{len(recommendations)} replica(s) recommended for retirement, {len(healthy)} healthy",
            data={
                "retire": recommendations,
                "keep": healthy,
                "threshold": threshold,
            }
        )

    async def _fleet_status(self, params: Dict) -> SkillResult:
        """Get overview of all tracked replicas."""
        fleet = []
        total_invested = 0.0
        total_revenue = 0.0
        total_cost = 0.0

        for record in self._replica_records.values():
            roi = self._calculate_roi(record)
            fleet.append({
                "replica_id": record.replica_id,
                "name": record.name,
                "purpose": record.purpose[:80],
                "budget_given": record.budget_given,
                "revenue": record.revenue_generated,
                "cost": record.cost_incurred,
                "roi": round(roi, 4),
                "status": record.status,
                "spawned_at": record.spawned_at,
            })
            total_invested += record.budget_given
            total_revenue += record.revenue_generated
            total_cost += record.cost_incurred

        active = sum(1 for r in self._replica_records.values() if r.status == "alive")
        fleet_roi = ((total_revenue - total_cost) / total_invested) if total_invested > 0 else 0

        return SkillResult(
            success=True,
            message=f"Fleet: {active} active, {len(fleet)} total, ROI: {fleet_roi:.1%}",
            data={
                "replicas": fleet,
                "summary": {
                    "total_replicas": len(fleet),
                    "active": active,
                    "total_invested": round(total_invested, 2),
                    "total_revenue": round(total_revenue, 2),
                    "total_cost": round(total_cost, 2),
                    "net_return": round(total_revenue - total_cost, 2),
                    "fleet_roi": round(fleet_roi, 4),
                },
                "spawn_history_count": len(self._spawn_history),
                "config": self._strategy_config,
            }
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update strategy configuration."""
        updated = {}
        for key in self._strategy_config:
            if key in params:
                old_val = self._strategy_config[key]
                new_val = params[key]
                # Validate ranges
                if key == "max_spawn_budget_ratio":
                    new_val = max(0.05, min(0.8, float(new_val)))
                elif key == "max_active_replicas":
                    new_val = max(1, min(20, int(new_val)))
                elif key == "min_roi_threshold":
                    new_val = max(-1.0, min(1.0, float(new_val)))
                elif key == "cooldown_seconds":
                    new_val = max(0, float(new_val))
                else:
                    new_val = float(new_val)

                self._strategy_config[key] = new_val
                updated[key] = {"old": old_val, "new": new_val}

        if not updated:
            return SkillResult(
                success=True,
                message="No changes made",
                data={"current_config": self._strategy_config}
            )

        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} config parameter(s)",
            data={"updated": updated, "current_config": self._strategy_config}
        )

    # === Internal helpers ===

    def _get_balance(self) -> float:
        """Get parent agent's current balance."""
        if self._parent_agent and hasattr(self._parent_agent, 'balance'):
            return self._parent_agent.balance
        return 0.0

    def _get_burn_rate(self) -> float:
        """Get parent agent's burn rate (cost per second)."""
        if self._parent_agent:
            cycle_interval = getattr(self._parent_agent, 'cycle_interval', 5.0)
            instance_cost = getattr(self._parent_agent, 'instance_cost_per_hour', 0.0)
            # Estimate: API cost per cycle ~$0.01 + instance cost
            est_api_per_cycle = 0.01
            est_cost_per_cycle = est_api_per_cycle + (instance_cost * cycle_interval / 3600)
            return est_cost_per_cycle / cycle_interval if cycle_interval > 0 else 0
        return 0.0

    def _count_active_replicas(self) -> int:
        """Count active replicas."""
        return sum(1 for r in self._replica_records.values() if r.status == "alive")

    def _calculate_roi(self, record: ReplicaRecord) -> float:
        """Calculate ROI for a replica. ROI = (revenue - cost) / budget_given."""
        if record.budget_given <= 0:
            return 0.0
        return (record.revenue_generated - record.cost_incurred) / record.budget_given

    def _average_replica_roi(self) -> Optional[float]:
        """Get average ROI across all replicas that have finished."""
        finished = [r for r in self._replica_records.values() if r.status != "alive"]
        if not finished:
            return None
        rois = [self._calculate_roi(r) for r in finished]
        return sum(rois) / len(rois)
