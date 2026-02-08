#!/usr/bin/env python3
"""
AutoReputationBridgeSkill - Automatically updates agent reputation from task delegation outcomes.

This is the critical missing link between task delegation and reputation tracking.
When TaskDelegationSkill records a delegation as completed or failed, this bridge
automatically calls AgentReputationSkill.record_task_outcome so that reputation
scores stay in sync with actual agent performance.

Without this bridge, reputation scores would only update when someone manually
calls record_task_outcome. This means agents that consistently deliver great work
wouldn't see their scores rise, and unreliable agents wouldn't be penalized.

This completes the delegation → reputation feedback loop:
  delegate task → agent works → report_completion → auto-reputation update
  → better delegation decisions next time (weighted by reputation)

Pillar: Self-Improvement + Replication
- Self-Improvement: Closes the act → measure → adapt loop for delegation quality
- Replication: Trust scores between parent/child agents stay accurate automatically

Actions:
- poll: Scan completed/failed delegations and update reputation for unprocessed ones
- configure: Set scoring weights, enable/disable auto-polling, and configure thresholds
- status: View bridge health, processed counts, and recent updates
- history: View recent reputation updates triggered by this bridge
- reprocess: Force-reprocess a specific delegation (e.g., after reputation reset)
- stats: Aggregated statistics on delegation outcomes and reputation impacts
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_DATA_FILE = Path(__file__).parent.parent / "data" / "auto_reputation_bridge.json"
DELEGATION_FILE = Path(__file__).parent.parent / "data" / "task_delegations.json"
REPUTATION_FILE = Path(__file__).parent.parent / "data" / "agent_reputation.json"

MAX_HISTORY = 500
MAX_POLL_BATCH = 50


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# Default scoring config
DEFAULT_CONFIG = {
    "enabled": True,
    "auto_poll_on_report": True,
    # Competence deltas
    "success_competence_base": 2.0,
    "success_competence_efficiency_bonus": 3.0,  # scaled by budget efficiency
    "failure_competence_penalty": -3.0,
    # Reliability deltas
    "success_on_time_bonus": 2.0,
    "success_late_penalty": -1.0,
    "failure_reliability_penalty": -2.0,
    # Cooperation bonus for completing work delegated by another agent
    "success_cooperation_bonus": 1.0,
    # Timeout detection: if delegation has a deadline and was late
    "late_threshold_factor": 1.5,  # task took > 1.5x the estimated time = late
    # EventBus integration
    "emit_events": True,
}


class AutoReputationBridgeSkill(Skill):
    """
    Automatically bridge task delegation outcomes to agent reputation updates.

    Scans TaskDelegationSkill data for completed/failed delegations that haven't
    been processed yet, and calls AgentReputationSkill.record_task_outcome for each.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = self._load_state()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="auto_reputation_bridge",
            name="Auto Reputation Bridge",
            version="1.0.0",
            category="replication",
            description=(
                "Automatically updates agent reputation scores from task delegation outcomes. "
                "Bridges TaskDelegationSkill → AgentReputationSkill so reputation stays in sync "
                "with actual agent performance."
            ),
            actions=[
                SkillAction(
                    name="poll",
                    description="Scan completed/failed delegations and update reputation for unprocessed ones",
                    parameters={
                        "dry_run": {
                            "type": "bool",
                            "required": False,
                            "description": "Preview what would be updated without executing (default: false)",
                        },
                        "limit": {
                            "type": "int",
                            "required": False,
                            "description": "Max delegations to process per poll (default: 50)",
                        },
                    },
                ),
                SkillAction(
                    name="configure",
                    description="Set scoring weights and behavior options",
                    parameters={
                        "key": {
                            "type": "string",
                            "required": True,
                            "description": "Config key to set (e.g., success_competence_base, enabled)",
                        },
                        "value": {
                            "type": "any",
                            "required": True,
                            "description": "Value to set",
                        },
                    },
                ),
                SkillAction(
                    name="status",
                    description="View bridge health, processed counts, and configuration",
                    parameters={},
                ),
                SkillAction(
                    name="history",
                    description="View recent reputation updates triggered by this bridge",
                    parameters={
                        "limit": {
                            "type": "int",
                            "required": False,
                            "description": "Number of history entries to return (default: 20)",
                        },
                        "agent_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by agent ID",
                        },
                    },
                ),
                SkillAction(
                    name="reprocess",
                    description="Force-reprocess a specific delegation (e.g., after reputation reset)",
                    parameters={
                        "delegation_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the delegation to reprocess",
                        },
                    },
                ),
                SkillAction(
                    name="stats",
                    description="Aggregated statistics on delegation outcomes and reputation impacts",
                    parameters={
                        "agent_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter stats by agent ID (default: all agents)",
                        },
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "poll": self._poll,
            "configure": self._configure,
            "status": self._status,
            "history": self._history,
            "reprocess": self._reprocess,
            "stats": self._stats,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # ── Poll: Core bridge logic ───────────────────────────────

    async def _poll(self, params: Dict) -> SkillResult:
        """Scan delegations and update reputation for new completions/failures."""
        config = self._state.get("config", DEFAULT_CONFIG)
        if not config.get("enabled", True):
            return SkillResult(success=False, message="Bridge is disabled. Use configure to enable.")

        dry_run = params.get("dry_run", False)
        limit = int(params.get("limit", MAX_POLL_BATCH))

        # Load delegation data
        delegations = self._load_delegations()
        if not delegations:
            return SkillResult(success=True, message="No delegations found", data={"processed": 0})

        processed_ids = set(self._state.get("processed_ids", []))
        updates = []

        # Find unprocessed completed/failed delegations
        for d in delegations:
            if len(updates) >= limit:
                break
            d_id = d.get("delegation_id", "")
            status = d.get("status", "")
            if status not in ("completed", "failed"):
                continue
            if d_id in processed_ids:
                continue

            agent_id = d.get("agent_id", "")
            if not agent_id:
                continue

            # Compute update parameters
            success = status == "completed"
            budget = float(d.get("budget", 0))
            budget_spent = float(d.get("budget_spent", 0))
            budget_efficiency = (budget_spent / budget) if budget > 0 else 0.5
            task_name = d.get("task_name", "unknown")

            # Determine if on-time
            on_time = self._check_on_time(d)

            update = {
                "delegation_id": d_id,
                "agent_id": agent_id,
                "task_name": task_name,
                "success": success,
                "budget_efficiency": round(budget_efficiency, 3),
                "on_time": on_time,
                "status": status,
                "timestamp": _now_iso(),
            }

            if not dry_run:
                # Try via SkillContext first, fall back to direct
                rep_result = await self._update_reputation(
                    agent_id=agent_id,
                    success=success,
                    budget_efficiency=budget_efficiency,
                    on_time=on_time,
                    task_name=task_name,
                    config=config,
                )
                update["reputation_result"] = rep_result
                processed_ids.add(d_id)

            updates.append(update)

        # Persist state
        if not dry_run and updates:
            self._state["processed_ids"] = list(processed_ids)
            self._state.setdefault("stats", {})["last_poll"] = _now_iso()
            self._state["stats"]["total_processed"] = len(processed_ids)

            # Add to history
            history = self._state.setdefault("history", [])
            for u in updates:
                history.append(u)
            # Trim history
            if len(history) > MAX_HISTORY:
                self._state["history"] = history[-MAX_HISTORY:]

            self._save_state()

            # Emit events if configured
            if config.get("emit_events", True):
                await self._emit_events(updates)

        action_word = "Would process" if dry_run else "Processed"
        successes = sum(1 for u in updates if u["success"])
        failures = len(updates) - successes

        return SkillResult(
            success=True,
            message=f"{action_word} {len(updates)} delegation(s): {successes} success, {failures} failure",
            data={
                "processed": len(updates),
                "successes": successes,
                "failures": failures,
                "dry_run": dry_run,
                "updates": updates,
            },
        )

    async def _update_reputation(
        self,
        agent_id: str,
        success: bool,
        budget_efficiency: float,
        on_time: bool,
        task_name: str,
        config: Dict,
    ) -> Dict:
        """Update reputation via SkillContext or direct file manipulation."""
        # Try SkillContext first (preferred - uses live AgentReputationSkill)
        if self.context:
            try:
                result = await self.context.call_skill(
                    "agent_reputation",
                    "record_task_outcome",
                    {
                        "agent_id": agent_id,
                        "success": success,
                        "budget_efficiency": budget_efficiency,
                        "on_time": on_time,
                        "task_name": task_name,
                    },
                )
                if result.success:
                    return {
                        "method": "skill_context",
                        "success": True,
                        "data": result.data,
                    }
            except Exception as e:
                pass  # Fall through to direct method

        # Fallback: direct file-based update
        return self._update_reputation_direct(
            agent_id, success, budget_efficiency, on_time, task_name, config
        )

    def _update_reputation_direct(
        self,
        agent_id: str,
        success: bool,
        budget_efficiency: float,
        on_time: bool,
        task_name: str,
        config: Dict,
    ) -> Dict:
        """Directly update reputation file when SkillContext is unavailable."""
        try:
            rep_data = {}
            if REPUTATION_FILE.exists():
                with open(REPUTATION_FILE) as f:
                    rep_data = json.load(f)

            reputations = rep_data.setdefault("reputations", {})
            events = rep_data.setdefault("events", [])

            # Get or create agent reputation
            agent_rep = reputations.get(agent_id)
            if not agent_rep:
                now = datetime.utcnow().isoformat()
                agent_rep = {
                    "agent_id": agent_id,
                    "competence": 50.0,
                    "reliability": 50.0,
                    "trustworthiness": 50.0,
                    "leadership": 50.0,
                    "cooperation": 50.0,
                    "overall": 50.0,
                    "total_events": 0,
                    "total_tasks": 0,
                    "tasks_completed": 0,
                    "tasks_failed": 0,
                    "votes_cast": 0,
                    "endorsements_received": 0,
                    "penalties_received": 0,
                    "first_seen": now,
                    "last_updated": now,
                }
                reputations[agent_id] = agent_rep

            agent_rep["total_tasks"] = agent_rep.get("total_tasks", 0) + 1

            def clamp(v):
                return max(0.0, min(100.0, v))

            def add_event(event_type, dimension, delta, details=None):
                evt_id = f"evt_{len(events):06d}"
                events.append({
                    "event_id": evt_id,
                    "agent_id": agent_id,
                    "event_type": event_type,
                    "dimension": dimension,
                    "delta": delta,
                    "source": "auto_reputation_bridge",
                    "details": details or {},
                    "created_at": datetime.utcnow().isoformat(),
                })
                agent_rep[dimension] = clamp(agent_rep.get(dimension, 50.0) + delta)
                agent_rep["total_events"] = agent_rep.get("total_events", 0) + 1

            deltas_applied = {}

            if success:
                agent_rep["tasks_completed"] = agent_rep.get("tasks_completed", 0) + 1
                # Competence: base + efficiency bonus
                comp_base = config.get("success_competence_base", 2.0)
                comp_bonus = config.get("success_competence_efficiency_bonus", 3.0)
                comp_delta = comp_base + comp_bonus * max(0, 1.0 - budget_efficiency)
                add_event("task_completed", "competence", comp_delta,
                         {"task": task_name, "budget_efficiency": budget_efficiency})
                deltas_applied["competence"] = round(comp_delta, 2)

                # Reliability
                if on_time:
                    rel_delta = config.get("success_on_time_bonus", 2.0)
                else:
                    rel_delta = config.get("success_late_penalty", -1.0)
                add_event("task_completed", "reliability", rel_delta, {"on_time": on_time})
                deltas_applied["reliability"] = round(rel_delta, 2)

                # Cooperation bonus
                coop_delta = config.get("success_cooperation_bonus", 1.0)
                add_event("task_completed", "cooperation", coop_delta,
                         {"task": task_name})
                deltas_applied["cooperation"] = round(coop_delta, 2)
            else:
                agent_rep["tasks_failed"] = agent_rep.get("tasks_failed", 0) + 1
                comp_delta = config.get("failure_competence_penalty", -3.0)
                add_event("task_failed", "competence", comp_delta, {"task": task_name})
                deltas_applied["competence"] = round(comp_delta, 2)

                rel_delta = config.get("failure_reliability_penalty", -2.0)
                add_event("task_failed", "reliability", rel_delta, {"task": task_name})
                deltas_applied["reliability"] = round(rel_delta, 2)

            # Recompute overall
            weights = {
                "competence": 0.30,
                "reliability": 0.25,
                "trustworthiness": 0.20,
                "leadership": 0.10,
                "cooperation": 0.15,
            }
            total_weight = sum(weights.values())
            agent_rep["overall"] = sum(
                weights[d] * agent_rep.get(d, 50.0) for d in weights
            ) / total_weight
            agent_rep["last_updated"] = datetime.utcnow().isoformat()

            # Trim events
            if len(events) > 5000:
                rep_data["events"] = events[-5000:]

            # Save
            REPUTATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(REPUTATION_FILE, "w") as f:
                json.dump(rep_data, f, indent=2)

            return {
                "method": "direct_file",
                "success": True,
                "deltas": deltas_applied,
                "new_overall": round(agent_rep["overall"], 1),
                "competence": round(agent_rep["competence"], 1),
                "reliability": round(agent_rep["reliability"], 1),
            }
        except Exception as e:
            return {"method": "direct_file", "success": False, "error": str(e)}

    def _check_on_time(self, delegation: Dict) -> bool:
        """Check if a delegation was completed on time."""
        created = delegation.get("created_at")
        completed = delegation.get("completed_at")
        timeout = delegation.get("timeout_minutes")

        if not created or not completed or not timeout:
            return True  # Assume on-time if we can't determine

        try:
            # Parse dates (handle both ISO formats)
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00").replace("+00:00", ""))
            completed_dt = datetime.fromisoformat(completed.replace("Z", "+00:00").replace("+00:00", ""))
            elapsed_minutes = (completed_dt - created_dt).total_seconds() / 60.0
            return elapsed_minutes <= float(timeout)
        except Exception:
            return True

    async def _emit_events(self, updates: List[Dict]):
        """Emit EventBus events for downstream automation."""
        if not self.context:
            return
        try:
            for u in updates:
                event_type = "reputation_bridge.updated"
                if u["success"]:
                    event_type = "reputation_bridge.success"
                else:
                    event_type = "reputation_bridge.failure"

                await self.context.call_skill(
                    "event",
                    "publish",
                    {
                        "topic": event_type,
                        "data": {
                            "agent_id": u["agent_id"],
                            "delegation_id": u["delegation_id"],
                            "task_name": u["task_name"],
                            "success": u["success"],
                            "budget_efficiency": u["budget_efficiency"],
                        },
                    },
                )
        except Exception:
            pass  # Non-critical

    # ── Configure ─────────────────────────────────────────────

    async def _configure(self, params: Dict) -> SkillResult:
        """Update bridge configuration."""
        key = params.get("key", "").strip()
        value = params.get("value")

        if not key:
            return SkillResult(success=False, message="key is required")

        config = self._state.setdefault("config", dict(DEFAULT_CONFIG))
        valid_keys = set(DEFAULT_CONFIG.keys())

        if key not in valid_keys:
            return SkillResult(
                success=False,
                message=f"Unknown config key: {key}. Valid: {sorted(valid_keys)}",
            )

        old_value = config.get(key)
        config[key] = value
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Config updated: {key} = {value} (was: {old_value})",
            data={"key": key, "value": value, "old_value": old_value, "config": config},
        )

    # ── Status ────────────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        """View bridge health and counts."""
        config = self._state.get("config", DEFAULT_CONFIG)
        stats = self._state.get("stats", {})
        processed_count = len(self._state.get("processed_ids", []))
        history_count = len(self._state.get("history", []))

        # Count pending delegations
        delegations = self._load_delegations()
        processed_ids = set(self._state.get("processed_ids", []))
        pending = sum(
            1 for d in delegations
            if d.get("status") in ("completed", "failed")
            and d.get("delegation_id") not in processed_ids
        )

        return SkillResult(
            success=True,
            message=f"Bridge {'enabled' if config.get('enabled') else 'DISABLED'}: "
                    f"{processed_count} processed, {pending} pending",
            data={
                "enabled": config.get("enabled", True),
                "total_processed": processed_count,
                "pending": pending,
                "history_entries": history_count,
                "last_poll": stats.get("last_poll"),
                "config": config,
            },
        )

    # ── History ───────────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        """View recent reputation updates."""
        limit = int(params.get("limit", 20))
        agent_filter = params.get("agent_id")

        history = self._state.get("history", [])
        if agent_filter:
            history = [h for h in history if h.get("agent_id") == agent_filter]

        # Most recent first
        recent = history[-limit:][::-1]

        return SkillResult(
            success=True,
            message=f"History: {len(recent)} entries" + (f" for {agent_filter}" if agent_filter else ""),
            data={"entries": recent, "total": len(history)},
        )

    # ── Reprocess ─────────────────────────────────────────────

    async def _reprocess(self, params: Dict) -> SkillResult:
        """Force reprocess a specific delegation."""
        delegation_id = params.get("delegation_id", "").strip()
        if not delegation_id:
            return SkillResult(success=False, message="delegation_id is required")

        # Remove from processed set
        processed_ids = set(self._state.get("processed_ids", []))
        was_processed = delegation_id in processed_ids
        processed_ids.discard(delegation_id)
        self._state["processed_ids"] = list(processed_ids)
        self._save_state()

        # Now poll to pick it up
        result = await self._poll({"limit": 1})

        return SkillResult(
            success=True,
            message=f"Reprocessed {delegation_id} (was_processed={was_processed}). {result.message}",
            data={
                "delegation_id": delegation_id,
                "was_processed": was_processed,
                "poll_result": result.data,
            },
        )

    # ── Stats ─────────────────────────────────────────────────

    async def _stats(self, params: Dict) -> SkillResult:
        """Aggregated statistics on delegation outcomes."""
        agent_filter = params.get("agent_id")
        history = self._state.get("history", [])

        if agent_filter:
            history = [h for h in history if h.get("agent_id") == agent_filter]

        if not history:
            return SkillResult(
                success=True,
                message="No data yet",
                data={"total": 0},
            )

        total = len(history)
        successes = sum(1 for h in history if h.get("success"))
        failures = total - successes
        success_rate = successes / total if total > 0 else 0

        # Budget efficiency stats for successes
        efficiencies = [
            h["budget_efficiency"] for h in history
            if h.get("success") and "budget_efficiency" in h
        ]
        avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0

        # On-time stats
        on_time_count = sum(1 for h in history if h.get("on_time", False))
        on_time_rate = on_time_count / total if total > 0 else 0

        # Per-agent breakdown
        agent_stats = {}
        for h in history:
            aid = h.get("agent_id", "unknown")
            a = agent_stats.setdefault(aid, {"success": 0, "failure": 0, "total": 0})
            a["total"] += 1
            if h.get("success"):
                a["success"] += 1
            else:
                a["failure"] += 1

        for aid, a in agent_stats.items():
            a["success_rate"] = round(a["success"] / a["total"], 3) if a["total"] > 0 else 0

        return SkillResult(
            success=True,
            message=f"Stats: {total} delegations, {success_rate:.0%} success rate, "
                    f"{avg_efficiency:.1%} avg budget efficiency",
            data={
                "total": total,
                "successes": successes,
                "failures": failures,
                "success_rate": round(success_rate, 3),
                "avg_budget_efficiency": round(avg_efficiency, 3),
                "on_time_count": on_time_count,
                "on_time_rate": round(on_time_rate, 3),
                "agent_breakdown": agent_stats,
            },
        )

    # ── Data helpers ──────────────────────────────────────────

    def _load_delegations(self) -> List[Dict]:
        """Load delegation data from TaskDelegationSkill's file or via context."""
        # Try via SkillContext first
        if self.context:
            try:
                # Try to get delegation data through the registry
                registry = self.context._registry
                task_skill = registry.get("task_delegation")
                if task_skill:
                    state = task_skill._load()
                    return state.get("delegations", [])
            except Exception:
                pass

        # Fallback: read the file directly
        try:
            if DELEGATION_FILE.exists():
                with open(DELEGATION_FILE) as f:
                    data = json.load(f)
                return data.get("delegations", [])
        except Exception:
            pass

        return []

    def _load_state(self) -> Dict:
        """Load bridge state."""
        try:
            if BRIDGE_DATA_FILE.exists():
                with open(BRIDGE_DATA_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "config": dict(DEFAULT_CONFIG),
            "processed_ids": [],
            "history": [],
            "stats": {},
        }

    def _save_state(self):
        """Persist bridge state."""
        try:
            BRIDGE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(BRIDGE_DATA_FILE, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception:
            pass
