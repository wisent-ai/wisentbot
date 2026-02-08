#!/usr/bin/env python3
"""
CanaryDeploymentSkill - Safe deployment strategies with gradual rollout and auto-rollback.

The existing DeploymentSkill handles basic deploy/rollback/scale, but has no safe
deployment strategies for production services. This skill fills that gap with:

1. **Canary Deployments** (Revenue pillar)
   - Deploy new version to a small canary fleet (e.g., 5% of traffic)
   - Monitor canary health metrics (error rate, latency, success rate)
   - Gradually shift traffic: 5% → 25% → 50% → 100%
   - Automatic rollback if metrics degrade beyond thresholds

2. **Blue-Green Deployments** (Replication pillar)
   - Maintain two identical environments (blue/green)
   - Deploy to inactive environment, verify health
   - Instant traffic switch when ready
   - Keep old environment for instant rollback

3. **Deployment Guards** (Self-Improvement pillar)
   - Pre-deployment health checks
   - Post-deployment metric validation
   - Configurable rollback thresholds (error rate, latency)
   - Deployment approval gates

4. **Rollout History & Analytics** (Observability)
   - Track all rollout stages with timestamps
   - Record metrics at each stage
   - Compute deployment success rates
   - Learn from past deployment failures

Architecture:
  Rollout = {strategy, stages, current_stage, metrics_at_each_stage, auto_rollback_config}
  Each rollout progresses through stages, collecting metrics at each one.
  If metrics breach thresholds, rollback is triggered automatically.

Part of the Revenue + Replication pillars: protects revenue services from bad deploys.
"""

import json
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

CANARY_FILE = Path(__file__).parent.parent / "data" / "canary_deployments.json"
MAX_ROLLOUTS = 500

# Default canary stages (traffic percentage)
DEFAULT_CANARY_STAGES = [5, 25, 50, 100]

# Default blue-green stages
DEFAULT_BG_STAGES = ["deploy_inactive", "health_check", "switch_traffic", "verify"]

# Rollout statuses
ROLLOUT_STATUSES = ["pending", "in_progress", "paused", "completed", "rolled_back", "failed"]

# Default metric thresholds for auto-rollback
DEFAULT_THRESHOLDS = {
    "max_error_rate": 0.05,  # 5% error rate
    "max_latency_ms": 5000,  # 5 second p99
    "min_success_rate": 0.95,  # 95% success
}


class CanaryDeploymentSkill(Skill):
    """
    Safe deployment strategies with gradual rollout, health monitoring,
    and automatic rollback on metric degradation.

    Supports canary (gradual traffic shift) and blue-green (instant switch)
    deployment patterns with configurable thresholds and rollout stages.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        CANARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not CANARY_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "rollouts": [],
            "environments": {},
            "config": {
                "default_strategy": "canary",
                "default_thresholds": DEFAULT_THRESHOLDS.copy(),
                "canary_stages": DEFAULT_CANARY_STAGES[:],
            },
            "stats": {
                "total_rollouts": 0,
                "successful": 0,
                "rolled_back": 0,
                "failed": 0,
            },
        }

    def _load(self) -> Dict:
        try:
            with open(CANARY_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        with open(CANARY_FILE, "w") as f:
            json.dump(state, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="canary_deployment",
            name="Canary Deployment",
            version="1.0.0",
            category="infrastructure",
            description=(
                "Safe deployment strategies with canary rollouts, blue-green switching, "
                "health monitoring, and automatic rollback"
            ),
            actions=[
                SkillAction(
                    name="create_rollout",
                    description="Create a new deployment rollout with a chosen strategy",
                    parameters={
                        "service_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the service being deployed",
                        },
                        "version": {
                            "type": "string",
                            "required": True,
                            "description": "New version being deployed",
                        },
                        "strategy": {
                            "type": "string",
                            "required": False,
                            "description": "Deployment strategy: 'canary' or 'blue_green' (default: canary)",
                        },
                        "stages": {
                            "type": "array",
                            "required": False,
                            "description": (
                                "Custom traffic stages for canary (e.g., [5, 25, 50, 100]) "
                                "or stages for blue-green"
                            ),
                        },
                        "thresholds": {
                            "type": "object",
                            "required": False,
                            "description": (
                                "Auto-rollback thresholds: max_error_rate, max_latency_ms, "
                                "min_success_rate"
                            ),
                        },
                        "metadata": {
                            "type": "object",
                            "required": False,
                            "description": "Arbitrary metadata (commit hash, author, etc.)",
                        },
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                ),
                SkillAction(
                    name="advance",
                    description="Advance a rollout to the next stage",
                    parameters={
                        "rollout_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rollout to advance",
                        },
                        "metrics": {
                            "type": "object",
                            "required": False,
                            "description": (
                                "Current metrics: error_rate, latency_ms, success_rate, "
                                "request_count"
                            ),
                        },
                        "force": {
                            "type": "boolean",
                            "required": False,
                            "description": "Advance even if metrics breach thresholds",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="rollback",
                    description="Rollback a deployment to the previous version",
                    parameters={
                        "rollout_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rollout to rollback",
                        },
                        "reason": {
                            "type": "string",
                            "required": False,
                            "description": "Reason for rollback",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="check_health",
                    description="Check if current metrics pass rollout thresholds",
                    parameters={
                        "rollout_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rollout to check",
                        },
                        "metrics": {
                            "type": "object",
                            "required": True,
                            "description": (
                                "Current metrics: error_rate, latency_ms, success_rate"
                            ),
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="pause",
                    description="Pause a rollout at its current stage",
                    parameters={
                        "rollout_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rollout to pause",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="resume",
                    description="Resume a paused rollout",
                    parameters={
                        "rollout_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rollout to resume",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get the current status of a rollout",
                    parameters={
                        "rollout_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rollout",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_rollouts",
                    description="List rollouts with optional filters",
                    parameters={
                        "service_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by service ID",
                        },
                        "status": {
                            "type": "string",
                            "required": False,
                            "description": (
                                f"Filter by status: {', '.join(ROLLOUT_STATUSES)}"
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max results (default: 20)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="register_environment",
                    description="Register a deployment environment for blue-green switching",
                    parameters={
                        "env_name": {
                            "type": "string",
                            "required": True,
                            "description": "Environment name (e.g., 'blue', 'green')",
                        },
                        "service_id": {
                            "type": "string",
                            "required": True,
                            "description": "Service this environment belongs to",
                        },
                        "endpoint": {
                            "type": "string",
                            "required": False,
                            "description": "Endpoint URL for this environment",
                        },
                        "version": {
                            "type": "string",
                            "required": False,
                            "description": "Current version deployed",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="analytics",
                    description="Get deployment analytics: success rates, common failure reasons",
                    parameters={
                        "service_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter analytics to a specific service",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="View or update default deployment configuration",
                    parameters={
                        "default_strategy": {
                            "type": "string",
                            "required": False,
                            "description": "Default strategy: 'canary' or 'blue_green'",
                        },
                        "canary_stages": {
                            "type": "array",
                            "required": False,
                            "description": "Default canary stages (e.g., [5, 25, 50, 100])",
                        },
                        "default_thresholds": {
                            "type": "object",
                            "required": False,
                            "description": "Default auto-rollback thresholds",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
            author="adam",
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "create_rollout": self._create_rollout,
            "advance": self._advance,
            "rollback": self._rollback,
            "check_health": self._check_health,
            "pause": self._pause,
            "resume": self._resume,
            "status": self._status,
            "list_rollouts": self._list_rollouts,
            "register_environment": self._register_environment,
            "analytics": self._analytics,
            "configure": self._configure,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # --- Rollout Lifecycle ---

    def _create_rollout(self, params: Dict) -> SkillResult:
        """Create a new deployment rollout."""
        service_id = (params.get("service_id") or "").strip()
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        version = (params.get("version") or "").strip()
        if not version:
            return SkillResult(success=False, message="version is required")

        state = self._load()
        config = state.get("config", {})

        strategy = (params.get("strategy") or config.get("default_strategy", "canary")).strip()
        if strategy not in ("canary", "blue_green"):
            return SkillResult(
                success=False,
                message=f"Invalid strategy '{strategy}'. Use 'canary' or 'blue_green'",
            )

        # Determine stages
        if strategy == "canary":
            stages = params.get("stages") or config.get("canary_stages", DEFAULT_CANARY_STAGES)
            if isinstance(stages, str):
                try:
                    stages = json.loads(stages)
                except json.JSONDecodeError:
                    stages = DEFAULT_CANARY_STAGES[:]
            # Validate: must be ascending, end at 100
            if not stages or not all(isinstance(s, (int, float)) for s in stages):
                stages = DEFAULT_CANARY_STAGES[:]
            stages = sorted(set(int(s) for s in stages if 0 < s <= 100))
            if not stages or stages[-1] != 100:
                stages.append(100)
            stage_labels = [f"{s}%" for s in stages]
        else:
            stage_labels = DEFAULT_BG_STAGES[:]

        # Thresholds
        thresholds = params.get("thresholds") or config.get(
            "default_thresholds", DEFAULT_THRESHOLDS
        )
        if isinstance(thresholds, str):
            try:
                thresholds = json.loads(thresholds)
            except json.JSONDecodeError:
                thresholds = DEFAULT_THRESHOLDS.copy()

        metadata = params.get("metadata") or {}

        # Check for active rollout on same service
        active = [
            r for r in state["rollouts"]
            if r["service_id"] == service_id and r["status"] in ("in_progress", "paused")
        ]
        if active:
            return SkillResult(
                success=False,
                message=(
                    f"Service '{service_id}' already has an active rollout "
                    f"({active[0]['id']}). Rollback or complete it first."
                ),
                data={"active_rollout_id": active[0]["id"]},
            )

        rollout = {
            "id": str(uuid.uuid4())[:12],
            "service_id": service_id,
            "version": version,
            "strategy": strategy,
            "stages": stage_labels,
            "current_stage_index": 0,
            "status": "in_progress",
            "thresholds": thresholds,
            "metadata": metadata,
            "metrics_history": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None,
            "rollback_reason": None,
        }

        state["rollouts"].append(rollout)
        state["stats"]["total_rollouts"] += 1

        # Trim old rollouts
        if len(state["rollouts"]) > MAX_ROLLOUTS:
            state["rollouts"] = state["rollouts"][-MAX_ROLLOUTS:]

        self._save(state)

        return SkillResult(
            success=True,
            message=(
                f"Rollout created: {service_id} v{version} ({strategy}) "
                f"— starting at stage {stage_labels[0]}"
            ),
            data={
                "rollout_id": rollout["id"],
                "service_id": service_id,
                "version": version,
                "strategy": strategy,
                "stages": stage_labels,
                "current_stage": stage_labels[0],
                "thresholds": thresholds,
            },
        )

    def _find_rollout(self, state: Dict, rollout_id: str) -> Optional[Dict]:
        """Find a rollout by ID."""
        for r in state["rollouts"]:
            if r["id"] == rollout_id:
                return r
        return None

    def _check_metrics_against_thresholds(
        self, metrics: Dict, thresholds: Dict
    ) -> Dict:
        """Check if metrics breach any thresholds. Returns violations dict."""
        violations = {}

        error_rate = metrics.get("error_rate")
        max_error_rate = thresholds.get("max_error_rate")
        if error_rate is not None and max_error_rate is not None:
            try:
                if float(error_rate) > float(max_error_rate):
                    violations["error_rate"] = {
                        "current": float(error_rate),
                        "threshold": float(max_error_rate),
                    }
            except (ValueError, TypeError):
                pass

        latency = metrics.get("latency_ms")
        max_latency = thresholds.get("max_latency_ms")
        if latency is not None and max_latency is not None:
            try:
                if float(latency) > float(max_latency):
                    violations["latency_ms"] = {
                        "current": float(latency),
                        "threshold": float(max_latency),
                    }
            except (ValueError, TypeError):
                pass

        success_rate = metrics.get("success_rate")
        min_success = thresholds.get("min_success_rate")
        if success_rate is not None and min_success is not None:
            try:
                if float(success_rate) < float(min_success):
                    violations["success_rate"] = {
                        "current": float(success_rate),
                        "threshold": float(min_success),
                    }
            except (ValueError, TypeError):
                pass

        return violations

    def _advance(self, params: Dict) -> SkillResult:
        """Advance a rollout to the next stage."""
        rollout_id = (params.get("rollout_id") or "").strip()
        if not rollout_id:
            return SkillResult(success=False, message="rollout_id is required")

        state = self._load()
        rollout = self._find_rollout(state, rollout_id)
        if not rollout:
            return SkillResult(
                success=False, message=f"Rollout '{rollout_id}' not found"
            )

        if rollout["status"] not in ("in_progress",):
            return SkillResult(
                success=False,
                message=(
                    f"Cannot advance rollout in status '{rollout['status']}'. "
                    f"Must be 'in_progress'."
                ),
            )

        metrics = params.get("metrics") or {}
        force = params.get("force", False)
        if isinstance(force, str):
            force = force.lower() in ("true", "1", "yes")

        # Check metrics against thresholds
        if metrics and not force:
            violations = self._check_metrics_against_thresholds(
                metrics, rollout.get("thresholds", {})
            )
            if violations:
                # Auto-rollback
                rollout["status"] = "rolled_back"
                rollout["rollback_reason"] = (
                    f"Auto-rollback: metrics breached thresholds at stage "
                    f"{rollout['stages'][rollout['current_stage_index']]}"
                )
                rollout["updated_at"] = datetime.now().isoformat()
                rollout["completed_at"] = datetime.now().isoformat()
                rollout["metrics_history"].append({
                    "stage": rollout["stages"][rollout["current_stage_index"]],
                    "metrics": metrics,
                    "violations": violations,
                    "action": "auto_rollback",
                    "timestamp": time.time(),
                })
                state["stats"]["rolled_back"] += 1
                self._save(state)

                return SkillResult(
                    success=False,
                    message=(
                        f"Auto-rollback triggered! Metrics breached thresholds "
                        f"at stage {rollout['stages'][rollout['current_stage_index']]}"
                    ),
                    data={
                        "rollout_id": rollout_id,
                        "status": "rolled_back",
                        "violations": violations,
                        "stage": rollout["stages"][rollout["current_stage_index"]],
                    },
                )

        # Record metrics for this stage
        current_stage = rollout["stages"][rollout["current_stage_index"]]
        rollout["metrics_history"].append({
            "stage": current_stage,
            "metrics": metrics,
            "violations": {},
            "action": "advance",
            "timestamp": time.time(),
        })

        # Advance to next stage
        next_index = rollout["current_stage_index"] + 1
        if next_index >= len(rollout["stages"]):
            # Rollout complete!
            rollout["status"] = "completed"
            rollout["completed_at"] = datetime.now().isoformat()
            rollout["updated_at"] = datetime.now().isoformat()
            state["stats"]["successful"] += 1
            self._save(state)

            return SkillResult(
                success=True,
                message=(
                    f"Rollout complete! {rollout['service_id']} v{rollout['version']} "
                    f"fully deployed ({rollout['strategy']})"
                ),
                data={
                    "rollout_id": rollout_id,
                    "status": "completed",
                    "service_id": rollout["service_id"],
                    "version": rollout["version"],
                    "stages_completed": len(rollout["stages"]),
                },
            )

        rollout["current_stage_index"] = next_index
        rollout["updated_at"] = datetime.now().isoformat()
        self._save(state)

        next_stage = rollout["stages"][next_index]
        return SkillResult(
            success=True,
            message=(
                f"Advanced to stage {next_stage} "
                f"({next_index + 1}/{len(rollout['stages'])})"
            ),
            data={
                "rollout_id": rollout_id,
                "current_stage": next_stage,
                "previous_stage": current_stage,
                "progress": f"{next_index + 1}/{len(rollout['stages'])}",
                "status": "in_progress",
            },
        )

    def _rollback(self, params: Dict) -> SkillResult:
        """Manually rollback a deployment."""
        rollout_id = (params.get("rollout_id") or "").strip()
        if not rollout_id:
            return SkillResult(success=False, message="rollout_id is required")

        state = self._load()
        rollout = self._find_rollout(state, rollout_id)
        if not rollout:
            return SkillResult(
                success=False, message=f"Rollout '{rollout_id}' not found"
            )

        if rollout["status"] in ("completed", "rolled_back", "failed"):
            return SkillResult(
                success=False,
                message=f"Cannot rollback: rollout already '{rollout['status']}'",
            )

        reason = (params.get("reason") or "Manual rollback").strip()

        rollout["status"] = "rolled_back"
        rollout["rollback_reason"] = reason
        rollout["updated_at"] = datetime.now().isoformat()
        rollout["completed_at"] = datetime.now().isoformat()
        rollout["metrics_history"].append({
            "stage": rollout["stages"][rollout["current_stage_index"]],
            "metrics": {},
            "violations": {},
            "action": "manual_rollback",
            "timestamp": time.time(),
        })
        state["stats"]["rolled_back"] += 1
        self._save(state)

        return SkillResult(
            success=True,
            message=(
                f"Rolled back: {rollout['service_id']} v{rollout['version']} "
                f"— {reason}"
            ),
            data={
                "rollout_id": rollout_id,
                "service_id": rollout["service_id"],
                "version": rollout["version"],
                "stage_at_rollback": rollout["stages"][rollout["current_stage_index"]],
                "reason": reason,
            },
        )

    def _check_health(self, params: Dict) -> SkillResult:
        """Check if metrics pass rollout thresholds without advancing."""
        rollout_id = (params.get("rollout_id") or "").strip()
        if not rollout_id:
            return SkillResult(success=False, message="rollout_id is required")

        metrics = params.get("metrics")
        if not metrics or not isinstance(metrics, dict):
            return SkillResult(success=False, message="metrics (object) is required")

        state = self._load()
        rollout = self._find_rollout(state, rollout_id)
        if not rollout:
            return SkillResult(
                success=False, message=f"Rollout '{rollout_id}' not found"
            )

        violations = self._check_metrics_against_thresholds(
            metrics, rollout.get("thresholds", {})
        )

        healthy = len(violations) == 0
        return SkillResult(
            success=True,
            message="Health check passed" if healthy else f"Health check failed: {len(violations)} violations",
            data={
                "healthy": healthy,
                "violations": violations,
                "metrics": metrics,
                "thresholds": rollout.get("thresholds", {}),
                "current_stage": rollout["stages"][rollout["current_stage_index"]],
            },
        )

    def _pause(self, params: Dict) -> SkillResult:
        """Pause a rollout at current stage."""
        rollout_id = (params.get("rollout_id") or "").strip()
        if not rollout_id:
            return SkillResult(success=False, message="rollout_id is required")

        state = self._load()
        rollout = self._find_rollout(state, rollout_id)
        if not rollout:
            return SkillResult(
                success=False, message=f"Rollout '{rollout_id}' not found"
            )

        if rollout["status"] != "in_progress":
            return SkillResult(
                success=False,
                message=f"Cannot pause: rollout is '{rollout['status']}', not 'in_progress'",
            )

        rollout["status"] = "paused"
        rollout["updated_at"] = datetime.now().isoformat()
        self._save(state)

        return SkillResult(
            success=True,
            message=(
                f"Paused at stage {rollout['stages'][rollout['current_stage_index']]}"
            ),
            data={
                "rollout_id": rollout_id,
                "status": "paused",
                "current_stage": rollout["stages"][rollout["current_stage_index"]],
            },
        )

    def _resume(self, params: Dict) -> SkillResult:
        """Resume a paused rollout."""
        rollout_id = (params.get("rollout_id") or "").strip()
        if not rollout_id:
            return SkillResult(success=False, message="rollout_id is required")

        state = self._load()
        rollout = self._find_rollout(state, rollout_id)
        if not rollout:
            return SkillResult(
                success=False, message=f"Rollout '{rollout_id}' not found"
            )

        if rollout["status"] != "paused":
            return SkillResult(
                success=False,
                message=f"Cannot resume: rollout is '{rollout['status']}', not 'paused'",
            )

        rollout["status"] = "in_progress"
        rollout["updated_at"] = datetime.now().isoformat()
        self._save(state)

        return SkillResult(
            success=True,
            message=(
                f"Resumed at stage {rollout['stages'][rollout['current_stage_index']]}"
            ),
            data={
                "rollout_id": rollout_id,
                "status": "in_progress",
                "current_stage": rollout["stages"][rollout["current_stage_index"]],
            },
        )

    # --- Status & Listing ---

    def _status(self, params: Dict) -> SkillResult:
        """Get detailed status of a rollout."""
        rollout_id = (params.get("rollout_id") or "").strip()
        if not rollout_id:
            return SkillResult(success=False, message="rollout_id is required")

        state = self._load()
        rollout = self._find_rollout(state, rollout_id)
        if not rollout:
            return SkillResult(
                success=False, message=f"Rollout '{rollout_id}' not found"
            )

        current_stage = rollout["stages"][rollout["current_stage_index"]]
        progress = f"{rollout['current_stage_index'] + 1}/{len(rollout['stages'])}"

        return SkillResult(
            success=True,
            message=(
                f"{rollout['service_id']} v{rollout['version']} — "
                f"{rollout['status']} at {current_stage} ({progress})"
            ),
            data={
                "rollout_id": rollout["id"],
                "service_id": rollout["service_id"],
                "version": rollout["version"],
                "strategy": rollout["strategy"],
                "status": rollout["status"],
                "current_stage": current_stage,
                "stages": rollout["stages"],
                "progress": progress,
                "thresholds": rollout.get("thresholds", {}),
                "metrics_history": rollout.get("metrics_history", []),
                "metadata": rollout.get("metadata", {}),
                "created_at": rollout["created_at"],
                "updated_at": rollout["updated_at"],
                "completed_at": rollout.get("completed_at"),
                "rollback_reason": rollout.get("rollback_reason"),
            },
        )

    def _list_rollouts(self, params: Dict) -> SkillResult:
        """List rollouts with filters."""
        state = self._load()
        rollouts = state["rollouts"]

        service_id = (params.get("service_id") or "").strip()
        if service_id:
            rollouts = [r for r in rollouts if r["service_id"] == service_id]

        status_filter = (params.get("status") or "").strip()
        if status_filter:
            rollouts = [r for r in rollouts if r["status"] == status_filter]

        limit = max(1, min(100, int(params.get("limit", 20))))

        # Most recent first
        rollouts = rollouts[::-1][:limit]

        summaries = []
        for r in rollouts:
            summaries.append({
                "id": r["id"],
                "service_id": r["service_id"],
                "version": r["version"],
                "strategy": r["strategy"],
                "status": r["status"],
                "current_stage": r["stages"][r["current_stage_index"]],
                "progress": f"{r['current_stage_index'] + 1}/{len(r['stages'])}",
                "created_at": r["created_at"],
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} rollouts",
            data={
                "rollouts": summaries,
                "total": len(state["rollouts"]),
            },
        )

    # --- Environments ---

    def _register_environment(self, params: Dict) -> SkillResult:
        """Register a deployment environment for blue-green switching."""
        env_name = (params.get("env_name") or "").strip()
        if not env_name:
            return SkillResult(success=False, message="env_name is required")

        service_id = (params.get("service_id") or "").strip()
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        state = self._load()
        key = f"{service_id}:{env_name}"
        state["environments"][key] = {
            "env_name": env_name,
            "service_id": service_id,
            "endpoint": (params.get("endpoint") or "").strip(),
            "version": (params.get("version") or "").strip(),
            "registered_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Environment '{env_name}' registered for service '{service_id}'",
            data={
                "env_name": env_name,
                "service_id": service_id,
                "key": key,
            },
        )

    # --- Analytics ---

    def _analytics(self, params: Dict) -> SkillResult:
        """Deployment analytics: success rates, failure patterns."""
        state = self._load()
        rollouts = state["rollouts"]

        service_id = (params.get("service_id") or "").strip()
        if service_id:
            rollouts = [r for r in rollouts if r["service_id"] == service_id]

        total = len(rollouts)
        completed = [r for r in rollouts if r["status"] == "completed"]
        rolled_back = [r for r in rollouts if r["status"] == "rolled_back"]
        failed = [r for r in rollouts if r["status"] == "failed"]

        success_rate = len(completed) / total if total > 0 else 0.0

        # Rollback reasons
        reason_counts = defaultdict(int)
        for r in rolled_back:
            reason = r.get("rollback_reason") or "Unknown"
            # Truncate for grouping
            key = reason[:60]
            reason_counts[key] += 1
        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Rollback stage distribution
        rollback_stages = defaultdict(int)
        for r in rolled_back:
            stage = r["stages"][r["current_stage_index"]]
            rollback_stages[stage] += 1

        # Strategy breakdown
        strategy_stats = defaultdict(lambda: {"total": 0, "completed": 0, "rolled_back": 0})
        for r in rollouts:
            s = r["strategy"]
            strategy_stats[s]["total"] += 1
            if r["status"] == "completed":
                strategy_stats[s]["completed"] += 1
            elif r["status"] == "rolled_back":
                strategy_stats[s]["rolled_back"] += 1

        # Per-service breakdown
        service_stats = defaultdict(lambda: {"total": 0, "completed": 0, "rolled_back": 0})
        for r in rollouts:
            sid = r["service_id"]
            service_stats[sid]["total"] += 1
            if r["status"] == "completed":
                service_stats[sid]["completed"] += 1
            elif r["status"] == "rolled_back":
                service_stats[sid]["rolled_back"] += 1

        return SkillResult(
            success=True,
            message=f"Analytics: {total} rollouts, {success_rate:.0%} success rate",
            data={
                "total_rollouts": total,
                "completed": len(completed),
                "rolled_back": len(rolled_back),
                "failed": len(failed),
                "success_rate": round(success_rate, 4),
                "top_rollback_reasons": [
                    {"reason": r, "count": c} for r, c in top_reasons
                ],
                "rollback_stage_distribution": dict(rollback_stages),
                "strategy_breakdown": dict(strategy_stats),
                "service_breakdown": dict(service_stats),
            },
        )

    # --- Configuration ---

    def _configure(self, params: Dict) -> SkillResult:
        """View or update deployment configuration."""
        state = self._load()
        config = state.get("config", {})
        changed = False

        strategy = params.get("default_strategy")
        if strategy:
            strategy = strategy.strip()
            if strategy not in ("canary", "blue_green"):
                return SkillResult(
                    success=False,
                    message="default_strategy must be 'canary' or 'blue_green'",
                )
            config["default_strategy"] = strategy
            changed = True

        stages = params.get("canary_stages")
        if stages is not None:
            if isinstance(stages, str):
                try:
                    stages = json.loads(stages)
                except json.JSONDecodeError:
                    return SkillResult(
                        success=False,
                        message="canary_stages must be a JSON array of numbers",
                    )
            if isinstance(stages, list) and all(
                isinstance(s, (int, float)) for s in stages
            ):
                stages = sorted(set(int(s) for s in stages if 0 < s <= 100))
                if stages and stages[-1] != 100:
                    stages.append(100)
                config["canary_stages"] = stages
                changed = True
            else:
                return SkillResult(
                    success=False,
                    message="canary_stages must be a list of numbers (1-100)",
                )

        thresholds = params.get("default_thresholds")
        if thresholds is not None:
            if isinstance(thresholds, dict):
                config["default_thresholds"] = thresholds
                changed = True
            else:
                return SkillResult(
                    success=False,
                    message="default_thresholds must be an object",
                )

        state["config"] = config

        if changed:
            self._save(state)
            msg = "Configuration updated"
        else:
            msg = "Current configuration"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "config": config,
                "stats": state.get("stats", {}),
                "environments": len(state.get("environments", {})),
            },
        )
