#!/usr/bin/env python3
"""
EventDrivenWorkflowSkill - Connect external events to autonomous agent workflows.

This is the critical bridge between WebhookSkill (external triggers) and
AutonomousLoopSkill (autonomous execution). External events from GitHub,
Stripe, customer requests, or any webhook source can now trigger pre-configured
multi-step workflows that execute autonomously.

Features:
- Register workflow templates: named sequences of skill actions
- Bind workflows to event patterns (webhook events, EventBus topics)
- Conditional execution with payload-based routing rules
- Workflow execution with inter-step data passing
- Execution history with success/failure tracking
- Retry logic with configurable backoff
- Parallel and sequential step execution modes
- Workflow templates are composable (a workflow can trigger another)

Example flows:
1. GitHub push webhook → code_review → summarize results → notify via messaging
2. Stripe payment → usage_tracking:register → send welcome email
3. New issue filed → goal_manager:create → planner:plan → notify agent

Pillars: Goal Setting (autonomous execution), Revenue (customer-triggered services),
         Self-Improvement (event-driven learning)
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from .base import Skill, SkillManifest, SkillAction, SkillResult


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    step_id: str
    name: str
    skill_id: str
    action: str
    # Static params always passed to this step
    params: Dict[str, Any] = field(default_factory=dict)
    # Map outputs from previous steps into this step's params
    # Format: {"prev_step_id.data.field": "param_name"}
    input_mapping: Dict[str, str] = field(default_factory=dict)
    # Map trigger event payload fields into this step's params
    # Format: {"payload.field.path": "param_name"}
    event_mapping: Dict[str, str] = field(default_factory=dict)
    # Condition: only run if this evaluates true
    # Format: {"field": "value"} checked against trigger payload
    condition: Dict[str, Any] = field(default_factory=dict)
    # Retry config
    max_retries: int = 0
    retry_delay_seconds: float = 1.0
    # Continue workflow even if this step fails
    continue_on_failure: bool = False
    # Timeout in seconds (0 = no timeout)
    timeout_seconds: float = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkflowTemplate:
    """A reusable workflow template."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    # Event bindings: list of patterns that trigger this workflow
    # Each binding has: {"source": "webhook|event_bus", "pattern": "github-push|payment.*"}
    event_bindings: List[Dict[str, str]] = field(default_factory=list)
    # Global settings
    enabled: bool = True
    max_concurrent_runs: int = 5
    created_at: str = ""
    updated_at: str = ""
    # Stats
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    last_run_at: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


@dataclass
class WorkflowRun:
    """Record of a single workflow execution."""
    run_id: str
    workflow_id: str
    workflow_name: str
    trigger_source: str  # "webhook", "event_bus", "manual"
    trigger_event: str   # Event name/pattern that triggered it
    trigger_payload: Dict[str, Any]
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"  # running, completed, failed, cancelled
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    total_duration_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class EventDrivenWorkflowSkill(Skill):
    """Skill for creating and executing event-triggered autonomous workflows."""

    def __init__(self, credentials: Dict[str, str] = None, data_dir: str = None):
        super().__init__(credentials)
        self._data_dir = Path(data_dir) if data_dir else Path("singularity/data")
        self._workflows: Dict[str, WorkflowTemplate] = {}
        self._runs: List[WorkflowRun] = []
        self._max_runs = 500
        self._active_runs: Dict[str, int] = {}  # workflow_id -> count of active runs
        self._load_data()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="event_driven_workflow",
            name="Event-Driven Workflows",
            version="1.0.0",
            category="autonomy",
            description=(
                "Connect external events (webhooks, EventBus topics) to autonomous "
                "multi-step workflows. Register workflow templates with skill action "
                "sequences, bind them to event patterns, and let the agent execute "
                "them autonomously when events arrive."
            ),
            actions=[
                SkillAction(
                    name="create_workflow",
                    description=(
                        "Create a new workflow template with a sequence of skill actions. "
                        "Each step specifies a skill_id and action, with optional data mapping "
                        "from the trigger event and previous step outputs."
                    ),
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Unique workflow name"},
                        "description": {"type": "string", "required": False,
                                        "description": "What this workflow does"},
                        "steps": {"type": "array", "required": True,
                                  "description": "List of step objects: [{name, skill_id, action, params, event_mapping, input_mapping, condition, max_retries, continue_on_failure}]"},
                        "event_bindings": {"type": "array", "required": False,
                                           "description": "Event triggers: [{source: 'webhook'|'event_bus', pattern: 'name-or-pattern'}]"},
                        "max_concurrent_runs": {"type": "integer", "required": False,
                                                "description": "Max simultaneous runs (default 5)"},
                    },
                ),
                SkillAction(
                    name="trigger",
                    description=(
                        "Manually trigger a workflow by name, or process an event that "
                        "may match one or more workflow bindings."
                    ),
                    parameters={
                        "workflow_name": {"type": "string", "required": False,
                                          "description": "Specific workflow to trigger (by name)"},
                        "event_source": {"type": "string", "required": False,
                                          "description": "Event source type: webhook, event_bus, manual"},
                        "event_name": {"type": "string", "required": False,
                                        "description": "Event name/topic for pattern matching"},
                        "payload": {"type": "object", "required": False,
                                    "description": "Event payload to pass to the workflow"},
                    },
                ),
                SkillAction(
                    name="list_workflows",
                    description="List all registered workflow templates.",
                    parameters={
                        "enabled_only": {"type": "boolean", "required": False,
                                         "description": "Only show enabled workflows"},
                    },
                ),
                SkillAction(
                    name="get_workflow",
                    description="Get full details of a specific workflow template.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Workflow name"},
                    },
                ),
                SkillAction(
                    name="update_workflow",
                    description="Update a workflow's settings (enable/disable, bindings, etc.).",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Workflow name to update"},
                        "enabled": {"type": "boolean", "required": False,
                                    "description": "Enable/disable the workflow"},
                        "event_bindings": {"type": "array", "required": False,
                                           "description": "New event bindings"},
                        "max_concurrent_runs": {"type": "integer", "required": False,
                                                "description": "New concurrency limit"},
                        "description": {"type": "string", "required": False,
                                        "description": "Updated description"},
                    },
                ),
                SkillAction(
                    name="delete_workflow",
                    description="Delete a workflow template.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Workflow name to delete"},
                    },
                ),
                SkillAction(
                    name="get_runs",
                    description="View workflow execution history.",
                    parameters={
                        "workflow_name": {"type": "string", "required": False,
                                          "description": "Filter by workflow name"},
                        "status": {"type": "string", "required": False,
                                   "description": "Filter by status: running, completed, failed"},
                        "limit": {"type": "integer", "required": False,
                                  "description": "Max results (default 20)"},
                    },
                ),
                SkillAction(
                    name="get_run",
                    description="Get detailed results of a specific workflow run.",
                    parameters={
                        "run_id": {"type": "string", "required": True,
                                   "description": "Run ID to look up"},
                    },
                ),
                SkillAction(
                    name="bind_webhook",
                    description=(
                        "Convenience action: register a webhook endpoint AND bind it "
                        "to a workflow in one step. Creates the webhook via WebhookSkill "
                        "and adds an event binding to the workflow."
                    ),
                    parameters={
                        "webhook_name": {"type": "string", "required": True,
                                          "description": "Name for the webhook endpoint"},
                        "workflow_name": {"type": "string", "required": True,
                                          "description": "Workflow to trigger"},
                        "description": {"type": "string", "required": False,
                                        "description": "Description of this binding"},
                        "secret": {"type": "string", "required": False,
                                   "description": "HMAC secret for webhook verification"},
                        "filters": {"type": "object", "required": False,
                                    "description": "Payload filters for the webhook"},
                    },
                ),
                SkillAction(
                    name="stats",
                    description="Get aggregate statistics across all workflows.",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "create_workflow": self._create_workflow,
            "trigger": self._trigger,
            "list_workflows": self._list_workflows,
            "get_workflow": self._get_workflow,
            "update_workflow": self._update_workflow,
            "delete_workflow": self._delete_workflow,
            "get_runs": self._get_runs,
            "get_run": self._get_run,
            "bind_webhook": self._bind_webhook,
            "stats": self._stats,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        return await handler(params)

    # --- Core Actions ---

    async def _create_workflow(self, params: Dict) -> SkillResult:
        """Create a new workflow template."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Workflow name is required")

        # Check for duplicate name
        for wf in self._workflows.values():
            if wf.name == name:
                return SkillResult(
                    success=False,
                    message=f"Workflow '{name}' already exists. Use update_workflow or delete first."
                )

        steps_data = params.get("steps", [])
        if not steps_data:
            return SkillResult(success=False, message="At least one step is required")

        # Parse steps
        steps = []
        for i, step_data in enumerate(steps_data):
            if not step_data.get("skill_id") or not step_data.get("action"):
                return SkillResult(
                    success=False,
                    message=f"Step {i+1} must have 'skill_id' and 'action'"
                )
            step = WorkflowStep(
                step_id=step_data.get("step_id", f"step_{i+1}"),
                name=step_data.get("name", f"Step {i+1}"),
                skill_id=step_data["skill_id"],
                action=step_data["action"],
                params=step_data.get("params", {}),
                input_mapping=step_data.get("input_mapping", {}),
                event_mapping=step_data.get("event_mapping", {}),
                condition=step_data.get("condition", {}),
                max_retries=step_data.get("max_retries", 0),
                retry_delay_seconds=step_data.get("retry_delay_seconds", 1.0),
                continue_on_failure=step_data.get("continue_on_failure", False),
                timeout_seconds=step_data.get("timeout_seconds", 0),
            )
            steps.append(step)

        now = datetime.utcnow().isoformat()
        workflow = WorkflowTemplate(
            workflow_id=str(uuid.uuid4()),
            name=name,
            description=params.get("description", ""),
            steps=steps,
            event_bindings=params.get("event_bindings", []),
            enabled=True,
            max_concurrent_runs=params.get("max_concurrent_runs", 5),
            created_at=now,
            updated_at=now,
        )

        self._workflows[workflow.workflow_id] = workflow
        self._save_data()

        return SkillResult(
            success=True,
            message=f"Workflow '{name}' created with {len(steps)} step(s) and {len(workflow.event_bindings)} event binding(s)",
            data={
                "workflow_id": workflow.workflow_id,
                "name": name,
                "steps_count": len(steps),
                "event_bindings": workflow.event_bindings,
            }
        )

    async def _trigger(self, params: Dict) -> SkillResult:
        """Trigger workflow(s) by name or by event matching."""
        workflow_name = params.get("workflow_name")
        event_source = params.get("event_source", "manual")
        event_name = params.get("event_name", "")
        payload = params.get("payload", {})

        workflows_to_run = []

        if workflow_name:
            # Direct trigger by name
            wf = self._find_workflow_by_name(workflow_name)
            if not wf:
                return SkillResult(
                    success=False,
                    message=f"Workflow '{workflow_name}' not found"
                )
            if not wf.enabled:
                return SkillResult(
                    success=False,
                    message=f"Workflow '{workflow_name}' is disabled"
                )
            workflows_to_run.append(wf)
        elif event_name:
            # Match event against all workflow bindings
            workflows_to_run = self._match_event(event_source, event_name)
            if not workflows_to_run:
                return SkillResult(
                    success=True,
                    message=f"No workflows matched event '{event_source}:{event_name}'",
                    data={"matched": 0, "event_source": event_source, "event_name": event_name}
                )
        else:
            return SkillResult(
                success=False,
                message="Provide either 'workflow_name' for direct trigger or 'event_name' for pattern matching"
            )

        # Execute matched workflows
        results = []
        for wf in workflows_to_run:
            # Check concurrency limit
            active = self._active_runs.get(wf.workflow_id, 0)
            if active >= wf.max_concurrent_runs:
                results.append({
                    "workflow": wf.name,
                    "status": "skipped",
                    "reason": f"Concurrency limit reached ({active}/{wf.max_concurrent_runs})",
                })
                continue

            run_result = await self._execute_workflow(wf, event_source, event_name, payload)
            results.append(run_result)

        total = len(results)
        succeeded = sum(1 for r in results if r.get("status") == "completed")

        return SkillResult(
            success=succeeded > 0 or total == 0,
            message=f"Triggered {total} workflow(s): {succeeded} completed, {total - succeeded} other",
            data={
                "trigger_source": event_source,
                "trigger_event": event_name,
                "workflows_triggered": total,
                "workflows_succeeded": succeeded,
                "results": results,
            }
        )

    async def _execute_workflow(
        self, workflow: WorkflowTemplate, trigger_source: str,
        trigger_event: str, payload: Dict
    ) -> Dict:
        """Execute a single workflow and return the run summary."""
        run_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()

        run = WorkflowRun(
            run_id=run_id,
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            trigger_source=trigger_source,
            trigger_event=trigger_event,
            trigger_payload=payload,
            started_at=now,
        )

        # Track active runs
        self._active_runs[workflow.workflow_id] = self._active_runs.get(workflow.workflow_id, 0) + 1

        start_time = time.monotonic()
        step_outputs: Dict[str, Any] = {}  # step_id -> output data
        all_succeeded = True

        for step in workflow.steps:
            step_start = time.monotonic()

            # Check condition
            if step.condition and not self._check_condition(payload, step.condition):
                run.step_results.append({
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": "skipped",
                    "reason": "Condition not met",
                    "duration_ms": 0,
                })
                continue

            # Build params from static, event mapping, and input mapping
            step_params = dict(step.params)
            step_params.update(self._apply_event_mapping(payload, step.event_mapping))
            step_params.update(self._apply_input_mapping(step_outputs, step.input_mapping))

            # Execute with retries
            result = None
            attempts = 0
            max_attempts = step.max_retries + 1

            while attempts < max_attempts:
                attempts += 1
                result = await self._execute_step(step.skill_id, step.action, step_params)

                if result.success:
                    break

                if attempts < max_attempts:
                    # Wait before retry (simple linear backoff)
                    import asyncio
                    await asyncio.sleep(step.retry_delay_seconds * attempts)

            step_duration = (time.monotonic() - step_start) * 1000

            step_record = {
                "step_id": step.step_id,
                "name": step.name,
                "skill_id": step.skill_id,
                "action": step.action,
                "params_used": step_params,
                "success": result.success if result else False,
                "message": result.message[:200] if result else "No result",
                "attempts": attempts,
                "duration_ms": round(step_duration, 2),
            }

            if result and result.success:
                step_outputs[step.step_id] = result.data or {}
                step_record["output_keys"] = list((result.data or {}).keys())
            else:
                all_succeeded = False
                if not step.continue_on_failure:
                    step_record["status"] = "failed"
                    run.step_results.append(step_record)
                    break

            step_record["status"] = "completed" if (result and result.success) else "failed_continue"
            run.step_results.append(step_record)

        # Finalize run
        total_duration = (time.monotonic() - start_time) * 1000
        run.completed_at = datetime.utcnow().isoformat()
        run.status = "completed" if all_succeeded else "failed"
        run.total_duration_ms = round(total_duration, 2)

        # Update workflow stats
        workflow.total_runs += 1
        if all_succeeded:
            workflow.successful_runs += 1
        else:
            workflow.failed_runs += 1
        workflow.last_run_at = run.completed_at
        workflow.updated_at = run.completed_at

        # Track active runs
        self._active_runs[workflow.workflow_id] = max(
            0, self._active_runs.get(workflow.workflow_id, 1) - 1
        )

        # Record and persist
        self._record_run(run)
        self._save_data()

        # Publish event about workflow completion if EventBus available
        if self.context:
            try:
                await self.context.call_skill("event", "publish", {
                    "topic": f"workflow.{'completed' if all_succeeded else 'failed'}",
                    "data": {
                        "workflow_name": workflow.name,
                        "run_id": run_id,
                        "trigger_event": trigger_event,
                        "success": all_succeeded,
                        "steps_executed": len(run.step_results),
                        "duration_ms": run.total_duration_ms,
                    },
                    "source": "event_driven_workflow",
                })
            except Exception:
                pass  # Event publishing is best-effort

        return {
            "workflow": workflow.name,
            "run_id": run_id,
            "status": run.status,
            "steps_executed": len(run.step_results),
            "duration_ms": run.total_duration_ms,
        }

    async def _execute_step(self, skill_id: str, action: str, params: Dict) -> SkillResult:
        """Execute a single step via SkillContext."""
        if self.context:
            try:
                return await self.context.call_skill(skill_id, action, params)
            except Exception as e:
                return SkillResult(
                    success=False,
                    message=f"Error executing {skill_id}:{action}: {str(e)[:150]}"
                )
        else:
            # No context - dry run
            return SkillResult(
                success=True,
                message=f"Dry run: {skill_id}:{action}",
                data={"params": params, "dry_run": True}
            )

    # --- Query Actions ---

    async def _list_workflows(self, params: Dict) -> SkillResult:
        """List all workflow templates."""
        enabled_only = params.get("enabled_only", False)
        workflows = list(self._workflows.values())
        if enabled_only:
            workflows = [wf for wf in workflows if wf.enabled]

        summaries = []
        for wf in workflows:
            summaries.append({
                "name": wf.name,
                "description": wf.description[:100] if wf.description else "",
                "steps_count": len(wf.steps),
                "event_bindings": len(wf.event_bindings),
                "enabled": wf.enabled,
                "total_runs": wf.total_runs,
                "success_rate": (
                    round(wf.successful_runs / wf.total_runs, 2)
                    if wf.total_runs > 0 else None
                ),
                "last_run_at": wf.last_run_at,
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} workflow(s)",
            data={"workflows": summaries, "total": len(summaries)}
        )

    async def _get_workflow(self, params: Dict) -> SkillResult:
        """Get full details of a specific workflow."""
        name = params.get("name", "").strip()
        wf = self._find_workflow_by_name(name)
        if not wf:
            return SkillResult(success=False, message=f"Workflow '{name}' not found")

        data = wf.to_dict()
        # Include recent runs
        recent_runs = [
            r.to_dict() for r in self._runs
            if r.workflow_name == name
        ][-5:]
        data["recent_runs"] = recent_runs

        return SkillResult(success=True, message=f"Workflow '{name}' details", data=data)

    async def _update_workflow(self, params: Dict) -> SkillResult:
        """Update a workflow template."""
        name = params.get("name", "").strip()
        wf = self._find_workflow_by_name(name)
        if not wf:
            return SkillResult(success=False, message=f"Workflow '{name}' not found")

        updated = []
        if "enabled" in params:
            wf.enabled = params["enabled"]
            updated.append(f"enabled={wf.enabled}")
        if "event_bindings" in params:
            wf.event_bindings = params["event_bindings"]
            updated.append(f"event_bindings ({len(wf.event_bindings)})")
        if "max_concurrent_runs" in params:
            wf.max_concurrent_runs = params["max_concurrent_runs"]
            updated.append(f"max_concurrent_runs={wf.max_concurrent_runs}")
        if "description" in params:
            wf.description = params["description"]
            updated.append("description")

        wf.updated_at = datetime.utcnow().isoformat()
        self._save_data()

        return SkillResult(
            success=True,
            message=f"Updated workflow '{name}': {', '.join(updated) if updated else 'no changes'}",
            data={"name": name, "updated_fields": updated}
        )

    async def _delete_workflow(self, params: Dict) -> SkillResult:
        """Delete a workflow template."""
        name = params.get("name", "").strip()
        wf = self._find_workflow_by_name(name)
        if not wf:
            return SkillResult(success=False, message=f"Workflow '{name}' not found")

        del self._workflows[wf.workflow_id]
        self._save_data()

        return SkillResult(
            success=True,
            message=f"Deleted workflow '{name}'",
            data={"deleted_workflow_id": wf.workflow_id, "name": name}
        )

    async def _get_runs(self, params: Dict) -> SkillResult:
        """View workflow execution history."""
        workflow_name = params.get("workflow_name")
        status_filter = params.get("status")
        limit = params.get("limit", 20)

        runs = list(self._runs)
        if workflow_name:
            runs = [r for r in runs if r.workflow_name == workflow_name]
        if status_filter:
            runs = [r for r in runs if r.status == status_filter]

        # Most recent first
        runs = runs[-limit:]
        runs.reverse()

        return SkillResult(
            success=True,
            message=f"Found {len(runs)} workflow run(s)",
            data={
                "runs": [r.to_dict() for r in runs],
                "total": len(runs),
            }
        )

    async def _get_run(self, params: Dict) -> SkillResult:
        """Get details of a specific run."""
        run_id = params.get("run_id", "").strip()
        for r in self._runs:
            if r.run_id == run_id:
                return SkillResult(
                    success=True,
                    message=f"Run {run_id}: {r.status}",
                    data=r.to_dict()
                )
        return SkillResult(success=False, message=f"Run '{run_id}' not found")

    async def _bind_webhook(self, params: Dict) -> SkillResult:
        """Register a webhook and bind it to a workflow."""
        webhook_name = params.get("webhook_name", "").strip()
        workflow_name = params.get("workflow_name", "").strip()

        if not webhook_name or not workflow_name:
            return SkillResult(
                success=False,
                message="Both webhook_name and workflow_name are required"
            )

        # Verify workflow exists
        wf = self._find_workflow_by_name(workflow_name)
        if not wf:
            return SkillResult(
                success=False,
                message=f"Workflow '{workflow_name}' not found. Create it first."
            )

        # Register webhook pointing to this skill's trigger action
        webhook_result = None
        if self.context:
            webhook_result = await self.context.call_skill("webhook", "register", {
                "name": webhook_name,
                "description": params.get("description", f"Triggers workflow: {workflow_name}"),
                "target_skill_id": "event_driven_workflow",
                "target_action": "trigger",
                "secret": params.get("secret"),
                "filters": params.get("filters", {}),
                "field_mapping": {},
                "static_params": {
                    "event_source": "webhook",
                    "event_name": webhook_name,
                    "workflow_name": workflow_name,
                },
            })
        else:
            webhook_result = SkillResult(
                success=True,
                message=f"Dry run: would register webhook '{webhook_name}'",
                data={"dry_run": True}
            )

        if not webhook_result.success:
            return SkillResult(
                success=False,
                message=f"Failed to register webhook: {webhook_result.message}"
            )

        # Add event binding to workflow
        binding = {"source": "webhook", "pattern": webhook_name}
        if binding not in wf.event_bindings:
            wf.event_bindings.append(binding)
            wf.updated_at = datetime.utcnow().isoformat()
            self._save_data()

        return SkillResult(
            success=True,
            message=f"Webhook '{webhook_name}' → workflow '{workflow_name}' bound successfully. "
                    f"URL path: /webhooks/{webhook_name}",
            data={
                "webhook_name": webhook_name,
                "workflow_name": workflow_name,
                "url_path": f"/webhooks/{webhook_name}",
                "webhook_result": webhook_result.data,
            }
        )

    async def _stats(self, params: Dict) -> SkillResult:
        """Get aggregate statistics."""
        total_workflows = len(self._workflows)
        enabled_workflows = sum(1 for wf in self._workflows.values() if wf.enabled)
        total_runs = len(self._runs)
        successful_runs = sum(1 for r in self._runs if r.status == "completed")
        failed_runs = sum(1 for r in self._runs if r.status == "failed")

        total_bindings = sum(len(wf.event_bindings) for wf in self._workflows.values())

        avg_duration = 0
        durations = [r.total_duration_ms for r in self._runs if r.total_duration_ms]
        if durations:
            avg_duration = round(sum(durations) / len(durations), 2)

        # Most triggered workflows
        trigger_counts = {}
        for r in self._runs:
            trigger_counts[r.workflow_name] = trigger_counts.get(r.workflow_name, 0) + 1
        most_triggered = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return SkillResult(
            success=True,
            message=f"{total_workflows} workflows ({enabled_workflows} enabled), "
                    f"{total_runs} total runs ({successful_runs} ok, {failed_runs} failed)",
            data={
                "total_workflows": total_workflows,
                "enabled_workflows": enabled_workflows,
                "total_event_bindings": total_bindings,
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": round(successful_runs / total_runs, 2) if total_runs > 0 else None,
                "avg_duration_ms": avg_duration,
                "most_triggered": [{"workflow": name, "count": count} for name, count in most_triggered],
            }
        )

    # --- Helper Methods ---

    def _find_workflow_by_name(self, name: str) -> Optional[WorkflowTemplate]:
        """Find a workflow by name."""
        for wf in self._workflows.values():
            if wf.name == name:
                return wf
        return None

    def _match_event(self, source: str, event_name: str) -> List[WorkflowTemplate]:
        """Find all enabled workflows whose bindings match the given event."""
        matched = []
        for wf in self._workflows.values():
            if not wf.enabled:
                continue
            for binding in wf.event_bindings:
                b_source = binding.get("source", "")
                b_pattern = binding.get("pattern", "")
                if self._pattern_matches(b_source, source, b_pattern, event_name):
                    matched.append(wf)
                    break  # Don't add same workflow twice
        return matched

    def _pattern_matches(self, bind_source: str, event_source: str,
                         bind_pattern: str, event_name: str) -> bool:
        """Check if an event binding pattern matches the incoming event."""
        # Source must match (or be wildcard)
        if bind_source != "*" and bind_source != event_source:
            return False

        # Pattern matching with wildcards
        if bind_pattern == "*":
            return True
        if bind_pattern == event_name:
            return True

        # Simple wildcard: "payment.*" matches "payment.received"
        if "*" in bind_pattern:
            parts = bind_pattern.split("*")
            if len(parts) == 2:
                prefix, suffix = parts
                if event_name.startswith(prefix) and event_name.endswith(suffix):
                    return True

        return False

    def _check_condition(self, payload: Dict, condition: Dict) -> bool:
        """Check if payload meets the condition criteria."""
        for path, expected in condition.items():
            actual = self._extract_field(payload, path)
            if actual != expected:
                return False
        return True

    def _apply_event_mapping(self, payload: Dict, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map fields from the trigger payload to step params."""
        result = {}
        for source_path, target_param in mapping.items():
            value = self._extract_field(payload, source_path)
            if value is not None:
                result[target_param] = value
        return result

    def _apply_input_mapping(self, step_outputs: Dict[str, Any],
                             mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map outputs from previous steps to this step's params."""
        result = {}
        for source_path, target_param in mapping.items():
            # source_path format: "step_id.field.path"
            parts = source_path.split(".", 1)
            if len(parts) == 2:
                step_id, field_path = parts
                step_data = step_outputs.get(step_id, {})
                value = self._extract_field(step_data, field_path)
                if value is not None:
                    result[target_param] = value
        return result

    def _extract_field(self, data: Any, path: str) -> Any:
        """Extract a nested field using dot notation."""
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _record_run(self, run: WorkflowRun):
        """Record a run and trim history."""
        self._runs.append(run)
        if len(self._runs) > self._max_runs:
            self._runs = self._runs[-self._max_runs:]

    # --- Persistence ---

    def _get_data_path(self) -> Path:
        return self._data_dir / "event_workflows.json"

    def _save_data(self):
        """Save workflows to disk."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "workflows": {
                    wid: wf.to_dict() for wid, wf in self._workflows.items()
                },
                "runs": [r.to_dict() for r in self._runs[-100:]],  # Keep last 100 runs on disk
            }
            with open(self._get_data_path(), "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_data(self):
        """Load workflows from disk."""
        path = self._get_data_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)

            for wid, wf_data in data.get("workflows", {}).items():
                # Parse steps
                steps = []
                for step_data in wf_data.get("steps", []):
                    steps.append(WorkflowStep(**step_data))
                wf_data["steps"] = steps
                self._workflows[wid] = WorkflowTemplate(**wf_data)

            for run_data in data.get("runs", []):
                self._runs.append(WorkflowRun(**run_data))
        except Exception:
            pass
