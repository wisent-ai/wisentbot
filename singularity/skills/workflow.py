#!/usr/bin/env python3
"""
Workflow Skill - Multi-step skill chaining for autonomous agents.

Allows agents to define, save, and execute multi-step workflows that
chain multiple skill actions together. This is the foundation for:
- Complex task execution (self-improvement)
- Service delivery pipelines (revenue)
- Reusable automation templates (goal setting)

Each workflow step can:
- Reference outputs from previous steps via {{step_name.field}}
- Run conditionally based on previous step success/failure
- Have retry logic for resilience
"""

import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from .base import Skill, SkillManifest, SkillAction, SkillResult


WORKFLOW_DIR = Path(__file__).parent.parent / "data" / "workflows"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    skill_id: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    condition: str = "always"  # "always", "on_success", "on_failure"
    retries: int = 0
    description: str = ""


@dataclass
class WorkflowDef:
    """A complete workflow definition."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: str = ""
    updated_at: str = ""
    run_count: int = 0
    last_run_at: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "WorkflowDef":
        steps = [WorkflowStep(**s) for s in d.get("steps", [])]
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            steps=steps,
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            run_count=d.get("run_count", 0),
            last_run_at=d.get("last_run_at", ""),
            tags=d.get("tags", []),
        )


@dataclass
class StepResult:
    """Result of executing a single workflow step."""
    step_name: str
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    skipped: bool = False
    retries_used: int = 0
    duration_ms: float = 0


class WorkflowSkill(Skill):
    """
    Skill for defining and executing multi-step workflows.

    Workflows chain multiple skill actions together with:
    - Parameter templating: reference previous step outputs via {{step_name.field}}
    - Conditional execution: run steps only on success/failure of previous steps
    - Retry logic: automatically retry failed steps
    - Persistence: workflows are saved to disk and survive restarts
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._workflows: Dict[str, WorkflowDef] = {}
        self._skill_executor = None
        self._load_workflows()

    def set_skill_executor(self, executor):
        """Set the function used to execute skill actions.

        Args:
            executor: async callable(skill_id, action, params) -> SkillResult
        """
        self._skill_executor = executor

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="workflow",
            name="Workflow",
            version="1.0.0",
            category="automation",
            description="Define and execute multi-step workflows that chain skill actions together",
            actions=[
                SkillAction(
                    name="create",
                    description="Create a new workflow with multiple steps",
                    parameters={
                        "id": {
                            "type": "string",
                            "required": True,
                            "description": "Unique workflow identifier (e.g., 'deploy_site')"
                        },
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Human-readable workflow name"
                        },
                        "description": {
                            "type": "string",
                            "required": False,
                            "description": "What this workflow does"
                        },
                        "steps": {
                            "type": "array",
                            "required": True,
                            "description": "List of steps. Each: {name, skill_id, action, params, condition?, retries?, description?}"
                        },
                        "tags": {
                            "type": "array",
                            "required": False,
                            "description": "Tags for categorizing workflows"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="run",
                    description="Execute a saved workflow. Params are passed as variables to step templates.",
                    parameters={
                        "workflow_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the workflow to run"
                        },
                        "variables": {
                            "type": "object",
                            "required": False,
                            "description": "Variables to pass into workflow step params (referenced via {{var.name}})"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List all saved workflows",
                    parameters={
                        "tag": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by tag"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="show",
                    description="Show full details of a workflow",
                    parameters={
                        "workflow_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the workflow to show"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="delete",
                    description="Delete a saved workflow",
                    parameters={
                        "workflow_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the workflow to delete"
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "create": self._create,
            "run": self._run,
            "list": self._list,
            "show": self._show,
            "delete": self._delete,
        }
        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    async def _create(self, params: Dict) -> SkillResult:
        """Create a new workflow."""
        wf_id = params.get("id", "").strip()
        name = params.get("name", "").strip()
        description = params.get("description", "")
        raw_steps = params.get("steps", [])
        tags = params.get("tags", [])

        if not wf_id:
            return SkillResult(success=False, message="Workflow ID is required")
        if not name:
            return SkillResult(success=False, message="Workflow name is required")
        if not raw_steps:
            return SkillResult(success=False, message="At least one step is required")

        # Validate and parse steps
        steps = []
        seen_names = set()
        for i, s in enumerate(raw_steps):
            if isinstance(s, str):
                try:
                    s = json.loads(s)
                except (json.JSONDecodeError, TypeError):
                    return SkillResult(success=False, message=f"Step {i} is not valid JSON")

            step_name = s.get("name", f"step_{i}")
            if step_name in seen_names:
                return SkillResult(success=False, message=f"Duplicate step name: {step_name}")
            seen_names.add(step_name)

            if not s.get("skill_id") or not s.get("action"):
                return SkillResult(
                    success=False,
                    message=f"Step '{step_name}' requires 'skill_id' and 'action'"
                )

            condition = s.get("condition", "always")
            if condition not in ("always", "on_success", "on_failure"):
                return SkillResult(
                    success=False,
                    message=f"Step '{step_name}' has invalid condition '{condition}'. Use: always, on_success, on_failure"
                )

            steps.append(WorkflowStep(
                name=step_name,
                skill_id=s["skill_id"],
                action=s["action"],
                params=s.get("params", {}),
                condition=condition,
                retries=int(s.get("retries", 0)),
                description=s.get("description", ""),
            ))

        now = datetime.now().isoformat()
        workflow = WorkflowDef(
            id=wf_id,
            name=name,
            description=description,
            steps=steps,
            created_at=now,
            updated_at=now,
            tags=tags,
        )

        self._workflows[wf_id] = workflow
        self._save_workflow(workflow)

        return SkillResult(
            success=True,
            message=f"Workflow '{name}' created with {len(steps)} step(s)",
            data={
                "id": wf_id,
                "name": name,
                "steps": len(steps),
                "step_names": [s.name for s in steps],
            }
        )

    async def _run(self, params: Dict) -> SkillResult:
        """Execute a workflow."""
        wf_id = params.get("workflow_id", "").strip()
        variables = params.get("variables", {})

        if not wf_id:
            return SkillResult(success=False, message="workflow_id is required")

        workflow = self._workflows.get(wf_id)
        if not workflow:
            return SkillResult(success=False, message=f"Workflow not found: {wf_id}")

        if not self._skill_executor:
            return SkillResult(
                success=False,
                message="Workflow executor not configured. Cannot run workflows."
            )

        # Execute steps
        step_results: Dict[str, StepResult] = {}
        all_outputs: Dict[str, Any] = {"var": variables}
        last_success = True
        total_steps = len(workflow.steps)
        steps_run = 0
        steps_skipped = 0
        steps_failed = 0

        for step in workflow.steps:
            # Check condition
            should_run = (
                step.condition == "always"
                or (step.condition == "on_success" and last_success)
                or (step.condition == "on_failure" and not last_success)
            )

            if not should_run:
                step_results[step.name] = StepResult(
                    step_name=step.name,
                    success=True,
                    skipped=True,
                    message="Skipped due to condition",
                )
                steps_skipped += 1
                continue

            # Resolve parameter templates
            resolved_params = self._resolve_templates(step.params, all_outputs)

            # Execute with retries
            attempt = 0
            max_attempts = 1 + step.retries
            step_success = False
            step_output = {}
            step_message = ""
            start_time = time.time()

            while attempt < max_attempts:
                attempt += 1
                try:
                    result = await self._skill_executor(
                        step.skill_id, step.action, resolved_params
                    )
                    step_success = result.success
                    step_output = result.data or {}
                    step_message = result.message
                    if step_success:
                        break
                except Exception as e:
                    step_message = f"Error: {e}"
                    step_success = False

            duration_ms = (time.time() - start_time) * 1000

            sr = StepResult(
                step_name=step.name,
                success=step_success,
                output=step_output,
                message=step_message,
                retries_used=attempt - 1,
                duration_ms=round(duration_ms, 1),
            )
            step_results[step.name] = sr
            steps_run += 1

            # Store outputs for template resolution
            all_outputs[step.name] = step_output

            last_success = step_success
            if not step_success:
                steps_failed += 1

        # Update workflow stats
        workflow.run_count += 1
        workflow.last_run_at = datetime.now().isoformat()
        self._save_workflow(workflow)

        overall_success = steps_failed == 0

        return SkillResult(
            success=overall_success,
            message=(
                f"Workflow '{workflow.name}' completed: "
                f"{steps_run} run, {steps_skipped} skipped, {steps_failed} failed"
            ),
            data={
                "workflow_id": wf_id,
                "success": overall_success,
                "total_steps": total_steps,
                "steps_run": steps_run,
                "steps_skipped": steps_skipped,
                "steps_failed": steps_failed,
                "step_results": {
                    name: {
                        "success": sr.success,
                        "skipped": sr.skipped,
                        "message": sr.message,
                        "output": sr.output,
                        "retries_used": sr.retries_used,
                        "duration_ms": sr.duration_ms,
                    }
                    for name, sr in step_results.items()
                },
                "run_count": workflow.run_count,
            }
        )

    async def _list(self, params: Dict) -> SkillResult:
        """List all workflows."""
        tag_filter = params.get("tag", "")
        workflows = []
        for wf in self._workflows.values():
            if tag_filter and tag_filter not in wf.tags:
                continue
            workflows.append({
                "id": wf.id,
                "name": wf.name,
                "description": wf.description[:100] if wf.description else "",
                "steps": len(wf.steps),
                "run_count": wf.run_count,
                "tags": wf.tags,
                "last_run_at": wf.last_run_at,
            })

        return SkillResult(
            success=True,
            message=f"{len(workflows)} workflow(s) found",
            data={"workflows": workflows, "count": len(workflows)}
        )

    async def _show(self, params: Dict) -> SkillResult:
        """Show workflow details."""
        wf_id = params.get("workflow_id", "").strip()
        if not wf_id:
            return SkillResult(success=False, message="workflow_id is required")

        workflow = self._workflows.get(wf_id)
        if not workflow:
            return SkillResult(success=False, message=f"Workflow not found: {wf_id}")

        return SkillResult(
            success=True,
            message=f"Workflow: {workflow.name}",
            data=workflow.to_dict()
        )

    async def _delete(self, params: Dict) -> SkillResult:
        """Delete a workflow."""
        wf_id = params.get("workflow_id", "").strip()
        if not wf_id:
            return SkillResult(success=False, message="workflow_id is required")

        if wf_id not in self._workflows:
            return SkillResult(success=False, message=f"Workflow not found: {wf_id}")

        name = self._workflows[wf_id].name
        del self._workflows[wf_id]

        # Remove file
        wf_file = WORKFLOW_DIR / f"{wf_id}.json"
        if wf_file.exists():
            wf_file.unlink()

        return SkillResult(
            success=True,
            message=f"Workflow '{name}' deleted",
            data={"id": wf_id, "name": name}
        )

    def _resolve_templates(self, params: Dict, context: Dict) -> Dict:
        """Resolve {{step_name.field}} templates in parameters.

        Supports nested field access: {{step_name.field.subfield}}
        and variable access: {{var.name}}
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_string(value, context)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_templates(value, context)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_string(v, context) if isinstance(v, str)
                    else self._resolve_templates(v, context) if isinstance(v, dict)
                    else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    def _resolve_string(self, text: str, context: Dict) -> str:
        """Resolve template variables in a string."""
        def replacer(match):
            path = match.group(1).strip()
            parts = path.split(".")
            obj = context
            for part in parts:
                if isinstance(obj, dict):
                    obj = obj.get(part, match.group(0))
                else:
                    return match.group(0)
            if isinstance(obj, (dict, list)):
                return json.dumps(obj)
            return str(obj)

        return re.sub(r"\{\{(.+?)\}\}", replacer, text)

    def _save_workflow(self, workflow: WorkflowDef):
        """Save a workflow to disk."""
        WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
        wf_file = WORKFLOW_DIR / f"{workflow.id}.json"
        with open(wf_file, "w") as f:
            json.dump(workflow.to_dict(), f, indent=2)

    def _load_workflows(self):
        """Load all workflows from disk."""
        if not WORKFLOW_DIR.exists():
            return
        for wf_file in WORKFLOW_DIR.glob("*.json"):
            try:
                with open(wf_file) as f:
                    data = json.load(f)
                wf = WorkflowDef.from_dict(data)
                self._workflows[wf.id] = wf
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
