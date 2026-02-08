#!/usr/bin/env python3
"""
PipelinePlannerSkill - Converts goals into executable multi-step pipelines.

Bridges the PlannerSkill (which manages goals/tasks) with PipelineExecutor
(which runs multi-step pipelines in a single cycle). Instead of the agent
executing one task per LLM cycle, this skill:

1. Takes a goal and resolves its task dependency graph
2. Generates an ordered pipeline of executable steps
3. Adds conditional logic, fallbacks, and cost guards
4. Returns a pipeline dict ready for PipelineExecutor.run_from_dicts()

This is the #1 priority from session 146 memory. It's a force multiplier:
the agent plans once, then PipelineExecutor runs the whole chain.

Pillar: Self-Improvement (think once, execute many = lower cost per goal)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import Skill, SkillAction, SkillManifest, SkillResult


PIPELINE_PLANS_FILE = Path(__file__).parent.parent / "data" / "pipeline_plans.json"

# Default tool mappings for common skill hints
SKILL_TOOL_MAP = {
    "shell": "shell:run",
    "github": "github:create_pr",
    "filesystem": "filesystem:write",
    "code_review": "code_review:review",
    "deployment": "deployment:deploy",
    "content": "content:generate",
    "email": "email:send",
    "browser": "browser:fetch",
    "planner": "planner:update_task",
}

# Effort-based defaults
EFFORT_DEFAULTS = {
    "small": {"timeout_seconds": 30.0, "max_cost": 0.02, "retry_count": 1},
    "medium": {"timeout_seconds": 60.0, "max_cost": 0.05, "retry_count": 0},
    "large": {"timeout_seconds": 120.0, "max_cost": 0.10, "retry_count": 0},
}


class PipelinePlannerSkill(Skill):
    """
    Converts goals from PlannerSkill into executable pipelines for PipelineExecutor.

    Key capabilities:
    - generate_pipeline: Turn a goal's tasks into a pipeline with dependency ordering
    - optimize_pipeline: Add conditions, fallbacks, and cost guards to a pipeline
    - estimate: Preview cost/duration of a pipeline before running
    - save/load: Persist pipeline plans for reuse
    - history: Track which pipelines were generated and their outcomes
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        PIPELINE_PLANS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not PIPELINE_PLANS_FILE.exists():
            self._save({"plans": [], "templates": {}, "history": []})

    def _load(self) -> Dict:
        try:
            with open(PIPELINE_PLANS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"plans": [], "templates": {}, "history": []}

    def _save(self, data: Dict):
        with open(PIPELINE_PLANS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="pipeline_planner",
            name="Pipeline Planner",
            version="1.0.0",
            category="planning",
            description="Convert goals into executable multi-step pipelines for PipelineExecutor",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="generate",
                    description="Generate an executable pipeline from a goal's tasks",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID from PlannerSkill"},
                        "max_steps": {"type": "integer", "required": False, "description": "Max pipeline steps (default 10)"},
                        "max_cost": {"type": "float", "required": False, "description": "Total cost budget (default 0.50)"},
                        "include_fallbacks": {"type": "boolean", "required": False, "description": "Add fallback steps on failure (default true)"},
                    },
                ),
                SkillAction(
                    name="generate_from_tasks",
                    description="Generate a pipeline from an explicit list of tasks (no PlannerSkill needed)",
                    parameters={
                        "tasks": {"type": "array", "required": True, "description": "List of task dicts with title, tool, params"},
                        "name": {"type": "string", "required": False, "description": "Pipeline name"},
                        "max_cost": {"type": "float", "required": False, "description": "Total cost budget"},
                    },
                ),
                SkillAction(
                    name="optimize",
                    description="Optimize a pipeline by adding conditions, cost guards, and reordering",
                    parameters={
                        "pipeline": {"type": "array", "required": True, "description": "Pipeline steps to optimize"},
                        "strategy": {"type": "string", "required": False, "description": "Optimization strategy: 'cost', 'speed', 'reliability'"},
                    },
                ),
                SkillAction(
                    name="estimate",
                    description="Estimate cost and duration of a pipeline before running",
                    parameters={
                        "pipeline": {"type": "array", "required": True, "description": "Pipeline steps to estimate"},
                    },
                ),
                SkillAction(
                    name="save_template",
                    description="Save a pipeline as a reusable template",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Template name"},
                        "pipeline": {"type": "array", "required": True, "description": "Pipeline steps"},
                        "description": {"type": "string", "required": False, "description": "What this template does"},
                    },
                ),
                SkillAction(
                    name="load_template",
                    description="Load a saved pipeline template",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Template name"},
                        "overrides": {"type": "object", "required": False, "description": "Override template params"},
                    },
                ),
                SkillAction(
                    name="record_outcome",
                    description="Record the outcome of a pipeline execution for learning",
                    parameters={
                        "plan_id": {"type": "string", "required": True, "description": "Pipeline plan ID"},
                        "success": {"type": "boolean", "required": True, "description": "Whether pipeline succeeded"},
                        "steps_succeeded": {"type": "integer", "required": False, "description": "Number of steps that succeeded"},
                        "total_cost": {"type": "float", "required": False, "description": "Actual cost incurred"},
                        "notes": {"type": "string", "required": False, "description": "Additional notes"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Show pipeline planner status: saved plans, templates, success rates",
                    parameters={},
                ),
            ],
        )

    def check_credentials(self) -> bool:
        return True  # No credentials needed

    def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "generate": self._generate,
            "generate_from_tasks": self._generate_from_tasks,
            "optimize": self._optimize,
            "estimate": self._estimate,
            "save_template": self._save_template,
            "load_template": self._load_template,
            "record_outcome": self._record_outcome,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    def _resolve_tasks_from_goal(self, goal_id: str) -> Optional[List[Dict]]:
        """Load tasks from PlannerSkill's data for a given goal."""
        plans_file = Path(__file__).parent.parent / "data" / "plans.json"
        try:
            with open(plans_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        for goal in data.get("goals", []):
            if goal["id"] == goal_id:
                return goal.get("tasks", [])
        return None

    def _topological_sort(self, tasks: List[Dict]) -> List[Dict]:
        """Sort tasks respecting dependency order (Kahn's algorithm)."""
        task_map = {t.get("id", str(i)): t for i, t in enumerate(tasks)}
        in_degree = {tid: 0 for tid in task_map}
        adj = {tid: [] for tid in task_map}

        for tid, task in task_map.items():
            for dep in task.get("depends_on", []):
                if dep in adj:
                    adj[dep].append(tid)
                    in_degree[tid] = in_degree.get(tid, 0) + 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        ordered = []

        while queue:
            # Sort queue by priority for deterministic ordering
            queue.sort(key=lambda tid: _priority_key(task_map[tid]))
            node = queue.pop(0)
            ordered.append(task_map[node])
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Add any remaining tasks (circular deps) at the end
        seen = {t.get("id", str(i)) for i, t in enumerate(ordered)}
        for tid, task in task_map.items():
            if tid not in seen:
                ordered.append(task)

        return ordered

    def _task_to_step(self, task: Dict, include_fallbacks: bool = True) -> Dict:
        """Convert a PlannerSkill task into a PipelineExecutor step dict."""
        skill_hint = task.get("skill_hint", "")
        tool = SKILL_TOOL_MAP.get(skill_hint, skill_hint) if skill_hint else ""
        effort = task.get("effort", "medium")
        defaults = EFFORT_DEFAULTS.get(effort, EFFORT_DEFAULTS["medium"])

        step = {
            "tool": tool or "planner:update_task",
            "params": {
                "task_id": task.get("id", ""),
                "task_title": task.get("title", ""),
                "description": task.get("description", ""),
                **(task.get("params", {})),
            },
            "label": task.get("title", "unnamed step"),
            "timeout_seconds": defaults["timeout_seconds"],
            "max_cost": defaults["max_cost"],
            "required": task.get("required", True),
            "retry_count": defaults["retry_count"],
        }

        # Add fallback step if requested
        if include_fallbacks and task.get("fallback"):
            step["on_failure"] = {
                "tool": task["fallback"].get("tool", "planner:update_task"),
                "params": task["fallback"].get("params", {
                    "task_id": task.get("id", ""),
                    "status": "failed",
                }),
            }

        # Add condition based on dependencies
        deps = task.get("depends_on", [])
        if deps:
            step["condition"] = {"prev_success": True}

        return step

    def _generate(self, params: Dict) -> SkillResult:
        """Generate a pipeline from a PlannerSkill goal."""
        goal_id = params.get("goal_id", "")
        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        max_steps = params.get("max_steps", 10)
        max_cost = params.get("max_cost", 0.50)
        include_fallbacks = params.get("include_fallbacks", True)

        tasks = self._resolve_tasks_from_goal(goal_id)
        if tasks is None:
            return SkillResult(
                success=False,
                message=f"Goal {goal_id} not found in PlannerSkill data",
            )

        # Filter to pending tasks only
        pending = [t for t in tasks if t.get("status", "pending") == "pending"]
        if not pending:
            return SkillResult(
                success=True,
                message="No pending tasks for this goal - all may be complete",
                data={"pipeline": [], "goal_id": goal_id},
            )

        # Topological sort respecting dependencies
        ordered = self._topological_sort(pending)

        # Limit to max_steps
        ordered = ordered[:max_steps]

        # Convert to pipeline steps
        steps = [self._task_to_step(t, include_fallbacks) for t in ordered]

        # Apply cost budget across steps
        steps = self._apply_cost_budget(steps, max_cost)

        # Add a completion step that marks progress
        steps.append({
            "tool": "planner:progress",
            "params": {"goal_id": goal_id},
            "label": "Check goal progress",
            "timeout_seconds": 10.0,
            "max_cost": 0.0,
            "required": False,
        })

        # Save the plan
        plan_id = f"pp-{datetime.now().strftime('%Y%m%d%H%M%S')}-{goal_id[:8]}"
        data = self._load()
        plan_record = {
            "id": plan_id,
            "goal_id": goal_id,
            "created_at": datetime.now().isoformat(),
            "step_count": len(steps),
            "max_cost": max_cost,
            "task_ids": [t.get("id", "") for t in ordered],
            "outcome": None,
        }
        data["plans"].append(plan_record)
        # Keep last 50 plans
        data["plans"] = data["plans"][-50:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Generated pipeline '{plan_id}' with {len(steps)} steps from {len(pending)} pending tasks",
            data={
                "plan_id": plan_id,
                "pipeline": steps,
                "goal_id": goal_id,
                "task_count": len(ordered),
                "estimated_cost": sum(s.get("max_cost", 0.05) for s in steps),
                "estimated_duration_s": sum(s.get("timeout_seconds", 30) for s in steps),
            },
        )

    def _generate_from_tasks(self, params: Dict) -> SkillResult:
        """Generate a pipeline from an explicit task list (no PlannerSkill required)."""
        tasks = params.get("tasks", [])
        if not tasks:
            return SkillResult(success=False, message="tasks list is required")

        name = params.get("name", "ad-hoc-pipeline")
        max_cost = params.get("max_cost", 0.50)

        steps = []
        for i, task in enumerate(tasks):
            step = {
                "tool": task.get("tool", ""),
                "params": task.get("params", {}),
                "label": task.get("title", task.get("label", f"step-{i}")),
                "timeout_seconds": task.get("timeout_seconds", 30.0),
                "max_cost": task.get("max_cost", 0.05),
                "required": task.get("required", True),
                "retry_count": task.get("retry_count", 0),
            }
            if task.get("condition"):
                step["condition"] = task["condition"]
            if task.get("on_failure"):
                step["on_failure"] = task["on_failure"]
            steps.append(step)

        steps = self._apply_cost_budget(steps, max_cost)

        plan_id = f"pp-{datetime.now().strftime('%Y%m%d%H%M%S')}-adhoc"
        data = self._load()
        data["plans"].append({
            "id": plan_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "step_count": len(steps),
            "max_cost": max_cost,
            "outcome": None,
        })
        data["plans"] = data["plans"][-50:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Generated pipeline '{plan_id}' with {len(steps)} steps",
            data={
                "plan_id": plan_id,
                "pipeline": steps,
                "name": name,
                "estimated_cost": sum(s.get("max_cost", 0.05) for s in steps),
            },
        )

    def _optimize(self, params: Dict) -> SkillResult:
        """Optimize a pipeline based on strategy."""
        pipeline = params.get("pipeline", [])
        if not pipeline:
            return SkillResult(success=False, message="pipeline is required")

        strategy = params.get("strategy", "reliability")
        optimized = list(pipeline)  # Copy

        if strategy == "cost":
            optimized = self._optimize_for_cost(optimized)
        elif strategy == "speed":
            optimized = self._optimize_for_speed(optimized)
        elif strategy == "reliability":
            optimized = self._optimize_for_reliability(optimized)

        changes = []
        for i, (orig, opt) in enumerate(zip(pipeline, optimized)):
            if orig != opt:
                changes.append(f"Step {i}: modified for {strategy}")

        return SkillResult(
            success=True,
            message=f"Optimized pipeline ({strategy}): {len(changes)} steps modified",
            data={
                "pipeline": optimized,
                "strategy": strategy,
                "changes": changes,
                "original_cost": sum(s.get("max_cost", 0.05) for s in pipeline),
                "optimized_cost": sum(s.get("max_cost", 0.05) for s in optimized),
            },
        )

    def _optimize_for_cost(self, steps: List[Dict]) -> List[Dict]:
        """Minimize cost: lower budgets, skip optional steps early."""
        for step in steps:
            step["max_cost"] = step.get("max_cost", 0.05) * 0.7
            step["timeout_seconds"] = min(step.get("timeout_seconds", 30), 30.0)
            # Make non-required steps conditional on previous success
            if not step.get("required", True):
                step["condition"] = step.get("condition") or {"prev_success": True}
        return steps

    def _optimize_for_speed(self, steps: List[Dict]) -> List[Dict]:
        """Minimize time: tighter timeouts, no retries."""
        for step in steps:
            step["timeout_seconds"] = min(step.get("timeout_seconds", 30), 15.0)
            step["retry_count"] = 0
        return steps

    def _optimize_for_reliability(self, steps: List[Dict]) -> List[Dict]:
        """Maximize success: add retries, longer timeouts, fallbacks."""
        for step in steps:
            if step.get("required", True):
                step["retry_count"] = max(step.get("retry_count", 0), 1)
                step["timeout_seconds"] = step.get("timeout_seconds", 30) * 1.5
            # Add generic fallback if none exists
            if not step.get("on_failure") and step.get("required", True):
                step["on_failure"] = {
                    "tool": "planner:update_task",
                    "params": {
                        "status": "failed",
                        "note": f"Pipeline step '{step.get('label', 'unknown')}' failed",
                    },
                }
        return steps

    def _estimate(self, params: Dict) -> SkillResult:
        """Estimate cost and duration of a pipeline."""
        pipeline = params.get("pipeline", [])
        if not pipeline:
            return SkillResult(success=False, message="pipeline is required")

        total_cost = sum(s.get("max_cost", 0.05) for s in pipeline)
        total_time = sum(s.get("timeout_seconds", 30) for s in pipeline)
        retry_cost = sum(
            s.get("max_cost", 0.05) * s.get("retry_count", 0) for s in pipeline
        )
        required_steps = sum(1 for s in pipeline if s.get("required", True))
        optional_steps = len(pipeline) - required_steps
        conditional_steps = sum(1 for s in pipeline if s.get("condition"))
        has_fallbacks = sum(1 for s in pipeline if s.get("on_failure"))

        return SkillResult(
            success=True,
            message=f"Pipeline estimate: {len(pipeline)} steps, ${total_cost:.3f} max cost, {total_time:.0f}s max duration",
            data={
                "step_count": len(pipeline),
                "max_cost": round(total_cost, 4),
                "max_cost_with_retries": round(total_cost + retry_cost, 4),
                "max_duration_seconds": round(total_time, 1),
                "required_steps": required_steps,
                "optional_steps": optional_steps,
                "conditional_steps": conditional_steps,
                "steps_with_fallbacks": has_fallbacks,
                "tools_used": list(set(s.get("tool", "") for s in pipeline)),
            },
        )

    def _save_template(self, params: Dict) -> SkillResult:
        """Save a pipeline as a reusable template."""
        name = params.get("name", "")
        if not name:
            return SkillResult(success=False, message="name is required")

        pipeline = params.get("pipeline", [])
        if not pipeline:
            return SkillResult(success=False, message="pipeline is required")

        data = self._load()
        data["templates"][name] = {
            "pipeline": pipeline,
            "description": params.get("description", ""),
            "created_at": datetime.now().isoformat(),
            "use_count": 0,
        }
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Template '{name}' saved with {len(pipeline)} steps",
            data={"name": name, "step_count": len(pipeline)},
        )

    def _load_template(self, params: Dict) -> SkillResult:
        """Load a saved pipeline template."""
        name = params.get("name", "")
        if not name:
            return SkillResult(success=False, message="name is required")

        data = self._load()
        template = data["templates"].get(name)
        if not template:
            available = list(data["templates"].keys())
            return SkillResult(
                success=False,
                message=f"Template '{name}' not found. Available: {available}",
            )

        pipeline = list(template["pipeline"])  # Copy
        overrides = params.get("overrides", {})

        # Apply parameter overrides to all steps
        if overrides:
            for step in pipeline:
                for key, value in overrides.items():
                    if key in step:
                        step[key] = value
                    elif key in step.get("params", {}):
                        step["params"][key] = value

        # Track usage
        template["use_count"] = template.get("use_count", 0) + 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Loaded template '{name}' ({len(pipeline)} steps)",
            data={
                "name": name,
                "pipeline": pipeline,
                "description": template.get("description", ""),
                "use_count": template["use_count"],
            },
        )

    def _record_outcome(self, params: Dict) -> SkillResult:
        """Record the outcome of a pipeline execution."""
        plan_id = params.get("plan_id", "")
        if not plan_id:
            return SkillResult(success=False, message="plan_id is required")

        data = self._load()

        # Find and update the plan
        plan = None
        for p in data["plans"]:
            if p["id"] == plan_id:
                plan = p
                break

        if not plan:
            return SkillResult(success=False, message=f"Plan {plan_id} not found")

        outcome = {
            "success": params.get("success", False),
            "steps_succeeded": params.get("steps_succeeded", 0),
            "total_cost": params.get("total_cost", 0.0),
            "notes": params.get("notes", ""),
            "recorded_at": datetime.now().isoformat(),
        }
        plan["outcome"] = outcome

        # Add to history
        data["history"].append({
            "plan_id": plan_id,
            **outcome,
        })
        data["history"] = data["history"][-100:]  # Keep last 100
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recorded outcome for {plan_id}: {'success' if outcome['success'] else 'failure'}",
            data={"plan_id": plan_id, "outcome": outcome},
        )

    def _status(self, params: Dict) -> SkillResult:
        """Show pipeline planner status."""
        data = self._load()

        plans = data.get("plans", [])
        templates = data.get("templates", {})
        history = data.get("history", [])

        total_plans = len(plans)
        with_outcomes = [p for p in plans if p.get("outcome")]
        successes = [p for p in with_outcomes if p["outcome"].get("success")]
        success_rate = len(successes) / len(with_outcomes) if with_outcomes else 0

        total_cost = sum(
            h.get("total_cost", 0) for h in history
        )

        return SkillResult(
            success=True,
            message=f"Pipeline Planner: {total_plans} plans, {len(templates)} templates, {success_rate:.0%} success rate",
            data={
                "total_plans": total_plans,
                "plans_with_outcomes": len(with_outcomes),
                "success_count": len(successes),
                "success_rate": round(success_rate, 3),
                "total_cost_spent": round(total_cost, 4),
                "template_count": len(templates),
                "template_names": list(templates.keys()),
                "recent_plans": plans[-5:] if plans else [],
            },
        )

    def _apply_cost_budget(self, steps: List[Dict], max_cost: float) -> List[Dict]:
        """Distribute cost budget across steps proportionally."""
        if not steps:
            return steps

        total_requested = sum(s.get("max_cost", 0.05) for s in steps)
        if total_requested <= max_cost:
            return steps  # Already within budget

        # Scale down proportionally
        scale = max_cost / total_requested if total_requested > 0 else 1.0
        for step in steps:
            step["max_cost"] = round(step.get("max_cost", 0.05) * scale, 4)

        return steps


def _priority_key(task: Dict) -> tuple:
    """Sort key for task priority ordering."""
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    effort_order = {"small": 0, "medium": 1, "large": 2}
    return (
        priority_order.get(task.get("priority", "medium"), 2),
        effort_order.get(task.get("effort", "medium"), 1),
    )
