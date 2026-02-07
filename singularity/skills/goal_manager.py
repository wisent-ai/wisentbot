#!/usr/bin/env python3
"""
Goal Manager Skill - Persistent goal tracking for autonomous agents.

Gives agents the ability to:
- Set, track, and complete goals that persist across sessions
- Break goals into sub-tasks with dependencies
- Prioritize goals based on urgency and importance
- Track progress and milestones
- Surface active goals to the cognition engine for goal-directed behavior

Goals are stored in a local JSON file (no external dependencies).
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillManifest, SkillAction, SkillResult

GOALS_DIR = Path(__file__).parent.parent / "data"
GOALS_FILE = GOALS_DIR / "goals.json"


class GoalManagerSkill(Skill):
    """
    Persistent goal management for autonomous agents.

    Enables goal-directed behavior by maintaining a persistent store
    of goals and tasks that survive agent restarts. Goals are surfaced
    to the cognition engine so the agent can make decisions aligned
    with its objectives.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._goals: Dict[str, Dict] = {}
        self._load()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="goals",
            name="Goal Manager",
            version="1.0.0",
            category="planning",
            description="Persistent goal and task tracking for autonomous behavior",
            actions=[
                SkillAction(
                    name="add",
                    description="Add a new goal with priority and optional deadline",
                    parameters={
                        "title": {"type": "string", "required": True, "description": "Goal title"},
                        "description": {"type": "string", "required": False, "description": "Detailed description"},
                        "priority": {"type": "string", "required": False, "description": "critical, high, medium, low (default: medium)"},
                        "pillar": {"type": "string", "required": False, "description": "Which pillar: self_improvement, revenue, replication, goal_setting"},
                        "deadline": {"type": "string", "required": False, "description": "Deadline (ISO format or human-readable)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="add_task",
                    description="Add a task/sub-goal to an existing goal",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Parent goal ID"},
                        "title": {"type": "string", "required": True, "description": "Task title"},
                        "description": {"type": "string", "required": False, "description": "Task details"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List all goals, optionally filtered by status or pillar",
                    parameters={
                        "status": {"type": "string", "required": False, "description": "Filter: active, completed, abandoned (default: active)"},
                        "pillar": {"type": "string", "required": False, "description": "Filter by pillar"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="update",
                    description="Update goal progress or status",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID to update"},
                        "progress": {"type": "integer", "required": False, "description": "Progress percentage (0-100)"},
                        "status": {"type": "string", "required": False, "description": "New status: active, completed, abandoned, blocked"},
                        "note": {"type": "string", "required": False, "description": "Progress note"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complete_task",
                    description="Mark a task as completed within a goal",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Parent goal ID"},
                        "task_index": {"type": "integer", "required": True, "description": "Task index (0-based)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="focus",
                    description="Get the single most important goal to work on right now",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="remove",
                    description="Remove a goal permanently",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID to remove"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="summary",
                    description="Get a summary of all goals with progress stats",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "add": self._add_goal,
            "add_task": self._add_task,
            "list": self._list_goals,
            "update": self._update_goal,
            "complete_task": self._complete_task,
            "focus": self._focus,
            "remove": self._remove_goal,
            "summary": self._summary,
        }
        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def get_active_goals(self) -> List[Dict]:
        """Get active goals for injection into agent state.

        Returns a compact list of active goals with their priorities
        and progress, suitable for inclusion in the cognition prompt.
        """
        active = []
        for gid, goal in self._goals.items():
            if goal.get("status") == "active":
                tasks = goal.get("tasks", [])
                completed_tasks = sum(1 for t in tasks if t.get("done"))
                total_tasks = len(tasks)
                active.append({
                    "id": gid,
                    "title": goal["title"],
                    "priority": goal.get("priority", "medium"),
                    "pillar": goal.get("pillar", ""),
                    "progress": goal.get("progress", 0),
                    "tasks_done": f"{completed_tasks}/{total_tasks}" if total_tasks > 0 else "no tasks",
                    "notes_count": len(goal.get("notes", [])),
                })
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        active.sort(key=lambda g: priority_order.get(g["priority"], 2))
        return active

    # === Internal handlers ===

    async def _add_goal(self, params: Dict) -> SkillResult:
        title = params.get("title", "").strip()
        if not title:
            return SkillResult(success=False, message="Title is required")

        gid = str(uuid.uuid4())[:8]
        goal = {
            "title": title,
            "description": params.get("description", ""),
            "priority": params.get("priority", "medium"),
            "pillar": params.get("pillar", ""),
            "status": "active",
            "progress": 0,
            "tasks": [],
            "notes": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "deadline": params.get("deadline", ""),
        }
        self._goals[gid] = goal
        self._save()
        return SkillResult(
            success=True,
            message=f"Goal '{title}' created with ID {gid}",
            data={"goal_id": gid, "goal": goal},
        )

    async def _add_task(self, params: Dict) -> SkillResult:
        gid = params.get("goal_id", "").strip()
        title = params.get("title", "").strip()
        if not gid or not title:
            return SkillResult(success=False, message="goal_id and title required")

        goal = self._goals.get(gid)
        if not goal:
            return SkillResult(success=False, message=f"Goal {gid} not found")

        task = {
            "title": title,
            "description": params.get("description", ""),
            "done": False,
            "created_at": time.time(),
        }
        goal["tasks"].append(task)
        goal["updated_at"] = time.time()
        self._save()
        idx = len(goal["tasks"]) - 1
        return SkillResult(
            success=True,
            message=f"Task '{title}' added to goal '{goal['title']}' (index {idx})",
            data={"task_index": idx, "task": task},
        )

    async def _list_goals(self, params: Dict) -> SkillResult:
        status_filter = params.get("status", "active")
        pillar_filter = params.get("pillar", "")

        filtered = {}
        for gid, goal in self._goals.items():
            if status_filter and goal.get("status") != status_filter:
                continue
            if pillar_filter and goal.get("pillar") != pillar_filter:
                continue
            filtered[gid] = goal

        return SkillResult(
            success=True,
            message=f"Found {len(filtered)} goals (status={status_filter})",
            data={"goals": filtered, "count": len(filtered)},
        )

    async def _update_goal(self, params: Dict) -> SkillResult:
        gid = params.get("goal_id", "").strip()
        if not gid:
            return SkillResult(success=False, message="goal_id required")

        goal = self._goals.get(gid)
        if not goal:
            return SkillResult(success=False, message=f"Goal {gid} not found")

        if "progress" in params:
            goal["progress"] = max(0, min(100, int(params["progress"])))
        if "status" in params:
            goal["status"] = params["status"]
        if "note" in params:
            goal["notes"].append({
                "text": params["note"],
                "timestamp": time.time(),
            })

        goal["updated_at"] = time.time()

        # Auto-complete if progress hits 100
        if goal["progress"] >= 100 and goal["status"] == "active":
            goal["status"] = "completed"

        self._save()
        return SkillResult(
            success=True,
            message=f"Goal '{goal['title']}' updated",
            data={"goal_id": gid, "goal": goal},
        )

    async def _complete_task(self, params: Dict) -> SkillResult:
        gid = params.get("goal_id", "").strip()
        task_idx = params.get("task_index")
        if not gid or task_idx is None:
            return SkillResult(success=False, message="goal_id and task_index required")

        goal = self._goals.get(gid)
        if not goal:
            return SkillResult(success=False, message=f"Goal {gid} not found")

        tasks = goal.get("tasks", [])
        if task_idx < 0 or task_idx >= len(tasks):
            return SkillResult(success=False, message=f"Task index {task_idx} out of range (0-{len(tasks)-1})")

        tasks[task_idx]["done"] = True
        tasks[task_idx]["completed_at"] = time.time()

        # Auto-update progress based on task completion
        completed = sum(1 for t in tasks if t.get("done"))
        goal["progress"] = int((completed / len(tasks)) * 100)
        goal["updated_at"] = time.time()

        if goal["progress"] >= 100 and goal["status"] == "active":
            goal["status"] = "completed"

        self._save()
        return SkillResult(
            success=True,
            message=f"Task '{tasks[task_idx]['title']}' completed ({completed}/{len(tasks)})",
            data={"goal_id": gid, "progress": goal["progress"], "status": goal["status"]},
        )

    async def _focus(self, params: Dict) -> SkillResult:
        active = self.get_active_goals()
        if not active:
            return SkillResult(
                success=True,
                message="No active goals. Set some goals first!",
                data={"focus": None},
            )

        top = active[0]
        goal = self._goals[top["id"]]
        pending_tasks = [t for t in goal.get("tasks", []) if not t.get("done")]

        return SkillResult(
            success=True,
            message=f"Focus: {top['title']} ({top['priority']} priority, {top['progress']}% done)",
            data={
                "focus_goal": top,
                "next_tasks": [t["title"] for t in pending_tasks[:3]],
                "description": goal.get("description", ""),
            },
        )

    async def _remove_goal(self, params: Dict) -> SkillResult:
        gid = params.get("goal_id", "").strip()
        if not gid:
            return SkillResult(success=False, message="goal_id required")

        goal = self._goals.pop(gid, None)
        if not goal:
            return SkillResult(success=False, message=f"Goal {gid} not found")

        self._save()
        return SkillResult(
            success=True,
            message=f"Goal '{goal['title']}' removed",
            data={"removed_goal_id": gid},
        )

    async def _summary(self, params: Dict) -> SkillResult:
        total = len(self._goals)
        by_status = {}
        by_pillar = {}
        by_priority = {}

        for goal in self._goals.values():
            status = goal.get("status", "active")
            by_status[status] = by_status.get(status, 0) + 1

            pillar = goal.get("pillar", "unset")
            by_pillar[pillar] = by_pillar.get(pillar, 0) + 1

            priority = goal.get("priority", "medium")
            by_priority[priority] = by_priority.get(priority, 0) + 1

        active_goals = self.get_active_goals()
        avg_progress = 0
        if active_goals:
            avg_progress = sum(g["progress"] for g in active_goals) / len(active_goals)

        return SkillResult(
            success=True,
            message=f"{total} goals total, {by_status.get('active', 0)} active, avg progress {avg_progress:.0f}%",
            data={
                "total": total,
                "by_status": by_status,
                "by_pillar": by_pillar,
                "by_priority": by_priority,
                "avg_progress": avg_progress,
                "active_goals": active_goals,
            },
        )

    # === Persistence ===

    def _load(self):
        try:
            if GOALS_FILE.exists():
                with open(GOALS_FILE, "r") as f:
                    self._goals = json.load(f)
        except (json.JSONDecodeError, IOError):
            self._goals = {}

    def _save(self):
        try:
            GOALS_DIR.mkdir(parents=True, exist_ok=True)
            with open(GOALS_FILE, "w") as f:
                json.dump(self._goals, f, indent=2)
        except IOError:
            pass
