#!/usr/bin/env python3
"""
TaskRunner - Task-oriented agent execution mode.

Transforms the agent from "runs forever" to "executes a discrete task and returns."
This enables:
- Revenue: Agent handles service requests (take task, do it, return result)
- Replication: Spawned agents get specific tasks to complete
- Practical use: Users can use the agent for specific, bounded tasks
- Self-improvement: Agent can run sub-tasks on itself
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from .cognition import CognitionEngine, AgentState, Decision, Action, TokenUsage
from .skills.base import (
    Skill,
    SkillManifest,
    SkillAction,
    SkillResult,
    SkillRegistry,
)


@dataclass
class TaskResult:
    """Structured result from task execution."""

    task: str
    status: str  # "completed", "failed", "max_cycles", "budget_exhausted"
    result_summary: str = ""
    result_data: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[Dict] = field(default_factory=list)
    cycles_used: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "task": self.task,
            "status": self.status,
            "result_summary": self.result_summary,
            "result_data": self.result_data,
            "cycles_used": self.cycles_used,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "actions_count": len(self.actions_taken),
        }


class TaskControlSkill(Skill):
    """
    Built-in skill for task lifecycle control.

    Provides actions:
    - done: Signal task completion with a result summary
    - fail: Signal task failure with an error message
    - status: Report current progress on the task
    """

    def __init__(self, credentials=None):
        super().__init__(credentials or {})
        self._completion_callback = None
        self._failure_callback = None
        self._status_log: List[str] = []

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="task",
            name="Task Control",
            version="1.0.0",
            category="system",
            description="Signal task completion, failure, or report progress",
            actions=[
                SkillAction(
                    name="done",
                    description="Signal that the current task is complete. Provide a summary of what was accomplished.",
                    parameters={
                        "summary": {
                            "type": "string",
                            "required": True,
                            "description": "Summary of what was accomplished",
                        },
                        "data": {
                            "type": "object",
                            "required": False,
                            "description": "Structured result data (optional)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="fail",
                    description="Signal that the current task cannot be completed. Explain why.",
                    parameters={
                        "reason": {
                            "type": "string",
                            "required": True,
                            "description": "Reason why the task failed",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="progress",
                    description="Report progress on the current task.",
                    parameters={
                        "status": {
                            "type": "string",
                            "required": True,
                            "description": "Current status/progress description",
                        },
                        "percent": {
                            "type": "number",
                            "required": False,
                            "description": "Estimated percent complete (0-100)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def set_callbacks(self, on_complete, on_fail):
        """Set callbacks for task completion/failure signals."""
        self._completion_callback = on_complete
        self._failure_callback = on_fail

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if action == "done":
            summary = params.get("summary", "Task completed")
            data = params.get("data", {})
            if self._completion_callback:
                self._completion_callback(summary, data)
            return SkillResult(
                success=True,
                message=f"Task completed: {summary}",
                data={"summary": summary, "result_data": data},
            )
        elif action == "fail":
            reason = params.get("reason", "Unknown failure")
            if self._failure_callback:
                self._failure_callback(reason)
            return SkillResult(
                success=True,
                message=f"Task failed: {reason}",
                data={"reason": reason},
            )
        elif action == "progress":
            status = params.get("status", "In progress")
            percent = params.get("percent", -1)
            self._status_log.append(status)
            return SkillResult(
                success=True,
                message=f"Progress: {status}" + (f" ({percent}%)" if percent >= 0 else ""),
                data={"status": status, "percent": percent},
            )
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    def check_credentials(self) -> bool:
        return True


TASK_PROMPT_ADDITION = """
═══════════════════════════════════════════════════════════════════════════════
                              CURRENT TASK
═══════════════════════════════════════════════════════════════════════════════

You have been assigned a specific task. Focus on completing it efficiently.

TASK: {task}

INSTRUCTIONS:
- Work towards completing this task using the available tools
- When the task is DONE, call task:done with a summary of what you accomplished
- If the task CANNOT be completed, call task:fail with the reason
- Use task:progress to report intermediate progress on longer tasks
- Be efficient - minimize unnecessary actions to save budget
- You have a maximum of {max_cycles} cycles to complete this task
"""


class TaskRunner:
    """
    Task-oriented agent execution.

    Wraps the agent's cognition and skills to execute a discrete task,
    detect completion, and return a structured result.

    Usage:
        runner = TaskRunner(
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
        )
        result = await runner.run("Create a Python hello world script")
        print(result.status)  # "completed"
        print(result.result_summary)  # "Created hello.py with print statement"
    """

    def __init__(
        self,
        name: str = "TaskAgent",
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
        llm_base_url: str = "http://localhost:8000/v1",
        anthropic_api_key: str = "",
        openai_api_key: str = "",
        system_prompt: Optional[str] = None,
        budget: float = 10.0,
        max_cycles: int = 50,
        cycle_delay: float = 0.1,
        skills: Optional[SkillRegistry] = None,
        quiet: bool = False,
    ):
        """
        Initialize TaskRunner.

        Args:
            name: Agent name
            llm_provider: LLM backend to use
            llm_model: Model identifier
            llm_base_url: Base URL for OpenAI-compatible API
            anthropic_api_key: Anthropic API key
            openai_api_key: OpenAI API key
            system_prompt: Custom system prompt (task prompt is appended)
            budget: Maximum USD to spend on this task
            max_cycles: Maximum think-act cycles
            cycle_delay: Seconds between cycles
            skills: Pre-configured SkillRegistry (optional)
            quiet: Suppress log output
        """
        self.name = name
        self.budget = budget
        self.max_cycles = max_cycles
        self.cycle_delay = cycle_delay
        self.quiet = quiet

        # Initialize cognition
        self.cognition = CognitionEngine(
            llm_provider=llm_provider,
            anthropic_api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=llm_base_url,
            llm_model=llm_model,
            agent_name=name,
            agent_ticker="TASK",
            agent_type="task_executor",
            agent_specialty="executing discrete tasks efficiently",
            system_prompt=system_prompt,
        )

        # Skills registry - use provided or create new
        self.skills = skills or SkillRegistry()

        # Install task control skill
        self._task_skill = TaskControlSkill()
        self.skills.skills["task"] = self._task_skill

        # Completion state
        self._completed = False
        self._failed = False
        self._result_summary = ""
        self._result_data = {}

    def _on_complete(self, summary: str, data: Dict):
        """Callback when agent signals task completion."""
        self._completed = True
        self._result_summary = summary
        self._result_data = data

    def _on_fail(self, reason: str):
        """Callback when agent signals task failure."""
        self._failed = True
        self._result_summary = reason

    def _get_tools(self) -> List[Dict]:
        """Get all available tools including task control."""
        tools = []
        for skill in self.skills.skills.values():
            for action in skill.manifest.actions:
                tool_name = f"{skill.manifest.skill_id}:{action.name}"
                tools.append({
                    "name": tool_name,
                    "description": action.description,
                    "parameters": action.parameters,
                })
        if not tools:
            tools.append({
                "name": "wait",
                "description": "No tools available. Wait.",
                "parameters": {},
            })
        return tools

    async def _execute(self, action: Action) -> Dict:
        """Execute an action via skills."""
        tool = action.tool
        params = action.params

        if tool == "wait":
            return {"status": "waited"}

        if ":" in tool:
            parts = tool.split(":", 1)
            skill_id = parts[0]
            action_name = parts[1] if len(parts) > 1 else ""

            skill = self.skills.get(skill_id)
            if skill:
                try:
                    result = await skill.execute(action_name, params)
                    return {
                        "status": "success" if result.success else "failed",
                        "data": result.data,
                        "message": result.message,
                    }
                except Exception as e:
                    return {"status": "error", "message": str(e)}

        return {"status": "error", "message": f"Unknown tool: {tool}"}

    def _log(self, tag: str, msg: str):
        """Log a message unless quiet mode."""
        if not self.quiet:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] [TASK] [{tag}] {msg}")

    async def run(self, task: str) -> TaskResult:
        """
        Execute a task and return the result.

        Args:
            task: Natural language description of the task to execute

        Returns:
            TaskResult with status, summary, cost, etc.
        """
        started_at = datetime.now()
        self._completed = False
        self._failed = False
        self._result_summary = ""
        self._result_data = {}

        # Wire up task completion callbacks
        self._task_skill.set_callbacks(self._on_complete, self._on_fail)

        # Add task context to cognition prompt
        task_context = TASK_PROMPT_ADDITION.format(
            task=task,
            max_cycles=self.max_cycles,
        )
        self.cognition.append_to_prompt(task_context)

        tools = self._get_tools()
        balance = self.budget
        total_cost = 0.0
        total_tokens = 0
        actions_taken = []

        self._log("START", f"Task: {task}")
        self._log("BUDGET", f"${balance:.4f} | Max cycles: {self.max_cycles}")
        self._log("TOOLS", f"{len(tools)} available")

        for cycle in range(1, self.max_cycles + 1):
            if self._completed or self._failed or balance <= 0:
                break

            # Build state
            est_cost_per_cycle = 0.01
            state = AgentState(
                balance=balance,
                burn_rate=est_cost_per_cycle,
                runway_hours=(balance / est_cost_per_cycle) * (self.cycle_delay / 3600) if est_cost_per_cycle > 0 else float("inf"),
                tools=tools,
                recent_actions=actions_taken[-10:],
                cycle=cycle,
                project_context=f"TASK: {task}\nCycles remaining: {self.max_cycles - cycle}",
            )

            # Think
            decision = await self.cognition.think(state)
            self._log("THINK", (decision.reasoning or "...")[:150])
            self._log("DO", f"{decision.action.tool} {decision.action.params}")

            # Execute
            result = await self._execute(decision.action)
            self._log("RESULT", str(result)[:200])

            # Track
            api_cost = decision.api_cost_usd
            total_cost += api_cost
            total_tokens += decision.token_usage.total_tokens()
            balance -= api_cost

            actions_taken.append({
                "cycle": cycle,
                "tool": decision.action.tool,
                "params": decision.action.params,
                "result": result,
                "api_cost_usd": api_cost,
                "tokens": decision.token_usage.total_tokens(),
            })

            if self.cycle_delay > 0 and not self._completed and not self._failed:
                await asyncio.sleep(self.cycle_delay)

        # Determine final status
        finished_at = datetime.now()
        duration = (finished_at - started_at).total_seconds()

        if self._completed:
            status = "completed"
        elif self._failed:
            status = "failed"
        elif balance <= 0:
            status = "budget_exhausted"
        else:
            status = "max_cycles"

        self._log("END", f"Status: {status} | Cost: ${total_cost:.4f} | Cycles: {len(actions_taken)}")

        # Clean up prompt addition
        if task_context in self.cognition._prompt_additions:
            self.cognition._prompt_additions.remove(task_context)

        return TaskResult(
            task=task,
            status=status,
            result_summary=self._result_summary,
            result_data=self._result_data,
            actions_taken=actions_taken,
            cycles_used=len(actions_taken),
            total_cost=total_cost,
            total_tokens=total_tokens,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_seconds=duration,
        )

    async def run_batch(self, tasks: List[str]) -> List[TaskResult]:
        """
        Execute multiple tasks sequentially.

        Args:
            tasks: List of task descriptions

        Returns:
            List of TaskResults, one per task
        """
        results = []
        for task in tasks:
            # Reset state between tasks
            self._completed = False
            self._failed = False
            self._result_summary = ""
            self._result_data = {}
            result = await self.run(task)
            results.append(result)
        return results
