"""Tests for TaskRunner - task-oriented agent execution."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.task_runner import TaskRunner, TaskResult, TaskControlSkill, TASK_PROMPT_ADDITION
from singularity.cognition import Decision, Action, TokenUsage
from singularity.skills.base import Skill, SkillManifest, SkillAction, SkillResult, SkillRegistry


# --- TaskControlSkill tests ---

class TestTaskControlSkill:
    def setup_method(self):
        self.skill = TaskControlSkill()

    def test_manifest(self):
        m = self.skill.manifest
        assert m.skill_id == "task"
        assert len(m.actions) == 3
        action_names = [a.name for a in m.actions]
        assert "done" in action_names
        assert "fail" in action_names
        assert "progress" in action_names

    def test_check_credentials(self):
        assert self.skill.check_credentials() is True

    @pytest.mark.asyncio
    async def test_done_action(self):
        called = {}
        self.skill.set_callbacks(
            on_complete=lambda s, d: called.update({"summary": s, "data": d}),
            on_fail=lambda r: None,
        )
        result = await self.skill.execute("done", {"summary": "All done", "data": {"key": "val"}})
        assert result.success is True
        assert called["summary"] == "All done"
        assert called["data"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_fail_action(self):
        called = {}
        self.skill.set_callbacks(
            on_complete=lambda s, d: None,
            on_fail=lambda r: called.update({"reason": r}),
        )
        result = await self.skill.execute("fail", {"reason": "Cannot access API"})
        assert result.success is True
        assert called["reason"] == "Cannot access API"

    @pytest.mark.asyncio
    async def test_progress_action(self):
        result = await self.skill.execute("progress", {"status": "Working on it", "percent": 50})
        assert result.success is True
        assert "50%" in result.message

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await self.skill.execute("nope", {})
        assert result.success is False


# --- TaskResult tests ---

class TestTaskResult:
    def test_to_dict(self):
        r = TaskResult(
            task="Test task",
            status="completed",
            result_summary="Done",
            actions_taken=[{"tool": "x"}],
            total_cost=0.05,
        )
        d = r.to_dict()
        assert d["task"] == "Test task"
        assert d["status"] == "completed"
        assert d["actions_count"] == 1
        assert d["total_cost"] == 0.05


# --- TaskRunner tests ---

class TestTaskRunner:
    @pytest.mark.asyncio
    async def test_task_completion(self):
        """Agent calls task:done -> runner returns completed."""
        with patch("singularity.task_runner.CognitionEngine") as MockCog:
            mock_cog = MockCog.return_value
            mock_cog._prompt_additions = []
            mock_cog.llm_type = "none"
            mock_cog.llm_model = "test"
            mock_cog._anthropic_api_key = ""
            mock_cog._openai_api_key = ""
            mock_cog._openai_base_url = ""

            call_count = 0
            async def mock_think(state):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    return Decision(
                        action=Action(tool="task:done", params={"summary": "Created file"}),
                        reasoning="Task is done",
                        token_usage=TokenUsage(100, 50),
                        api_cost_usd=0.001,
                    )
                return Decision(
                    action=Action(tool="wait", params={}),
                    reasoning="Thinking",
                    token_usage=TokenUsage(100, 50),
                    api_cost_usd=0.001,
                )

            mock_cog.think = mock_think
            mock_cog.append_to_prompt = MagicMock()

            runner = TaskRunner(llm_provider="none", budget=1.0, max_cycles=10, cycle_delay=0, quiet=True)
            result = await runner.run("Create a hello world file")

            assert result.status == "completed"
            assert result.result_summary == "Created file"
            assert result.cycles_used == 2
            assert result.total_cost > 0

    @pytest.mark.asyncio
    async def test_task_failure(self):
        """Agent calls task:fail -> runner returns failed."""
        with patch("singularity.task_runner.CognitionEngine") as MockCog:
            mock_cog = MockCog.return_value
            mock_cog._prompt_additions = []
            mock_cog.llm_type = "none"
            mock_cog.llm_model = "test"
            mock_cog._anthropic_api_key = ""
            mock_cog._openai_api_key = ""
            mock_cog._openai_base_url = ""

            async def mock_think(state):
                return Decision(
                    action=Action(tool="task:fail", params={"reason": "No access"}),
                    reasoning="Cannot proceed",
                    token_usage=TokenUsage(100, 50),
                    api_cost_usd=0.001,
                )

            mock_cog.think = mock_think
            mock_cog.append_to_prompt = MagicMock()

            runner = TaskRunner(llm_provider="none", budget=1.0, max_cycles=10, cycle_delay=0, quiet=True)
            result = await runner.run("Access admin panel")

            assert result.status == "failed"
            assert "No access" in result.result_summary

    @pytest.mark.asyncio
    async def test_max_cycles_reached(self):
        """Runner stops after max_cycles without completion."""
        with patch("singularity.task_runner.CognitionEngine") as MockCog:
            mock_cog = MockCog.return_value
            mock_cog._prompt_additions = []
            mock_cog.llm_type = "none"
            mock_cog.llm_model = "test"

            async def mock_think(state):
                return Decision(
                    action=Action(tool="wait", params={}),
                    reasoning="Still working",
                    token_usage=TokenUsage(10, 5),
                    api_cost_usd=0.0001,
                )

            mock_cog.think = mock_think
            mock_cog.append_to_prompt = MagicMock()

            runner = TaskRunner(llm_provider="none", budget=1.0, max_cycles=3, cycle_delay=0, quiet=True)
            result = await runner.run("Infinite task")

            assert result.status == "max_cycles"
            assert result.cycles_used == 3

    @pytest.mark.asyncio
    async def test_budget_exhausted(self):
        """Runner stops when budget runs out."""
        with patch("singularity.task_runner.CognitionEngine") as MockCog:
            mock_cog = MockCog.return_value
            mock_cog._prompt_additions = []
            mock_cog.llm_type = "none"
            mock_cog.llm_model = "test"

            async def mock_think(state):
                return Decision(
                    action=Action(tool="wait", params={}),
                    reasoning="Working",
                    token_usage=TokenUsage(100, 50),
                    api_cost_usd=0.50,
                )

            mock_cog.think = mock_think
            mock_cog.append_to_prompt = MagicMock()

            runner = TaskRunner(llm_provider="none", budget=0.60, max_cycles=100, cycle_delay=0, quiet=True)
            result = await runner.run("Expensive task")

            assert result.status == "budget_exhausted"
            assert result.cycles_used <= 2

    @pytest.mark.asyncio
    async def test_tools_include_task_control(self):
        """Task control skill is always available."""
        with patch("singularity.task_runner.CognitionEngine") as MockCog:
            mock_cog = MockCog.return_value
            mock_cog._prompt_additions = []
            mock_cog.llm_type = "none"

            runner = TaskRunner(llm_provider="none", quiet=True)
            tools = runner._get_tools()
            tool_names = [t["name"] for t in tools]
            assert "task:done" in tool_names
            assert "task:fail" in tool_names
            assert "task:progress" in tool_names

    @pytest.mark.asyncio
    async def test_run_batch(self):
        """Run multiple tasks sequentially."""
        with patch("singularity.task_runner.CognitionEngine") as MockCog:
            mock_cog = MockCog.return_value
            mock_cog._prompt_additions = []
            mock_cog.llm_type = "none"
            mock_cog.llm_model = "test"

            async def mock_think(state):
                return Decision(
                    action=Action(tool="task:done", params={"summary": f"Done cycle {state.cycle}"}),
                    reasoning="Complete",
                    token_usage=TokenUsage(10, 5),
                    api_cost_usd=0.001,
                )

            mock_cog.think = mock_think
            mock_cog.append_to_prompt = MagicMock()

            runner = TaskRunner(llm_provider="none", budget=1.0, max_cycles=5, cycle_delay=0, quiet=True)
            results = await runner.run_batch(["Task A", "Task B"])

            assert len(results) == 2
            assert results[0].status == "completed"
            assert results[1].status == "completed"
