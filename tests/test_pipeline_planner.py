"""Tests for PipelinePlannerSkill."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from singularity.skills.pipeline_planner import PipelinePlannerSkill, PIPELINE_PLANS_FILE


@pytest.fixture
def skill(tmp_path):
    """Create a PipelinePlannerSkill with temp data dir."""
    test_file = tmp_path / "pipeline_plans.json"
    with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", test_file):
        s = PipelinePlannerSkill()
        yield s


@pytest.fixture
def plans_file(tmp_path):
    """Create a fake PlannerSkill plans.json with goals/tasks."""
    plans_dir = tmp_path / "data"
    plans_dir.mkdir(exist_ok=True)
    plans_file = plans_dir / "plans.json"
    plans_file.write_text(json.dumps({
        "goals": [
            {
                "id": "goal-1",
                "title": "Deploy new service",
                "status": "active",
                "priority": "high",
                "tasks": [
                    {"id": "t1", "title": "Write code", "status": "pending", "effort": "medium", "skill_hint": "shell", "depends_on": []},
                    {"id": "t2", "title": "Run tests", "status": "pending", "effort": "small", "skill_hint": "shell", "depends_on": ["t1"]},
                    {"id": "t3", "title": "Create PR", "status": "pending", "effort": "small", "skill_hint": "github", "depends_on": ["t2"]},
                    {"id": "t4", "title": "Deploy", "status": "pending", "effort": "large", "skill_hint": "deployment", "depends_on": ["t3"]},
                    {"id": "t5", "title": "Completed task", "status": "completed", "effort": "small", "skill_hint": "shell", "depends_on": []},
                ],
            },
            {
                "id": "goal-2",
                "title": "Empty goal",
                "status": "active",
                "priority": "medium",
                "tasks": [],
            },
        ],
        "metadata": {},
    }))
    return plans_file


class TestManifest:
    def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "pipeline_planner"
        assert len(m.actions) == 8
        action_names = [a.name for a in m.actions]
        assert "generate" in action_names
        assert "optimize" in action_names
        assert "estimate" in action_names

    def test_check_credentials(self, skill):
        assert skill.check_credentials() is True


class TestGenerate:
    def test_generate_from_goal(self, skill, plans_file, tmp_path):
        """Test generating a pipeline from a goal's tasks."""
        with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", tmp_path / "pp.json"):
            skill._ensure_data()
            # Patch the plans.json path
            planner_path = plans_file
            with patch("singularity.skills.pipeline_planner.Path") as mock_path:
                mock_path.return_value = planner_path
                mock_path.__truediv__ = Path.__truediv__
                # Use the real method but override the file path
                original = skill._resolve_tasks_from_goal
                def patched_resolve(goal_id):
                    try:
                        with open(planner_path) as f:
                            data = json.load(f)
                        for g in data.get("goals", []):
                            if g["id"] == goal_id:
                                return g.get("tasks", [])
                    except Exception:
                        pass
                    return None
                skill._resolve_tasks_from_goal = patched_resolve
                result = skill.execute("generate", {"goal_id": "goal-1"})
                assert result.success
                pipeline = result.data["pipeline"]
                # 4 pending tasks + 1 progress check step
                assert len(pipeline) == 5
                assert result.data["task_count"] == 4
                assert "plan_id" in result.data

    def test_generate_missing_goal(self, skill, tmp_path):
        with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", tmp_path / "pp.json"):
            skill._ensure_data()
            result = skill.execute("generate", {"goal_id": "nonexistent"})
            assert not result.success
            assert "not found" in result.message

    def test_generate_no_goal_id(self, skill):
        result = skill.execute("generate", {})
        assert not result.success
        assert "required" in result.message


class TestGenerateFromTasks:
    def test_basic(self, skill, tmp_path):
        with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", tmp_path / "pp.json"):
            skill._ensure_data()
            tasks = [
                {"tool": "shell:run", "params": {"command": "echo hello"}, "title": "Say hello"},
                {"tool": "shell:run", "params": {"command": "echo world"}, "title": "Say world"},
            ]
            result = skill.execute("generate_from_tasks", {"tasks": tasks, "name": "test-pipe"})
            assert result.success
            assert len(result.data["pipeline"]) == 2
            assert result.data["pipeline"][0]["label"] == "Say hello"

    def test_empty_tasks(self, skill):
        result = skill.execute("generate_from_tasks", {"tasks": []})
        assert not result.success


class TestOptimize:
    def test_optimize_cost(self, skill):
        pipeline = [
            {"tool": "shell:run", "params": {}, "max_cost": 0.10, "timeout_seconds": 60, "required": True},
            {"tool": "shell:run", "params": {}, "max_cost": 0.05, "timeout_seconds": 30, "required": False},
        ]
        result = skill.execute("optimize", {"pipeline": pipeline, "strategy": "cost"})
        assert result.success
        opt = result.data["pipeline"]
        assert opt[0]["max_cost"] < 0.10  # Cost reduced
        assert "condition" in opt[1]  # Optional step now conditional

    def test_optimize_reliability(self, skill):
        pipeline = [
            {"tool": "shell:run", "params": {}, "required": True, "retry_count": 0, "label": "step1"},
        ]
        result = skill.execute("optimize", {"pipeline": pipeline, "strategy": "reliability"})
        assert result.success
        opt = result.data["pipeline"]
        assert opt[0]["retry_count"] >= 1
        assert "on_failure" in opt[0]

    def test_optimize_speed(self, skill):
        pipeline = [
            {"tool": "shell:run", "params": {}, "timeout_seconds": 60, "retry_count": 3},
        ]
        result = skill.execute("optimize", {"pipeline": pipeline, "strategy": "speed"})
        assert result.success
        opt = result.data["pipeline"]
        assert opt[0]["timeout_seconds"] <= 15
        assert opt[0]["retry_count"] == 0

    def test_optimize_empty(self, skill):
        result = skill.execute("optimize", {"pipeline": []})
        assert not result.success


class TestEstimate:
    def test_basic_estimate(self, skill):
        pipeline = [
            {"tool": "shell:run", "max_cost": 0.05, "timeout_seconds": 30, "required": True, "retry_count": 1},
            {"tool": "github:create_pr", "max_cost": 0.10, "timeout_seconds": 60, "required": False},
        ]
        result = skill.execute("estimate", {"pipeline": pipeline})
        assert result.success
        assert result.data["step_count"] == 2
        assert result.data["max_cost"] == 0.15
        assert result.data["required_steps"] == 1
        assert result.data["optional_steps"] == 1

    def test_empty_estimate(self, skill):
        result = skill.execute("estimate", {"pipeline": []})
        assert not result.success


class TestTemplates:
    def test_save_and_load(self, skill, tmp_path):
        with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", tmp_path / "pp.json"):
            skill._ensure_data()
            pipeline = [{"tool": "shell:run", "params": {"command": "echo hi"}}]
            save_result = skill.execute("save_template", {"name": "test-tmpl", "pipeline": pipeline, "description": "Test template"})
            assert save_result.success

            load_result = skill.execute("load_template", {"name": "test-tmpl"})
            assert load_result.success
            assert len(load_result.data["pipeline"]) == 1
            assert load_result.data["use_count"] == 1

    def test_load_missing(self, skill, tmp_path):
        with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", tmp_path / "pp.json"):
            skill._ensure_data()
            result = skill.execute("load_template", {"name": "nonexistent"})
            assert not result.success


class TestRecordOutcome:
    def test_record(self, skill, tmp_path):
        with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", tmp_path / "pp.json"):
            skill._ensure_data()
            # First create a plan
            data = skill._load()
            data["plans"].append({"id": "pp-test-123", "outcome": None})
            skill._save(data)

            result = skill.execute("record_outcome", {
                "plan_id": "pp-test-123",
                "success": True,
                "steps_succeeded": 3,
                "total_cost": 0.05,
            })
            assert result.success
            assert result.data["outcome"]["success"] is True

    def test_record_missing_plan(self, skill, tmp_path):
        with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", tmp_path / "pp.json"):
            skill._ensure_data()
            result = skill.execute("record_outcome", {"plan_id": "nonexistent", "success": True})
            assert not result.success


class TestStatus:
    def test_status(self, skill, tmp_path):
        with patch("singularity.skills.pipeline_planner.PIPELINE_PLANS_FILE", tmp_path / "pp.json"):
            skill._ensure_data()
            result = skill.execute("status", {})
            assert result.success
            assert "total_plans" in result.data
            assert "template_count" in result.data


class TestTopologicalSort:
    def test_dependency_ordering(self, skill):
        tasks = [
            {"id": "c", "title": "Deploy", "depends_on": ["b"]},
            {"id": "a", "title": "Write code", "depends_on": []},
            {"id": "b", "title": "Test", "depends_on": ["a"]},
        ]
        ordered = skill._topological_sort(tasks)
        ids = [t["id"] for t in ordered]
        assert ids.index("a") < ids.index("b")
        assert ids.index("b") < ids.index("c")

    def test_no_deps(self, skill):
        tasks = [
            {"id": "a", "title": "First", "depends_on": []},
            {"id": "b", "title": "Second", "depends_on": []},
        ]
        ordered = skill._topological_sort(tasks)
        assert len(ordered) == 2


class TestCostBudget:
    def test_within_budget(self, skill):
        steps = [{"max_cost": 0.05}, {"max_cost": 0.05}]
        result = skill._apply_cost_budget(steps, 0.50)
        assert result[0]["max_cost"] == 0.05  # Unchanged

    def test_over_budget_scales_down(self, skill):
        steps = [{"max_cost": 0.50}, {"max_cost": 0.50}]
        result = skill._apply_cost_budget(steps, 0.50)
        total = sum(s["max_cost"] for s in result)
        assert total <= 0.51  # Allow small float rounding


class TestUnknownAction:
    def test_unknown(self, skill):
        result = skill.execute("nonexistent_action", {})
        assert not result.success
        assert "Unknown" in result.message
