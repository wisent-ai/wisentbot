"""Tests for EventDrivenWorkflowSkill."""

import pytest
import json
from pathlib import Path
from singularity.skills.event_driven_workflow import EventDrivenWorkflowSkill


@pytest.fixture
def skill(tmp_path):
    return EventDrivenWorkflowSkill(data_dir=str(tmp_path))


@pytest.fixture
def sample_steps():
    return [
        {"name": "Review Code", "skill_id": "revenue_services", "action": "code_review",
         "params": {"language": "python"}, "event_mapping": {"repository.name": "repo"}},
        {"name": "Summarize", "skill_id": "revenue_services", "action": "summarize",
         "input_mapping": {"step_1.summary": "text"}, "continue_on_failure": True},
    ]


@pytest.mark.asyncio
async def test_create_workflow(skill, sample_steps):
    result = await skill.execute("create_workflow", {
        "name": "code-review-pipeline",
        "description": "Review code on push",
        "steps": sample_steps,
        "event_bindings": [{"source": "webhook", "pattern": "github-push"}],
    })
    assert result.success
    assert result.data["steps_count"] == 2
    assert result.data["event_bindings"] == [{"source": "webhook", "pattern": "github-push"}]


@pytest.mark.asyncio
async def test_create_duplicate_fails(skill, sample_steps):
    await skill.execute("create_workflow", {"name": "dup", "steps": sample_steps})
    result = await skill.execute("create_workflow", {"name": "dup", "steps": sample_steps})
    assert not result.success
    assert "already exists" in result.message


@pytest.mark.asyncio
async def test_create_no_steps_fails(skill):
    result = await skill.execute("create_workflow", {"name": "empty", "steps": []})
    assert not result.success


@pytest.mark.asyncio
async def test_trigger_by_name(skill, sample_steps):
    await skill.execute("create_workflow", {"name": "my-wf", "steps": sample_steps})
    result = await skill.execute("trigger", {
        "workflow_name": "my-wf",
        "payload": {"repository": {"name": "singularity"}},
    })
    assert result.success
    assert result.data["workflows_triggered"] == 1
    assert result.data["results"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_trigger_by_event_matching(skill, sample_steps):
    await skill.execute("create_workflow", {
        "name": "on-push",
        "steps": sample_steps,
        "event_bindings": [{"source": "webhook", "pattern": "github-push"}],
    })
    result = await skill.execute("trigger", {
        "event_source": "webhook",
        "event_name": "github-push",
        "payload": {"repository": {"name": "test"}},
    })
    assert result.success
    assert result.data["workflows_triggered"] == 1


@pytest.mark.asyncio
async def test_wildcard_event_matching(skill, sample_steps):
    await skill.execute("create_workflow", {
        "name": "all-payment-events",
        "steps": sample_steps,
        "event_bindings": [{"source": "event_bus", "pattern": "payment.*"}],
    })
    result = await skill.execute("trigger", {
        "event_source": "event_bus", "event_name": "payment.received", "payload": {},
    })
    assert result.success
    assert result.data["workflows_triggered"] == 1


@pytest.mark.asyncio
async def test_no_match_returns_zero(skill, sample_steps):
    await skill.execute("create_workflow", {
        "name": "specific",
        "steps": sample_steps,
        "event_bindings": [{"source": "webhook", "pattern": "stripe-payment"}],
    })
    result = await skill.execute("trigger", {
        "event_source": "webhook", "event_name": "github-push", "payload": {},
    })
    assert result.success
    assert result.data["matched"] == 0


@pytest.mark.asyncio
async def test_step_condition_skips(skill):
    steps = [
        {"name": "Only PRs", "skill_id": "revenue_services", "action": "code_review",
         "condition": {"action": "opened"}},
    ]
    await skill.execute("create_workflow", {"name": "conditional", "steps": steps})
    result = await skill.execute("trigger", {
        "workflow_name": "conditional",
        "payload": {"action": "closed"},  # Doesn't match condition
    })
    assert result.success
    run_result = result.data["results"][0]
    assert run_result["status"] == "completed"


@pytest.mark.asyncio
async def test_list_workflows(skill, sample_steps):
    await skill.execute("create_workflow", {"name": "wf1", "steps": sample_steps})
    await skill.execute("create_workflow", {"name": "wf2", "steps": sample_steps})
    result = await skill.execute("list_workflows", {})
    assert result.success
    assert result.data["total"] == 2


@pytest.mark.asyncio
async def test_update_workflow(skill, sample_steps):
    await skill.execute("create_workflow", {"name": "updatable", "steps": sample_steps})
    result = await skill.execute("update_workflow", {
        "name": "updatable", "enabled": False, "description": "disabled",
    })
    assert result.success
    assert any("enabled" in f for f in result.data["updated_fields"])
    # Verify disabled workflow can't be triggered
    trigger = await skill.execute("trigger", {"workflow_name": "updatable", "payload": {}})
    assert not trigger.success


@pytest.mark.asyncio
async def test_delete_workflow(skill, sample_steps):
    await skill.execute("create_workflow", {"name": "deleteme", "steps": sample_steps})
    result = await skill.execute("delete_workflow", {"name": "deleteme"})
    assert result.success
    get_result = await skill.execute("get_workflow", {"name": "deleteme"})
    assert not get_result.success


@pytest.mark.asyncio
async def test_get_runs(skill, sample_steps):
    await skill.execute("create_workflow", {"name": "tracked", "steps": sample_steps})
    await skill.execute("trigger", {"workflow_name": "tracked", "payload": {}})
    await skill.execute("trigger", {"workflow_name": "tracked", "payload": {}})
    result = await skill.execute("get_runs", {"workflow_name": "tracked"})
    assert result.success
    assert result.data["total"] == 2


@pytest.mark.asyncio
async def test_stats(skill, sample_steps):
    await skill.execute("create_workflow", {"name": "stats-wf", "steps": sample_steps})
    await skill.execute("trigger", {"workflow_name": "stats-wf", "payload": {}})
    result = await skill.execute("stats", {})
    assert result.success
    assert result.data["total_workflows"] == 1
    assert result.data["total_runs"] == 1


@pytest.mark.asyncio
async def test_persistence(tmp_path, sample_steps):
    skill1 = EventDrivenWorkflowSkill(data_dir=str(tmp_path))
    await skill1.execute("create_workflow", {"name": "persistent", "steps": sample_steps})
    skill2 = EventDrivenWorkflowSkill(data_dir=str(tmp_path))
    result = await skill2.execute("get_workflow", {"name": "persistent"})
    assert result.success
    assert result.data["name"] == "persistent"
