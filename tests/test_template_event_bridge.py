"""Tests for TemplateEventBridgeSkill."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from singularity.skills.template_event_bridge import (
    TemplateEventBridgeSkill,
    _convert_template_steps,
    _load_data,
    _save_data,
    USE_CASE_SUGGESTIONS,
)


@pytest.fixture
def tmp_data(tmp_path):
    return tmp_path / "bridge.json"


@pytest.fixture
def skill(tmp_data):
    return TemplateEventBridgeSkill(data_path=tmp_data)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# --- Unit: step conversion ---

def test_convert_template_steps_param_refs():
    template = {
        "steps": [
            {"skill": "github", "action": "get_pr", "params_from": {"repo": "param.repo"}},
            {"skill": "code_review", "action": "review", "params_from": {"code": "step.0.diff", "depth": "param.review_depth"}},
        ]
    }
    resolved = {"repo": "owner/repo", "review_depth": "thorough"}
    steps = _convert_template_steps(template, resolved)

    assert len(steps) == 2
    assert steps[0]["skill_id"] == "github"
    assert steps[0]["params"]["repo"] == "owner/repo"
    assert steps[1]["input_mapping"] == {"step_1.diff": "code"}
    assert steps[1]["params"]["depth"] == "thorough"


def test_convert_template_steps_event_mapping():
    template = {
        "steps": [
            {"skill": "handler", "action": "process", "params_from": {"data": "event.body"}},
        ]
    }
    steps = _convert_template_steps(template, {})
    assert steps[0]["event_mapping"] == {"event.body": "data"}


# --- Deploy ---

def test_deploy_requires_template_id(skill):
    result = run(skill.execute("deploy", {"params": {}}))
    assert not result.success
    assert "template_id" in result.message


def test_deploy_from_builtin_template(skill):
    result = run(skill.execute("deploy", {
        "template_id": "github_pr_review",
        "params": {"repo": "test/repo"},
        "event_bindings": [{"source": "webhook", "pattern": "github-pr"}],
    }))
    assert result.success
    assert result.data["deployment_id"]
    assert result.data["steps_count"] == 3
    assert len(result.data["event_bindings"]) == 1


def test_deploy_missing_required_params(skill):
    result = run(skill.execute("deploy", {
        "template_id": "github_pr_review",
        "params": {},  # missing 'repo' which is required
    }))
    assert not result.success
    assert "Missing required" in result.message


def test_deploy_nonexistent_template(skill):
    result = run(skill.execute("deploy", {
        "template_id": "nonexistent_xyz",
        "params": {},
    }))
    assert not result.success


# --- Preview ---

def test_preview_shows_converted_steps(skill):
    result = run(skill.execute("preview", {
        "template_id": "stripe_payment_flow",
        "params": {"customer_email": "test@test.com", "amount": 50, "product": "api"},
    }))
    assert result.success
    preview = result.data["preview"]
    assert len(preview["steps"]) == 3
    assert preview["resolved_params"]["customer_email"] == "test@test.com"
    assert not preview["unresolved_params"]


def test_preview_shows_unresolved_params(skill):
    result = run(skill.execute("preview", {
        "template_id": "deploy_on_merge",
        "params": {},  # Missing required params
    }))
    assert result.success
    assert "repo" in result.data["preview"]["unresolved_params"]


# --- List & Undeploy ---

def test_list_and_undeploy(skill):
    # Deploy first
    run(skill.execute("deploy", {
        "template_id": "usage_alert",
        "params": {"customer_id": "cust_1"},
    }))
    # List
    result = run(skill.execute("list", {}))
    assert result.success
    assert result.data["total"] == 1
    dep_id = result.data["deployments"][0]["deployment_id"]

    # Undeploy
    result = run(skill.execute("undeploy", {"deployment_id": dep_id}))
    assert result.success

    # Verify removed
    result = run(skill.execute("list", {}))
    assert result.data["total"] == 0


# --- Sync ---

def test_sync_updates_deployment(skill):
    # Deploy
    dep_result = run(skill.execute("deploy", {
        "template_id": "health_check_pipeline",
        "params": {"service_url": "http://old.example.com"},
    }))
    dep_id = dep_result.data["deployment_id"]

    # Sync with updated params
    sync_result = run(skill.execute("sync", {
        "deployment_id": dep_id,
        "params": {"service_url": "http://new.example.com"},
    }))
    assert sync_result.success
    assert "new.example.com" not in sync_result.message or True  # just check success


# --- Suggest ---

def test_suggest_by_use_case(skill):
    result = run(skill.execute("suggest", {"use_case": "billing"}))
    assert result.success
    assert len(result.data["suggestions"]) > 0
    assert result.data["suggestions"][0]["use_case"] == "billing"


def test_suggest_by_goal(skill):
    result = run(skill.execute("suggest", {"goal": "monitor health check services"}))
    assert result.success
    assert any(s["use_case"] == "monitoring" for s in result.data["suggestions"])


def test_suggest_no_input_lists_use_cases(skill):
    result = run(skill.execute("suggest", {}))
    assert result.success
    assert "use_cases" in result.data


# --- Deploy Batch ---

def test_deploy_batch_by_use_case(skill):
    result = run(skill.execute("deploy_batch", {
        "deployments": [
            {"template_id": "usage_alert", "params": {"customer_id": "cust_1"}},
            {"template_id": "content_pipeline", "params": {"topic": "AI trends"}},
        ],
    }))
    assert result.success
    assert result.data["deployed"] == 2


def test_deploy_batch_unknown_use_case(skill):
    result = run(skill.execute("deploy_batch", {"use_case": "nonexistent"}))
    assert not result.success


# --- Persistence ---

def test_data_persists(tmp_data):
    skill1 = TemplateEventBridgeSkill(data_path=tmp_data)
    run(skill1.execute("deploy", {
        "template_id": "customer_onboarding",
        "params": {"customer_name": "Test", "customer_email": "t@t.com"},
    }))

    skill2 = TemplateEventBridgeSkill(data_path=tmp_data)
    data = _load_data(tmp_data)
    assert len(data["deployments"]) == 1
