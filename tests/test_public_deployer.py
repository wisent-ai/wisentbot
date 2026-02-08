"""Tests for PublicServiceDeployerSkill - Deploy agent services with public URLs."""
import json
import pytest
from unittest.mock import patch, MagicMock
from singularity.skills.public_deployer import (
    PublicServiceDeployerSkill,
    _load_deployments,
    _save_deployments,
    _allocate_port,
    _generate_caddy_config,
    _generate_nginx_config,
    _generate_docker_compose,
    DEPLOYMENTS_FILE,
)


@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.public_deployer.DATA_DIR", tmp_path)
    monkeypatch.setattr("singularity.skills.public_deployer.DEPLOYMENTS_FILE", tmp_path / "public_deployments.json")
    monkeypatch.setattr("singularity.skills.public_deployer.PROXY_CONFIG_DIR", tmp_path / "proxy_configs")
    return PublicServiceDeployerSkill()


def _mock_docker_success(*args, **kwargs):
    return {"success": True, "stdout": "abc123def456", "stderr": "", "returncode": 0}


def _mock_docker_fail(*args, **kwargs):
    return {"success": False, "stdout": "", "stderr": "Docker not found", "returncode": -1}


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_deploy_service(skill):
    r = await skill.execute("deploy", {
        "agent_name": "adam",
        "service_name": "Code Review",
        "docker_image": "ghcr.io/wisent-ai/adam-services:latest",
        "endpoints": [{"path": "/code_review", "method": "POST", "price_per_request": 0.05}],
    })
    assert r.success
    assert "adam.singularity.wisent.ai" in r.data["public_url"]
    assert r.data["status"] == "running"
    assert r.data["deployment_id"]


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_fail)
async def test_deploy_records_failure(skill):
    r = await skill.execute("deploy", {
        "agent_name": "adam",
        "service_name": "Test Service",
        "docker_image": "test:latest",
    })
    assert r.success  # Deployment is registered even if container fails
    assert r.data["status"] == "failed"


@pytest.mark.asyncio
async def test_deploy_requires_params(skill):
    r = await skill.execute("deploy", {})
    assert not r.success
    assert "required" in r.message.lower()


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_status_single(skill):
    dep = await skill.execute("deploy", {
        "agent_name": "adam", "service_name": "Svc", "docker_image": "img:v1",
    })
    dep_id = dep.data["deployment_id"]
    r = await skill.execute("status", {"deployment_id": dep_id})
    assert r.success
    assert r.data["deployment"]["status"] == "running"


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_status_all(skill):
    await skill.execute("deploy", {"agent_name": "a", "service_name": "s1", "docker_image": "i:1"})
    await skill.execute("deploy", {"agent_name": "b", "service_name": "s2", "docker_image": "i:2"})
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["total"] == 2


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_stop_deployment(skill):
    dep = await skill.execute("deploy", {
        "agent_name": "adam", "service_name": "Svc", "docker_image": "img:v1",
    })
    dep_id = dep.data["deployment_id"]
    r = await skill.execute("stop", {"deployment_id": dep_id})
    assert r.success
    s = await skill.execute("status", {"deployment_id": dep_id})
    assert s.data["deployment"]["status"] == "stopped"


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_redeploy(skill):
    dep = await skill.execute("deploy", {
        "agent_name": "adam", "service_name": "Svc", "docker_image": "img:v1",
    })
    dep_id = dep.data["deployment_id"]
    r = await skill.execute("redeploy", {"deployment_id": dep_id, "docker_image": "img:v2"})
    assert r.success
    assert r.data["new_image"] == "img:v2"


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_setup_billing(skill):
    dep = await skill.execute("deploy", {
        "agent_name": "adam", "service_name": "Svc", "docker_image": "img:v1",
        "endpoints": [{"path": "/review", "method": "POST", "price_per_request": 0.10}],
    })
    dep_id = dep.data["deployment_id"]
    r = await skill.execute("setup_billing", {
        "deployment_id": dep_id, "price_per_request": 0.05, "free_tier_requests": 20,
    })
    assert r.success
    assert r.data["billing"]["enabled"]
    assert r.data["billing"]["price_per_request"] == 0.05
    assert r.data["billing"]["free_tier_requests"] == 20


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_generate_caddy_config(skill):
    await skill.execute("deploy", {
        "agent_name": "adam", "service_name": "Svc", "docker_image": "img:v1",
    })
    r = await skill.execute("generate_routing_config", {"proxy_type": "caddy"})
    assert r.success
    assert "adam.singularity.wisent.ai" in r.data["config"]
    assert "reverse_proxy" in r.data["config"]


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_generate_nginx_config(skill):
    await skill.execute("deploy", {
        "agent_name": "adam", "service_name": "Svc", "docker_image": "img:v1",
    })
    r = await skill.execute("generate_routing_config", {"proxy_type": "nginx"})
    assert r.success
    assert "server_name" in r.data["config"]
    assert "proxy_pass" in r.data["config"]


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_deployment_stats(skill):
    await skill.execute("deploy", {"agent_name": "a", "service_name": "s1", "docker_image": "i:1"})
    await skill.execute("deploy", {"agent_name": "a", "service_name": "s2", "docker_image": "i:2"})
    r = await skill.execute("get_deployment_stats", {"agent_name": "a"})
    assert r.success
    assert r.data["total_deployments"] == 2
    assert r.data["running"] == 2


@pytest.mark.asyncio
@patch("singularity.skills.public_deployer._run_docker_cmd", _mock_docker_success)
async def test_custom_domain(skill):
    r = await skill.execute("deploy", {
        "agent_name": "adam", "service_name": "Svc", "docker_image": "img:v1",
        "custom_domain": "api.adam.dev",
    })
    assert r.success
    assert r.data["public_domain"] == "api.adam.dev"
    assert r.data["public_url"] == "https://api.adam.dev"


def test_allocate_port():
    data = {"port_allocations": {"a": 10000, "b": 10001}}
    assert _allocate_port(data) == 10002


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "public_service_deployer"
    assert len(m.actions) == 10
    action_names = [a.name for a in m.actions]
    assert "deploy" in action_names
    assert "setup_billing" in action_names
