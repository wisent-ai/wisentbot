"""Tests for ServerlessServiceHostingBridgeSkill."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from singularity.skills.serverless_service_hosting_bridge import (
    ServerlessServiceHostingBridgeSkill,
    BRIDGE_FILE,
)


@pytest.fixture(autouse=True)
def clean_bridge_file():
    if BRIDGE_FILE.exists():
        BRIDGE_FILE.unlink()
    yield
    if BRIDGE_FILE.exists():
        BRIDGE_FILE.unlink()


@pytest.fixture
def bridge():
    return ServerlessServiceHostingBridgeSkill()


def deploy_params(fn_id="fn-abc123", name="my_func", agent="agent-1", route="/api/test"):
    return {
        "function_id": fn_id, "function_name": name,
        "agent_name": agent, "route": route,
        "methods": ["GET", "POST"], "price_per_call": 0.05,
        "description": "Test function",
    }


@pytest.mark.asyncio
async def test_on_deploy_registers_service(bridge):
    result = await bridge.execute("on_deploy", deploy_params())
    assert result.success
    assert "auto-registered" in result.message
    assert result.data["service_id"].startswith("svc-fn-")
    assert len(result.data["endpoints"]) == 2  # GET + POST


@pytest.mark.asyncio
async def test_on_deploy_duplicate_rejected(bridge):
    await bridge.execute("on_deploy", deploy_params())
    result = await bridge.execute("on_deploy", deploy_params())
    assert not result.success
    assert "already registered" in result.message


@pytest.mark.asyncio
async def test_on_deploy_auto_register_disabled(bridge):
    await bridge.execute("configure", {"auto_register": False})
    result = await bridge.execute("on_deploy", deploy_params())
    assert result.success
    assert "disabled" in result.message.lower()


@pytest.mark.asyncio
async def test_on_deploy_missing_params(bridge):
    result = await bridge.execute("on_deploy", {"function_id": "fn-x"})
    assert not result.success
    assert "Required" in result.message


@pytest.mark.asyncio
async def test_on_remove_deregisters(bridge):
    await bridge.execute("on_deploy", deploy_params())
    result = await bridge.execute("on_remove", {"function_id": "fn-abc123"})
    assert result.success
    assert "deregistered" in result.message.lower()
    # Verify binding removed
    dash = await bridge.execute("dashboard", {})
    assert dash.data["total_bindings"] == 0


@pytest.mark.asyncio
async def test_on_remove_no_binding(bridge):
    result = await bridge.execute("on_remove", {"function_id": "fn-nonexistent"})
    assert not result.success


@pytest.mark.asyncio
async def test_on_remove_auto_deregister_disabled(bridge):
    await bridge.execute("on_deploy", deploy_params())
    await bridge.execute("configure", {"auto_deregister": False})
    result = await bridge.execute("on_remove", {"function_id": "fn-abc123"})
    assert result.success
    assert "orphaned" in result.message.lower()


@pytest.mark.asyncio
async def test_on_status_change_syncs(bridge):
    await bridge.execute("on_deploy", deploy_params())
    result = await bridge.execute("on_status_change", {"function_id": "fn-abc123", "new_status": "disabled"})
    assert result.success
    assert "stopped" in result.data["new_status"]


@pytest.mark.asyncio
async def test_on_status_change_invalid(bridge):
    await bridge.execute("on_deploy", deploy_params())
    result = await bridge.execute("on_status_change", {"function_id": "fn-abc123", "new_status": "invalid"})
    assert not result.success


@pytest.mark.asyncio
async def test_sync_all_with_functions(bridge):
    functions = [
        {"id": "fn-001", "name": "func1", "agent_name": "agent-1", "route": "/f1", "status": "active"},
        {"id": "fn-002", "name": "func2", "agent_name": "agent-2", "route": "/f2", "status": "active"},
        {"id": "fn-003", "name": "func3", "agent_name": "agent-1", "route": "/f3", "status": "disabled"},
    ]
    result = await bridge.execute("sync_all", {"functions": functions})
    assert result.success
    assert len(result.data["registered"]) == 2  # fn-003 skipped (disabled)
    assert len(result.data["skipped"]) == 1


@pytest.mark.asyncio
async def test_sync_all_dry_run(bridge):
    functions = [{"id": "fn-001", "name": "f1", "agent_name": "a1", "route": "/r1", "status": "active"}]
    result = await bridge.execute("sync_all", {"functions": functions, "dry_run": True})
    assert result.success
    assert "DRY RUN" in result.message
    # No actual bindings created
    dash = await bridge.execute("dashboard", {})
    assert dash.data["total_bindings"] == 0


@pytest.mark.asyncio
async def test_unsync_removes_binding(bridge):
    await bridge.execute("on_deploy", deploy_params())
    result = await bridge.execute("unsync", {"function_id": "fn-abc123"})
    assert result.success
    assert result.data["binding_removed"]


@pytest.mark.asyncio
async def test_dashboard(bridge):
    await bridge.execute("on_deploy", deploy_params("fn-1", "f1", "agent-1", "/a"))
    await bridge.execute("on_deploy", deploy_params("fn-2", "f2", "agent-2", "/b"))
    result = await bridge.execute("dashboard", {})
    assert result.success
    assert result.data["total_bindings"] == 2
    assert result.data["grade"] == "A"
    assert "agent-1" in result.data["by_agent"]


@pytest.mark.asyncio
async def test_revenue(bridge):
    await bridge.execute("on_deploy", deploy_params())
    result = await bridge.execute("revenue", {})
    assert result.success
    assert "total_revenue" in result.data


@pytest.mark.asyncio
async def test_configure(bridge):
    result = await bridge.execute("configure", {"auto_register": False, "default_port": 9090})
    assert result.success
    assert result.data["config"]["auto_register"] is False
    assert result.data["config"]["default_port"] == 9090


@pytest.mark.asyncio
async def test_status(bridge):
    result = await bridge.execute("status", {})
    assert result.success
    assert "config" in result.data
    assert "stats" in result.data


@pytest.mark.asyncio
async def test_unknown_action(bridge):
    result = await bridge.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message
