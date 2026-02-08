"""Tests for CircuitBreakerEventBridgeSkill."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from singularity.skills.circuit_breaker_event_bridge import (
    CircuitBreakerEventBridgeSkill,
    BRIDGE_DATA_FILE,
)


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """Create a bridge skill with temp data dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(
        "singularity.skills.circuit_breaker_event_bridge.BRIDGE_DATA_FILE",
        data_dir / "cb_event_bridge.json",
    )
    return CircuitBreakerEventBridgeSkill()


@pytest.fixture
def mock_registry():
    """Create a mock skill registry."""
    registry = AsyncMock()
    return registry


def _make_dashboard(circuits, budget_critical=False):
    """Helper to build dashboard data."""
    return {
        "circuits": circuits,
        "budget_critical": budget_critical,
    }


class TestManifest:
    def test_manifest_id(self, bridge):
        assert bridge.manifest.skill_id == "circuit_breaker_event_bridge"

    def test_manifest_actions(self, bridge):
        names = [a.name for a in bridge.manifest.actions]
        assert "sync" in names
        assert "configure" in names
        assert "status" in names
        assert "history" in names
        assert "reset" in names
        assert "emit_test" in names


class TestSync:
    @pytest.mark.asyncio
    async def test_sync_baselines_new_circuits(self, bridge, mock_registry):
        """First sync should baseline circuits without emitting events."""
        dashboard = _make_dashboard({
            "skill_a": {"state": "closed", "failure_rate": 0, "total_requests": 10},
            "skill_b": {"state": "open", "failure_rate": 0.8, "total_requests": 5},
        })
        mock_registry.execute_skill = AsyncMock(side_effect=[
            MagicMock(success=True, data=dashboard),  # dashboard call
        ])
        bridge._skill_registry = mock_registry

        result = await bridge.execute("sync", {})
        assert result.success
        assert result.data["circuits_checked"] == 2
        assert result.data["events_emitted"] == 0  # First sync = baseline only
        assert bridge._known_states["skill_a"] == "closed"
        assert bridge._known_states["skill_b"] == "open"

    @pytest.mark.asyncio
    async def test_sync_detects_transition(self, bridge, mock_registry):
        """Second sync should detect state changes and emit events."""
        # Baseline
        bridge._known_states = {"skill_a": "closed"}

        dashboard = _make_dashboard({
            "skill_a": {"state": "open", "failure_rate": 0.75, "total_requests": 20, "opened_count": 1},
        })

        event_result = MagicMock(success=True)
        mock_registry.execute_skill = AsyncMock(side_effect=[
            MagicMock(success=True, data=dashboard),  # dashboard
            event_result,  # event publish
        ])
        bridge._skill_registry = mock_registry

        result = await bridge.execute("sync", {})
        assert result.success
        assert len(result.data["transitions"]) == 1
        t = result.data["transitions"][0]
        assert t["skill_id"] == "skill_a"
        assert t["from_state"] == "closed"
        assert t["to_state"] == "open"
        assert result.data["events_emitted"] == 1

    @pytest.mark.asyncio
    async def test_sync_no_change_no_event(self, bridge, mock_registry):
        """No state change = no events."""
        bridge._known_states = {"skill_a": "closed"}
        dashboard = _make_dashboard({
            "skill_a": {"state": "closed", "failure_rate": 0},
        })
        mock_registry.execute_skill = AsyncMock(return_value=MagicMock(success=True, data=dashboard))
        bridge._skill_registry = mock_registry

        result = await bridge.execute("sync", {})
        assert result.success
        assert len(result.data["transitions"]) == 0
        assert result.data["events_emitted"] == 0

    @pytest.mark.asyncio
    async def test_sync_disabled(self, bridge):
        """Disabled bridge skips sync entirely."""
        bridge._config["enabled"] = False
        result = await bridge.execute("sync", {})
        assert result.success
        assert result.data["enabled"] is False

    @pytest.mark.asyncio
    async def test_sync_without_registry(self, bridge):
        """Sync without skill registry fails gracefully."""
        result = await bridge.execute("sync", {})
        assert not result.success


class TestConfigure:
    @pytest.mark.asyncio
    async def test_configure_updates(self, bridge):
        result = await bridge.execute("configure", {
            "emit_on_opened": False,
            "priority_opened": "critical",
        })
        assert result.success
        assert bridge._config["emit_on_opened"] is False
        assert bridge._config["priority_opened"] == "critical"
        assert "emit_on_opened" in result.data["updated"]

    @pytest.mark.asyncio
    async def test_configure_no_changes(self, bridge):
        result = await bridge.execute("configure", {"unknown_key": True})
        assert result.success
        assert len(result.data["updated"]) == 0


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_empty(self, bridge):
        result = await bridge.execute("status", {})
        assert result.success
        assert result.data["circuits_tracked"] == 0

    @pytest.mark.asyncio
    async def test_status_with_circuits(self, bridge):
        bridge._known_states = {"a": "closed", "b": "open", "c": "half_open"}
        result = await bridge.execute("status", {})
        assert result.success
        assert result.data["circuits_tracked"] == 3
        assert "b" in result.data["open_circuits"]
        assert "c" in result.data["half_open_circuits"]


class TestHistory:
    @pytest.mark.asyncio
    async def test_history_empty(self, bridge):
        result = await bridge.execute("history", {})
        assert result.success
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_history_filtered(self, bridge):
        bridge._transition_log = [
            {"skill_id": "a", "from_state": "closed", "to_state": "open"},
            {"skill_id": "b", "from_state": "closed", "to_state": "open"},
            {"skill_id": "a", "from_state": "open", "to_state": "half_open"},
        ]
        result = await bridge.execute("history", {"skill_id": "a"})
        assert result.data["count"] == 2


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_all(self, bridge):
        bridge._known_states = {"a": "open"}
        bridge._transition_log = [{"x": 1}]
        result = await bridge.execute("reset", {})
        assert result.success
        assert result.data["circuits_cleared"] == 1
        assert len(bridge._known_states) == 0


class TestEmitTest:
    @pytest.mark.asyncio
    async def test_emit_without_registry(self, bridge):
        result = await bridge.execute("emit_test", {})
        assert result.success
        assert result.data["emitted"] is False

    @pytest.mark.asyncio
    async def test_emit_with_registry(self, bridge, mock_registry):
        mock_registry.execute_skill = AsyncMock(return_value=MagicMock(success=True))
        bridge._skill_registry = mock_registry
        result = await bridge.execute("emit_test", {})
        assert result.success
        assert result.data["emitted"] is True


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load(self, bridge, tmp_path, monkeypatch):
        data_file = tmp_path / "data" / "cb_event_bridge.json"
        monkeypatch.setattr(
            "singularity.skills.circuit_breaker_event_bridge.BRIDGE_DATA_FILE",
            data_file,
        )
        bridge._known_states = {"skill_x": "open"}
        bridge._transition_log = [{"skill_id": "skill_x", "to_state": "open"}]
        bridge._save_state()

        bridge2 = CircuitBreakerEventBridgeSkill()
        # Need to point the new instance at same file
        monkeypatch.setattr(
            "singularity.skills.circuit_breaker_event_bridge.BRIDGE_DATA_FILE",
            data_file,
        )
        bridge2._load_state()
        assert bridge2._known_states["skill_x"] == "open"
        assert len(bridge2._transition_log) == 1
