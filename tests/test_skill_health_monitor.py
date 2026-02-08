"""Tests for SkillHealthMonitorSkill."""
import pytest
from unittest.mock import MagicMock, AsyncMock

from singularity.skills.skill_health_monitor import (
    SkillHealthMonitorSkill, HEALTHY, DEGRADED, UNHEALTHY,
)
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


# ── Helper: Fake skill for testing ──────────────────────────────────

class FakeHealthySkill(Skill):
    def __init__(self):
        super().__init__({})

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="fake_healthy", name="Fake Healthy", version="1.0.0",
            category="test", description="A healthy test skill",
            actions=self.get_actions(), required_credentials=[],
        )

    def get_actions(self):
        return [
            SkillAction(name="status", description="Get status", parameters={}),
            SkillAction(name="do_thing", description="Do thing", parameters={}),
        ]

    async def execute(self, action, params):
        if action == "status":
            return SkillResult(success=True, message="OK", data={"healthy": True})
        return SkillResult(success=True, message="Done")


class FakeBrokenSkill(Skill):
    @property
    def manifest(self):
        raise RuntimeError("Manifest is broken!")

    def get_actions(self):
        return []

    async def execute(self, action, params):
        raise RuntimeError("Everything is broken")


class FakeNoActionsSkill(Skill):
    def __init__(self):
        super().__init__({})

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="no_actions", name="No Actions", version="1.0.0",
            category="test", description="Skill with no actions",
            actions=[], required_credentials=[],
        )

    def get_actions(self):
        return []

    async def execute(self, action, params):
        return SkillResult(success=False, message="No actions")


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def monitor(tmp_path):
    """Create a SkillHealthMonitorSkill with temp storage."""
    skill = SkillHealthMonitorSkill()
    # Override data file to use tmp
    import singularity.skills.skill_health_monitor as mod
    mod.HEALTH_FILE = tmp_path / "health.json"
    mod.DATA_DIR = tmp_path
    skill._store = None
    return skill


@pytest.fixture
def monitor_with_context(monitor):
    """Monitor with a mock context containing fake skills."""
    ctx = MagicMock()
    registry = MagicMock()
    registry._skills = {
        "fake_healthy": FakeHealthySkill(),
        "fake_broken": FakeBrokenSkill(),
        "no_actions": FakeNoActionsSkill(),
    }
    ctx._registry = registry
    ctx.call_skill = AsyncMock(return_value=SkillResult(success=True, message="OK"))
    monitor.context = ctx
    return monitor


# ── Tests ───────────────────────────────────────────────────────────

def test_manifest(monitor):
    m = monitor.manifest
    assert m.skill_id == "skill_health_monitor"
    assert len(monitor.get_actions()) == 8


def test_check_healthy_skill(monitor):
    skill = FakeHealthySkill()
    report = monitor._check_skill_health(skill)
    assert report["state"] == HEALTHY
    assert "manifest_access" in report["checks_passed"]
    assert "actions_list" in report["checks_passed"]
    assert report["action_count"] == 2


def test_check_broken_skill(monitor):
    skill = FakeBrokenSkill()
    report = monitor._check_skill_health(skill)
    assert report["state"] == UNHEALTHY
    assert "manifest_access" in report["checks_failed"]


def test_check_no_actions_skill(monitor):
    skill = FakeNoActionsSkill()
    report = monitor._check_skill_health(skill)
    assert report["state"] == DEGRADED
    assert "actions_list" in report["checks_failed"]


@pytest.mark.asyncio
async def test_check_all(monitor_with_context):
    result = await monitor_with_context.execute("check_all", {})
    assert result.success
    assert result.data["total"] == 3
    healthy = [s for s in result.data["healthy"] if s["state"] == HEALTHY]
    assert len(healthy) == 1
    assert healthy[0]["skill_id"] == "fake_healthy"
    assert len(result.data["unhealthy"]) == 1
    assert result.data["unhealthy"][0]["skill_id"] == "fake_broken"


@pytest.mark.asyncio
async def test_check_one(monitor_with_context):
    result = await monitor_with_context.execute("check_one", {"skill_id": "fake_healthy"})
    assert result.success
    assert result.data["state"] == HEALTHY
    assert result.data["skill_id"] == "fake_healthy"


@pytest.mark.asyncio
async def test_check_one_not_found(monitor_with_context):
    result = await monitor_with_context.execute("check_one", {"skill_id": "nonexistent"})
    assert not result.success


@pytest.mark.asyncio
async def test_probe_healthy_skill(monitor):
    skill = FakeHealthySkill()
    probe = await monitor._probe_skill(skill, "fake_healthy")
    assert probe["probed"]
    assert probe["probe_success"]
    assert probe["probe_action"] == "status"
    assert probe["probe_duration_ms"] >= 0


@pytest.mark.asyncio
async def test_status_after_check(monitor_with_context):
    await monitor_with_context.execute("check_all", {})
    result = await monitor_with_context.execute("status", {})
    assert result.success
    assert len(result.data["skills"]) == 3
    assert result.data["counts"][HEALTHY] == 1


@pytest.mark.asyncio
async def test_degraded_filter(monitor_with_context):
    await monitor_with_context.execute("check_all", {})
    result = await monitor_with_context.execute("degraded", {})
    assert result.success
    ids = [p["skill_id"] for p in result.data["problems"]]
    assert "fake_broken" in ids
    assert "fake_healthy" not in ids


@pytest.mark.asyncio
async def test_history(monitor_with_context):
    await monitor_with_context.execute("check_all", {})
    result = await monitor_with_context.execute("history", {"skill_id": "fake_healthy"})
    assert result.success
    assert len(result.data["checks"]) == 1


@pytest.mark.asyncio
async def test_stats(monitor_with_context):
    await monitor_with_context.execute("check_all", {})
    result = await monitor_with_context.execute("stats", {})
    assert result.success
    assert result.data["total_skills"] == 3
    assert result.data["health_percentage"] > 0


@pytest.mark.asyncio
async def test_configure(monitor):
    result = await monitor.execute("configure", {"default_timeout": 5.0, "auto_emit_metrics": False})
    assert result.success
    assert "default_timeout" in result.data["changed"]
    store = monitor._load()
    assert store["config"]["default_timeout"] == 5.0
    assert store["config"]["auto_emit_metrics"] is False


@pytest.mark.asyncio
async def test_no_skills_in_registry(monitor):
    ctx = MagicMock()
    registry = MagicMock()
    registry._skills = {}
    ctx._registry = registry
    monitor.context = ctx
    result = await monitor.execute("check_all", {})
    assert result.success
    assert result.data["total"] == 0


@pytest.mark.asyncio
async def test_check_all_with_probe(monitor_with_context):
    result = await monitor_with_context.execute("check_all", {"probe": True})
    assert result.success
    assert result.data["probed"]
