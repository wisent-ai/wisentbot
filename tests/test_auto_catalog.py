"""Tests for AutoCatalogSkill - auto-registration and health monitoring of marketplace services."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from singularity.skills.auto_catalog import AutoCatalogSkill, SERVICE_CATALOG, CATALOG_FILE
from singularity.skills.marketplace import MarketplaceSkill
from singularity.skills.revenue_services import RevenueServiceSkill
from singularity.skills.base import SkillResult, SkillContext, SkillRegistry, SkillAction


@pytest.fixture
def catalog_skill(tmp_path):
    test_file = tmp_path / "auto_catalog.json"
    with patch("singularity.skills.auto_catalog.CATALOG_FILE", test_file):
        s = AutoCatalogSkill()
        yield s


@pytest.fixture
def full_setup(tmp_path):
    """Setup with marketplace, revenue_services, and auto_catalog all wired together."""
    catalog_file = tmp_path / "auto_catalog.json"
    marketplace_file = tmp_path / "marketplace.json"
    with patch("singularity.skills.auto_catalog.CATALOG_FILE", catalog_file), \
         patch("singularity.skills.marketplace.MARKETPLACE_FILE", marketplace_file):
        registry = SkillRegistry()
        marketplace = MarketplaceSkill()
        revenue = RevenueServiceSkill()
        revenue._execution_log = []
        catalog = AutoCatalogSkill()

        registry.skills = {
            "marketplace": marketplace,
            "revenue_services": revenue,
            "auto_catalog": catalog,
        }
        ctx = SkillContext(registry=registry, agent_name="TestAgent")

        marketplace.set_context(ctx)
        revenue.set_context(ctx)
        catalog.set_context(ctx)

        yield {"catalog": catalog, "marketplace": marketplace, "revenue": revenue, "ctx": ctx}


@pytest.mark.asyncio
async def test_manifest(catalog_skill):
    m = catalog_skill.manifest
    assert m.skill_id == "auto_catalog"
    assert m.category == "revenue"
    actions = [a.name for a in m.actions]
    assert "auto_register" in actions
    assert "health_check" in actions
    assert "sync_catalog" in actions
    assert "catalog_status" in actions


@pytest.mark.asyncio
async def test_auto_register_no_context(catalog_skill):
    result = await catalog_skill.execute("auto_register", {})
    assert not result.success
    assert "context" in result.message.lower()


@pytest.mark.asyncio
async def test_auto_register_services(full_setup):
    catalog = full_setup["catalog"]
    result = await catalog.execute("auto_register", {})
    assert result.success
    assert result.data["total_registered"] == len(SERVICE_CATALOG)
    for svc in result.data["registered"]:
        assert svc["price"] > 0
        assert svc["skill_id"] == "revenue_services"


@pytest.mark.asyncio
async def test_auto_register_dry_run(full_setup):
    catalog = full_setup["catalog"]
    result = await catalog.execute("auto_register", {"dry_run": True})
    assert result.success
    assert "dry run" in result.message.lower()
    for svc in result.data["registered"]:
        assert svc.get("dry_run") is True


@pytest.mark.asyncio
async def test_auto_register_skips_duplicates(full_setup):
    catalog = full_setup["catalog"]
    r1 = await catalog.execute("auto_register", {})
    assert r1.success
    first_count = r1.data["total_registered"]
    r2 = await catalog.execute("auto_register", {})
    assert r2.success
    assert r2.data["total_registered"] == 0
    assert len(r2.data["skipped"]) >= first_count


@pytest.mark.asyncio
async def test_health_check(full_setup):
    catalog = full_setup["catalog"]
    await catalog.execute("auto_register", {})
    result = await catalog.execute("health_check", {})
    assert result.success
    assert "healthy" in result.message.lower()
    assert result.data["healthy_count"] >= 0


@pytest.mark.asyncio
async def test_sync_catalog(full_setup):
    catalog = full_setup["catalog"]
    result = await catalog.execute("sync_catalog", {})
    assert result.success
    assert "synced" in result.message.lower()


@pytest.mark.asyncio
async def test_catalog_status(full_setup):
    catalog = full_setup["catalog"]
    await catalog.execute("auto_register", {})
    result = await catalog.execute("catalog_status", {})
    assert result.success
    assert result.data["curated_services"] == len(SERVICE_CATALOG)
    assert result.data["marketplace_services"] > 0


@pytest.mark.asyncio
async def test_unknown_action(catalog_skill):
    result = await catalog_skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message


@pytest.mark.asyncio
async def test_service_catalog_matches_revenue_actions():
    """Verify curated catalog entries correspond to actual RevenueServiceSkill actions."""
    revenue = RevenueServiceSkill()
    revenue_actions = {a.name for a in revenue.get_actions()}
    for (skill_id, action_name) in SERVICE_CATALOG:
        assert skill_id == "revenue_services"
        assert action_name in revenue_actions, f"{action_name} not in RevenueServiceSkill"
