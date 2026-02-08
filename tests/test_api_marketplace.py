"""Tests for ExternalAPIMarketplaceSkill."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from singularity.skills.api_marketplace import (
    ExternalAPIMarketplaceSkill,
    BUILTIN_APIS,
    SUBSCRIPTION_TIERS,
)


@pytest.fixture
def skill():
    with patch("singularity.skills.api_marketplace._load_json", return_value={}):
        s = ExternalAPIMarketplaceSkill()
    s._usage = {"calls": [], "revenue": {"total": 0.0, "by_api": {}, "by_customer": {}}, "rate_limits": {}}
    s._subscriptions = {}
    s._custom_apis = {}
    return s


@pytest.mark.asyncio
async def test_browse_all(skill):
    result = await skill.execute("browse", {})
    assert result.success
    assert result.data["total"] == len(BUILTIN_APIS)
    assert len(result.data["apis"]) == len(BUILTIN_APIS)


@pytest.mark.asyncio
async def test_browse_by_category(skill):
    result = await skill.execute("browse", {"category": "weather"})
    assert result.success
    assert all(a["category"] == "weather" for a in result.data["apis"])


@pytest.mark.asyncio
async def test_browse_by_tag(skill):
    result = await skill.execute("browse", {"tag": "free-tier"})
    assert result.success
    assert result.data["total"] > 0


@pytest.mark.asyncio
async def test_browse_by_search(skill):
    result = await skill.execute("browse", {"search": "currency"})
    assert result.success
    assert result.data["total"] >= 1


@pytest.mark.asyncio
async def test_details(skill):
    result = await skill.execute("details", {"api_id": "weather_current"})
    assert result.success
    assert result.data["api"]["name"] == "Current Weather"
    assert result.data["is_builtin"] is True


@pytest.mark.asyncio
async def test_details_not_found(skill):
    result = await skill.execute("details", {"api_id": "nonexistent"})
    assert not result.success


@pytest.mark.asyncio
async def test_call_api_no_http(skill):
    result = await skill.execute("call", {
        "customer_id": "cust1", "api_id": "weather_current", "params": {"city": "London"}
    })
    assert result.success
    assert result.data["price_charged"] > 0
    assert result.revenue > 0
    assert "simulated" in result.data.get("note", "").lower() or "queued" in result.message.lower()


@pytest.mark.asyncio
async def test_call_api_missing_param(skill):
    result = await skill.execute("call", {
        "customer_id": "cust1", "api_id": "weather_current", "params": {}
    })
    assert not result.success
    assert "required" in result.message.lower()


@pytest.mark.asyncio
async def test_call_api_not_found(skill):
    result = await skill.execute("call", {
        "customer_id": "cust1", "api_id": "nonexistent", "params": {}
    })
    assert not result.success


@pytest.mark.asyncio
async def test_subscribe(skill):
    result = await skill.execute("subscribe", {"customer_id": "cust1", "tier": "pro"})
    assert result.success
    assert result.data["new_tier"] == "pro"
    assert result.data["discount_pct"] == 25
    assert result.revenue == 29.99


@pytest.mark.asyncio
async def test_subscribe_invalid_tier(skill):
    result = await skill.execute("subscribe", {"customer_id": "cust1", "tier": "mega"})
    assert not result.success


@pytest.mark.asyncio
async def test_tier_discount_on_call(skill):
    await skill.execute("subscribe", {"customer_id": "cust1", "tier": "pro"})
    result = await skill.execute("call", {
        "customer_id": "cust1", "api_id": "weather_current", "params": {"city": "London"}
    })
    assert result.success
    base = BUILTIN_APIS["weather_current"]["price_per_call"]
    assert result.data["price_charged"] == round(base * 0.75, 6)


@pytest.mark.asyncio
async def test_add_custom_api(skill):
    result = await skill.execute("add_api", {
        "api_id": "my_api", "name": "My API", "description": "Test",
        "base_url": "https://api.example.com/{query}",
        "price_per_call": 0.01,
    })
    assert result.success
    assert "my_api" in skill._catalog


@pytest.mark.asyncio
async def test_add_api_cannot_override_builtin(skill):
    result = await skill.execute("add_api", {
        "api_id": "weather_current", "name": "Override", "base_url": "https://evil.com",
    })
    assert not result.success


@pytest.mark.asyncio
async def test_remove_custom_api(skill):
    await skill.execute("add_api", {
        "api_id": "temp_api", "name": "Temp", "description": "x", "base_url": "https://a.com",
    })
    result = await skill.execute("remove_api", {"api_id": "temp_api"})
    assert result.success
    assert "temp_api" not in skill._catalog


@pytest.mark.asyncio
async def test_remove_builtin_blocked(skill):
    result = await skill.execute("remove_api", {"api_id": "weather_current"})
    assert not result.success


@pytest.mark.asyncio
async def test_usage_tracking(skill):
    await skill.execute("call", {"customer_id": "c1", "api_id": "dns_lookup", "params": {"domain": "x.com"}})
    await skill.execute("call", {"customer_id": "c1", "api_id": "dns_lookup", "params": {"domain": "y.com"}})
    result = await skill.execute("usage", {"customer_id": "c1"})
    assert result.success
    assert result.data["total_calls"] == 2


@pytest.mark.asyncio
async def test_revenue_report(skill):
    await skill.execute("call", {"customer_id": "c1", "api_id": "weather_current", "params": {"city": "NYC"}})
    result = await skill.execute("revenue", {})
    assert result.success
    assert result.data["api_call_revenue"] > 0


@pytest.mark.asyncio
async def test_tiers_listing(skill):
    result = await skill.execute("tiers", {})
    assert result.success
    assert len(result.data["tiers"]) == len(SUBSCRIPTION_TIERS)


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "api_marketplace"
    assert m.category == "revenue"
    assert len(m.actions) == 9


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
