"""Tests for RevenueSkill - service catalog, invoicing, expenses, and financial reports."""

import pytest
from singularity.skills.revenue import RevenueSkill


@pytest.fixture
def skill():
    s = RevenueSkill()
    s.initialized = True
    return s


@pytest.mark.asyncio
async def test_create_service(skill):
    r = await skill.execute("create_service", {
        "name": "Code Review", "description": "Review PRs", "price": 25.0
    })
    assert r.success
    assert r.data["name"] == "Code Review"
    assert r.data["price"] == 25.0
    assert r.data["price_unit"] == "per_task"
    assert r.data["service_id"].startswith("svc_")


@pytest.mark.asyncio
async def test_create_service_validation(skill):
    r = await skill.execute("create_service", {"name": "", "description": "x", "price": 10})
    assert not r.success
    r = await skill.execute("create_service", {"name": "x", "description": "", "price": 10})
    assert not r.success
    r = await skill.execute("create_service", {"name": "x", "description": "x", "price": -5})
    assert not r.success
    r = await skill.execute("create_service", {
        "name": "x", "description": "x", "price": 10, "price_unit": "invalid"
    })
    assert not r.success


@pytest.mark.asyncio
async def test_list_services(skill):
    r = await skill.execute("list_services", {})
    assert r.success
    assert r.data["total"] == 0

    await skill.execute("create_service", {"name": "A", "description": "a", "price": 10})
    await skill.execute("create_service", {"name": "B", "description": "b", "price": 20})
    r = await skill.execute("list_services", {})
    assert r.data["total"] == 2
    assert r.data["active"] == 2


@pytest.mark.asyncio
async def test_update_service(skill):
    cr = await skill.execute("create_service", {"name": "Test", "description": "d", "price": 10})
    sid = cr.data["service_id"]

    r = await skill.execute("update_service", {"service_id": sid, "price": 50})
    assert r.success
    assert "price=$50.00" in r.message

    r = await skill.execute("update_service", {"service_id": sid, "active": False})
    assert r.success

    r = await skill.execute("update_service", {"service_id": "bad_id"})
    assert not r.success

    r = await skill.execute("update_service", {"service_id": sid})
    assert not r.success  # no updates provided


@pytest.mark.asyncio
async def test_create_invoice(skill):
    cr = await skill.execute("create_service", {"name": "Svc", "description": "d", "price": 100})
    sid = cr.data["service_id"]

    r = await skill.execute("create_invoice", {"service_id": sid, "client": "Alice"})
    assert r.success
    assert r.data["amount"] == 100
    assert r.data["client"] == "Alice"
    assert r.data["invoice_id"].startswith("inv_")


@pytest.mark.asyncio
async def test_create_invoice_inactive_service(skill):
    cr = await skill.execute("create_service", {"name": "S", "description": "d", "price": 10})
    sid = cr.data["service_id"]
    await skill.execute("update_service", {"service_id": sid, "active": False})

    r = await skill.execute("create_invoice", {"service_id": sid, "client": "Bob"})
    assert not r.success


@pytest.mark.asyncio
async def test_record_payment(skill):
    cr = await skill.execute("create_service", {"name": "S", "description": "d", "price": 50})
    sid = cr.data["service_id"]
    inv = await skill.execute("create_invoice", {"service_id": sid, "client": "C"})
    inv_id = inv.data["invoice_id"]

    r = await skill.execute("record_payment", {"invoice_id": inv_id})
    assert r.success
    assert r.revenue == 50

    # Double payment should fail
    r = await skill.execute("record_payment", {"invoice_id": inv_id})
    assert not r.success


@pytest.mark.asyncio
async def test_record_expense(skill):
    r = await skill.execute("record_expense", {
        "amount": 5.0, "category": "compute", "description": "LLM calls"
    })
    assert r.success
    assert r.cost == 5.0

    r = await skill.execute("record_expense", {"amount": -1, "description": "bad"})
    assert not r.success

    r = await skill.execute("record_expense", {"amount": 1, "description": ""})
    assert not r.success


@pytest.mark.asyncio
async def test_financial_report(skill):
    cr = await skill.execute("create_service", {"name": "S", "description": "d", "price": 100})
    sid = cr.data["service_id"]
    inv = await skill.execute("create_invoice", {"service_id": sid, "client": "C"})
    await skill.execute("record_payment", {"invoice_id": inv.data["invoice_id"]})
    await skill.execute("record_expense", {"amount": 30, "category": "compute", "description": "API"})

    r = await skill.execute("financial_report", {})
    assert r.success
    assert r.data["total_revenue"] == 100
    assert r.data["total_expenses"] == 30
    assert r.data["profit"] == 70
    assert r.data["status"] == "profitable"


@pytest.mark.asyncio
async def test_top_services(skill):
    r = await skill.execute("top_services", {})
    assert r.success
    assert r.data["total_services"] == 0

    cr1 = await skill.execute("create_service", {"name": "A", "description": "a", "price": 10})
    cr2 = await skill.execute("create_service", {"name": "B", "description": "b", "price": 50})

    inv = await skill.execute("create_invoice", {"service_id": cr2.data["service_id"], "client": "X"})
    await skill.execute("record_payment", {"invoice_id": inv.data["invoice_id"]})

    r = await skill.execute("top_services", {})
    assert r.success
    assert r.data["rankings"][0]["name"] == "B"


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "revenue"
    assert m.category == "economics"
    assert len(m.actions) == 9
    assert skill.check_credentials()
