"""Tests for SpawnStrategySkill."""
import pytest
import asyncio
from singularity.skills.spawn_strategy import SpawnStrategySkill


class MockAgent:
    def __init__(self, balance=50.0):
        self.balance = balance
        self.cycle_interval = 5.0
        self.instance_cost_per_hour = 0.0


@pytest.fixture
def skill():
    s = SpawnStrategySkill()
    s.set_parent_agent(MockAgent(balance=50.0))
    return s


@pytest.mark.asyncio
async def test_evaluate_healthy(skill):
    result = await skill.execute("evaluate", {})
    assert result.success
    assert result.data["should_spawn"] is True
    assert result.data["balance"] == 50.0


@pytest.mark.asyncio
async def test_evaluate_low_balance():
    s = SpawnStrategySkill()
    s.set_parent_agent(MockAgent(balance=1.0))
    result = await s.execute("evaluate", {})
    assert result.success
    assert result.data["should_spawn"] is False


@pytest.mark.asyncio
async def test_recommend(skill):
    result = await skill.execute("recommend", {"purpose_hint": "code review"})
    assert result.success
    assert result.data["suggested_budget"] > 0
    assert result.data["suggested_model"]


@pytest.mark.asyncio
async def test_recommend_low_budget():
    s = SpawnStrategySkill()
    s.set_parent_agent(MockAgent(balance=0.5))
    result = await s.execute("recommend", {})
    assert not result.success


@pytest.mark.asyncio
async def test_calculate_budget(skill):
    result = await skill.execute("calculate_budget", {"purpose": "testing", "priority": "high"})
    assert result.success
    assert result.data["can_afford"] is True
    assert result.data["recommended_budget"] > 0
    assert result.data["remaining_after_spawn"] > 0


@pytest.mark.asyncio
async def test_register_and_fleet(skill):
    await skill.execute("register_replica", {
        "replica_id": "r1", "name": "TestBot", "purpose": "test", "budget_given": 5.0
    })
    result = await skill.execute("fleet_status", {})
    assert result.success
    assert result.data["summary"]["total_replicas"] == 1
    assert result.data["summary"]["active"] == 1


@pytest.mark.asyncio
async def test_update_replica(skill):
    await skill.execute("register_replica", {
        "replica_id": "r2", "name": "Bot2", "purpose": "earn", "budget_given": 10.0
    })
    result = await skill.execute("update_replica", {
        "replica_id": "r2", "revenue": 15.0, "cost": 5.0
    })
    assert result.success
    assert result.data["roi"] == 1.0  # (15-5)/10


@pytest.mark.asyncio
async def test_retirement_check(skill):
    await skill.execute("register_replica", {
        "replica_id": "r3", "name": "BadBot", "purpose": "fail", "budget_given": 10.0
    })
    await skill.execute("update_replica", {
        "replica_id": "r3", "cost": 8.0, "revenue": 0.0
    })
    result = await skill.execute("retirement_check", {})
    assert result.success
    assert len(result.data["retire"]) == 1


@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {"max_active_replicas": 10})
    assert result.success
    assert result.data["current_config"]["max_active_replicas"] == 10


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
