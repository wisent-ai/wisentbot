"""Tests for action middleware system."""
import asyncio
import pytest
from singularity.middleware import (
    ActionMiddleware, MiddlewareChain, ActionContext,
    RetryMiddleware, CostGuardMiddleware, ActionLogMiddleware,
    RateLimitMiddleware,
)


def make_ctx(**kwargs):
    defaults = {"tool": "test:action", "params": {"key": "val"}, "cycle": 1, "balance": 10.0}
    defaults.update(kwargs)
    return ActionContext(**defaults)


# --- MiddlewareChain ---

class BlockingMiddleware(ActionMiddleware):
    async def before_action(self, ctx):
        return None

class PassthroughMiddleware(ActionMiddleware):
    async def before_action(self, ctx):
        ctx.metadata["passed"] = True
        return ctx


@pytest.mark.asyncio
async def test_chain_before_passthrough():
    chain = MiddlewareChain()
    chain.add(PassthroughMiddleware())
    ctx = make_ctx()
    result = await chain.run_before(ctx)
    assert result is not None
    assert result.metadata["passed"] is True


@pytest.mark.asyncio
async def test_chain_before_blocks():
    chain = MiddlewareChain()
    chain.add(BlockingMiddleware())
    result = await chain.run_before(make_ctx())
    assert result is None


@pytest.mark.asyncio
async def test_chain_order_matters():
    chain = MiddlewareChain()
    chain.add(BlockingMiddleware())
    chain.add(PassthroughMiddleware())
    # Blocker runs first, should block
    result = await chain.run_before(make_ctx())
    assert result is None


@pytest.mark.asyncio
async def test_chain_after_runs_reverse():
    order = []
    class MW1(ActionMiddleware):
        async def after_action(self, ctx, result):
            order.append("MW1")
            return result
    class MW2(ActionMiddleware):
        async def after_action(self, ctx, result):
            order.append("MW2")
            return result
    chain = MiddlewareChain()
    chain.add(MW1())
    chain.add(MW2())
    await chain.run_after(make_ctx(), {"status": "success"})
    assert order == ["MW2", "MW1"]


@pytest.mark.asyncio
async def test_chain_on_error_first_handler_wins():
    class Handler(ActionMiddleware):
        async def on_error(self, ctx, error):
            return {"status": "recovered"}
    class Handler2(ActionMiddleware):
        async def on_error(self, ctx, error):
            return {"status": "second"}
    chain = MiddlewareChain()
    chain.add(Handler())
    chain.add(Handler2())
    result = await chain.run_on_error(make_ctx(), Exception("fail"))
    assert result["status"] == "recovered"


def test_chain_list_and_remove():
    chain = MiddlewareChain()
    chain.add(PassthroughMiddleware())
    chain.add(BlockingMiddleware())
    assert chain.list() == ["PassthroughMiddleware", "BlockingMiddleware"]
    assert chain.remove("BlockingMiddleware")
    assert chain.list() == ["PassthroughMiddleware"]
    assert not chain.remove("NonExistent")


def test_chain_clear():
    chain = MiddlewareChain()
    chain.add(PassthroughMiddleware())
    chain.clear()
    assert chain.is_empty


# --- CostGuardMiddleware ---

@pytest.mark.asyncio
async def test_cost_guard_blocks_low_balance():
    mw = CostGuardMiddleware(min_balance=5.0)
    ctx = make_ctx(balance=2.0)
    result = await mw.before_action(ctx)
    assert result is None
    assert len(mw.block_history) == 1


@pytest.mark.asyncio
async def test_cost_guard_allows_sufficient_balance():
    mw = CostGuardMiddleware(min_balance=5.0)
    ctx = make_ctx(balance=10.0)
    result = await mw.before_action(ctx)
    assert result is not None


@pytest.mark.asyncio
async def test_cost_guard_always_allows_safe_tools():
    mw = CostGuardMiddleware(min_balance=5.0)
    ctx = make_ctx(tool="filesystem:ls", balance=0.01)
    result = await mw.before_action(ctx)
    assert result is not None


# --- RetryMiddleware ---

@pytest.mark.asyncio
async def test_retry_handles_retryable_error():
    mw = RetryMiddleware(max_retries=2)
    ctx = make_ctx()
    result = await mw.on_error(ctx, Exception("connection timeout"))
    assert result is not None
    assert result["status"] == "retry"


@pytest.mark.asyncio
async def test_retry_ignores_non_retryable():
    mw = RetryMiddleware(max_retries=2)
    ctx = make_ctx()
    result = await mw.on_error(ctx, Exception("invalid parameter"))
    assert result is None


@pytest.mark.asyncio
async def test_retry_respects_max():
    mw = RetryMiddleware(max_retries=1)
    ctx = make_ctx()
    r1 = await mw.on_error(ctx, Exception("timeout"))
    assert r1 is not None  # First retry allowed
    r2 = await mw.on_error(ctx, Exception("timeout"))
    assert r2 is None  # Max reached


@pytest.mark.asyncio
async def test_retry_clears_on_success():
    mw = RetryMiddleware(max_retries=1)
    ctx = make_ctx()
    await mw.on_error(ctx, Exception("timeout"))
    await mw.after_action(ctx, {"status": "success"})
    # After success, counter should be cleared, can retry again
    r = await mw.on_error(ctx, Exception("timeout"))
    assert r is not None


# --- ActionLogMiddleware ---

@pytest.mark.asyncio
async def test_action_log_records():
    mw = ActionLogMiddleware()
    ctx = make_ctx()
    await mw.before_action(ctx)
    await mw.after_action(ctx, {"status": "success"})
    assert len(mw.action_log) == 1
    assert mw.action_log[0]["status"] == "success"


@pytest.mark.asyncio
async def test_action_log_stats():
    mw = ActionLogMiddleware()
    for i in range(3):
        ctx = make_ctx(cycle=i)
        await mw.before_action(ctx)
        await mw.after_action(ctx, {"status": "success"})
    ctx = make_ctx(cycle=99)
    await mw.before_action(ctx)
    await mw.on_error(ctx, Exception("fail"))
    stats = mw.get_stats()
    assert stats["total"] == 4
    assert stats["successes"] == 3
    assert stats["errors"] == 1


# --- RateLimitMiddleware ---

@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    mw = RateLimitMiddleware(max_per_minute=5)
    ctx = make_ctx()
    result = await mw.before_action(ctx)
    assert result is not None


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    mw = RateLimitMiddleware(max_per_minute=2, per_tool_per_minute=10)
    for i in range(2):
        await mw.before_action(make_ctx(cycle=i))
    result = await mw.before_action(make_ctx(cycle=99))
    assert result is None
