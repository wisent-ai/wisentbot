"""Tests for EventBus and EventSkill."""

import asyncio
import json
import tempfile
import pytest
from pathlib import Path

from singularity.event_bus import EventBus, Event, EventPriority, Subscription


@pytest.fixture
def bus():
    return EventBus(max_history=100)


@pytest.fixture
def persisted_bus(tmp_path):
    return EventBus(max_history=100, persist_path=str(tmp_path / "events.json"))


class TestEvent:
    def test_event_creation(self):
        e = Event(topic="test.created", data={"key": "value"}, source="test")
        assert e.topic == "test.created"
        assert e.data == {"key": "value"}
        assert e.source == "test"
        assert e.event_id.startswith("evt_")
        assert e.timestamp != ""
        assert e.priority == EventPriority.NORMAL

    def test_event_to_dict_and_back(self):
        e = Event(topic="test.round_trip", data={"x": 1}, priority=EventPriority.HIGH)
        d = e.to_dict()
        assert d["topic"] == "test.round_trip"
        assert d["priority"] == 2  # HIGH = 2
        e2 = Event.from_dict(d)
        assert e2.topic == e.topic
        assert e2.priority == EventPriority.HIGH


class TestSubscription:
    def test_exact_match(self):
        s = Subscription(pattern="payment.received", handler=lambda e: None)
        assert s.matches("payment.received")
        assert not s.matches("payment.sent")

    def test_wildcard_suffix(self):
        s = Subscription(pattern="payment.*", handler=lambda e: None)
        assert s.matches("payment.received")
        assert s.matches("payment.sent")
        assert not s.matches("task.completed")

    def test_wildcard_prefix(self):
        s = Subscription(pattern="*.error", handler=lambda e: None)
        assert s.matches("payment.error")
        assert s.matches("task.error")
        assert not s.matches("task.success")

    def test_global_wildcard(self):
        s = Subscription(pattern="*", handler=lambda e: None)
        assert s.matches("anything")
        assert s.matches("payment.received")


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self, bus):
        count = await bus.publish("test.event", {"key": "value"})
        assert count == 0

    @pytest.mark.asyncio
    async def test_publish_with_subscriber(self, bus):
        received = []
        bus.subscribe("test.*", lambda e: received.append(e))
        count = await bus.publish("test.event", {"key": "value"})
        assert count == 1
        assert len(received) == 1
        assert received[0].topic == "test.event"

    @pytest.mark.asyncio
    async def test_async_handler(self, bus):
        received = []
        async def handler(e):
            received.append(e)
        bus.subscribe("test.*", handler)
        await bus.publish("test.event")
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus):
        r1, r2 = [], []
        bus.subscribe("test.*", lambda e: r1.append(e))
        bus.subscribe("test.*", lambda e: r2.append(e))
        count = await bus.publish("test.event")
        assert count == 2
        assert len(r1) == 1
        assert len(r2) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        received = []
        sub_id = bus.subscribe("test.*", lambda e: received.append(e))
        await bus.publish("test.event")
        assert len(received) == 1
        bus.unsubscribe(sub_id)
        await bus.publish("test.event")
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_once_subscription(self, bus):
        received = []
        bus.subscribe("test.*", lambda e: received.append(e), once=True)
        await bus.publish("test.first")
        await bus.publish("test.second")
        assert len(received) == 1
        assert received[0].topic == "test.first"

    @pytest.mark.asyncio
    async def test_filter_fn(self, bus):
        received = []
        bus.subscribe(
            "test.*",
            lambda e: received.append(e),
            filter_fn=lambda e: e.data.get("important", False),
        )
        await bus.publish("test.event", {"important": False})
        await bus.publish("test.event", {"important": True})
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_dead_letter_on_handler_error(self, bus):
        def bad_handler(e):
            raise ValueError("Handler failed!")
        bus.subscribe("test.*", bad_handler)
        count = await bus.publish("test.event")
        assert count == 0  # Handler failed
        dead = bus.get_dead_letters()
        assert len(dead) == 1
        assert "Handler failed!" in dead[0]["error"]

    @pytest.mark.asyncio
    async def test_history(self, bus):
        await bus.publish("a.one", source="src1")
        await bus.publish("b.two", source="src2")
        await bus.publish("a.three", source="src1")

        all_events = bus.get_history()
        assert len(all_events) == 3

        a_events = bus.get_history(topic_pattern="a.*")
        assert len(a_events) == 2

        src1_events = bus.get_history(source="src1")
        assert len(src1_events) == 2

    @pytest.mark.asyncio
    async def test_history_limit(self, bus):
        for i in range(10):
            await bus.publish(f"test.{i}")
        events = bus.get_history(limit=3)
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_stats(self, bus):
        bus.subscribe("test.*", lambda e: None)
        await bus.publish("test.one")
        await bus.publish("test.two")
        stats = bus.get_stats()
        assert stats["total_events_published"] == 2
        assert stats["active_subscriptions"] == 1
        assert "test.one" in stats["top_topics"]

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, bus):
        bus.subscribe("a.*", lambda e: None)
        bus.subscribe("b.*", lambda e: None)
        bus.subscribe("a.*", lambda e: None)
        removed = bus.unsubscribe_all("a.*")
        assert removed == 2
        assert bus.get_stats()["active_subscriptions"] == 1

    @pytest.mark.asyncio
    async def test_clear_history(self, bus):
        await bus.publish("test.one")
        await bus.publish("test.two")
        count = bus.clear_history()
        assert count == 2
        assert len(bus.get_history()) == 0

    @pytest.mark.asyncio
    async def test_max_history(self):
        bus = EventBus(max_history=5)
        for i in range(10):
            await bus.publish(f"test.{i}")
        assert len(bus.get_history(limit=100)) == 5

    @pytest.mark.asyncio
    async def test_persistence(self, persisted_bus, tmp_path):
        await persisted_bus.publish("test.persist", {"val": 42})
        # Create new bus from same path
        bus2 = EventBus(persist_path=str(tmp_path / "events.json"))
        events = bus2.get_history()
        assert len(events) == 1
        assert events[0].topic == "test.persist"

    @pytest.mark.asyncio
    async def test_replay(self, bus):
        received = []
        await bus.publish("test.one")
        await bus.publish("test.two")
        await bus.publish("other.three")
        bus.subscribe("test.*", lambda e: received.append(e))
        count = await bus.replay("test.*")
        assert count == 2
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_event_priority(self, bus):
        e = Event(topic="urgent", priority=EventPriority.CRITICAL)
        received = []
        bus.subscribe("*", lambda ev: received.append(ev))
        await bus.publish(e)
        assert received[0].priority == EventPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_correlation_id(self, bus):
        await bus.publish("step.1", correlation_id="flow-123")
        await bus.publish("step.2", correlation_id="flow-123")
        events = bus.get_history()
        assert all(e.correlation_id == "flow-123" for e in events)


class TestEventSkill:
    @pytest.fixture
    def skill(self):
        from singularity.skills.event import EventSkill
        s = EventSkill()
        s.set_event_bus(EventBus())
        return s

    @pytest.mark.asyncio
    async def test_publish(self, skill):
        result = await skill.execute("publish", {"topic": "test.hello", "data": {"msg": "hi"}})
        assert result.success
        assert result.data["topic"] == "test.hello"

    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self, skill):
        sub_result = await skill.execute("subscribe", {"pattern": "test.*", "reaction": "log it"})
        assert sub_result.success
        sub_id = sub_result.data["subscription_id"]

        await skill.execute("publish", {"topic": "test.event", "data": {"x": 1}})

        pending = skill.get_pending_events()
        assert len(pending) == 1
        assert pending[0]["event"]["topic"] == "test.event"
        assert pending[0]["reaction"] == "log it"

        # Pending events are cleared after retrieval
        assert len(skill.get_pending_events()) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe(self, skill):
        sub_result = await skill.execute("subscribe", {"pattern": "test.*"})
        sub_id = sub_result.data["subscription_id"]
        unsub_result = await skill.execute("unsubscribe", {"subscription_id": sub_id})
        assert unsub_result.success

    @pytest.mark.asyncio
    async def test_history(self, skill):
        await skill.execute("publish", {"topic": "a.one"})
        await skill.execute("publish", {"topic": "b.two"})
        result = await skill.execute("history", {"topic_pattern": "a.*"})
        assert result.success
        assert result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_stats(self, skill):
        await skill.execute("publish", {"topic": "test.event"})
        result = await skill.execute("stats", {})
        assert result.success
        assert result.data["total_events_published"] >= 1

    @pytest.mark.asyncio
    async def test_clear(self, skill):
        await skill.execute("publish", {"topic": "test.event"})
        result = await skill.execute("clear", {"target": "all"})
        assert result.success

    @pytest.mark.asyncio
    async def test_publish_requires_topic(self, skill):
        result = await skill.execute("publish", {})
        assert not result.success

    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        result = await skill.execute("nonexistent", {})
        assert not result.success
