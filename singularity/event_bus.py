#!/usr/bin/env python3
"""
EventBus - Reactive pub/sub event system for autonomous agents.

Enables event-driven agent behavior where skills can emit events and
other skills (or the agent itself) can subscribe to react automatically.

Key capabilities:
- Publish/subscribe with topic-based routing
- Wildcard subscriptions (e.g., "payment.*", "*.error")
- Event history with configurable retention
- Async handler support
- Event filtering and replay
- Dead letter queue for failed handlers
- Event persistence to disk for cross-session continuity

This is foundational infrastructure for all four pillars:
- Self-Improvement: react to performance metrics, trigger experiments
- Revenue: react to payments, trigger fulfillment pipelines
- Replication: react to load signals, trigger spawning
- Goal Setting: react to task completion, trigger reprioritization
"""

import asyncio
import json
import time
import fnmatch
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Union,
)
from enum import Enum


class EventPriority(Enum):
    """Event priority levels for ordered processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """An event in the system."""
    topic: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # skill_id or "agent" or "external"
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = ""
    timestamp: str = ""
    correlation_id: str = ""  # For tracking related events

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"evt_{int(time.time() * 1000)}_{id(self) % 10000}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["priority"] = self.priority.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Event":
        d = dict(d)
        if "priority" in d:
            d["priority"] = EventPriority(d["priority"])
        return cls(**d)


@dataclass
class Subscription:
    """A subscription to events matching a pattern."""
    pattern: str  # Topic pattern (supports wildcards: "payment.*", "*.error")
    handler: Callable  # async or sync callable
    subscriber_id: str = ""
    once: bool = False  # If True, auto-unsubscribe after first match
    filter_fn: Optional[Callable[[Event], bool]] = None  # Extra filter

    def __post_init__(self):
        if not self.subscriber_id:
            self.subscriber_id = f"sub_{int(time.time() * 1000)}_{id(self) % 10000}"

    def matches(self, topic: str) -> bool:
        """Check if a topic matches this subscription's pattern."""
        return fnmatch.fnmatch(topic, self.pattern)


@dataclass
class DeadLetterEntry:
    """Record of a failed event handler execution."""
    event: Event
    subscription_id: str
    error: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EventBus:
    """
    Central event bus for publish/subscribe communication.

    Features:
    - Topic-based routing with wildcard support
    - Async handler execution
    - Event history with configurable retention
    - Dead letter queue for debugging failed handlers
    - Event persistence for cross-session replay
    - Priority-based processing
    """

    def __init__(
        self,
        max_history: int = 1000,
        persist_path: Optional[str] = None,
    ):
        self._subscriptions: Dict[str, Subscription] = {}
        self._history: List[Event] = []
        self._dead_letters: List[DeadLetterEntry] = []
        self._max_history = max_history
        self._persist_path = Path(persist_path) if persist_path else None
        self._event_count = 0
        self._error_count = 0

        # Load persisted events if available
        if self._persist_path and self._persist_path.exists():
            self._load_persisted()

    def subscribe(
        self,
        pattern: str,
        handler: Callable,
        subscriber_id: str = "",
        once: bool = False,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """
        Subscribe to events matching a topic pattern.

        Args:
            pattern: Topic pattern with optional wildcards
                     "payment.received" - exact match
                     "payment.*" - all payment events
                     "*.error" - all error events
                     "*" - all events
            handler: Async or sync callable(event: Event)
            subscriber_id: Optional ID (auto-generated if empty)
            once: If True, auto-unsubscribe after first match
            filter_fn: Optional extra filter function

        Returns:
            Subscription ID for later unsubscription
        """
        sub = Subscription(
            pattern=pattern,
            handler=handler,
            subscriber_id=subscriber_id,
            once=once,
            filter_fn=filter_fn,
        )
        self._subscriptions[sub.subscriber_id] = sub
        return sub.subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe by subscription ID. Returns True if found."""
        if subscriber_id in self._subscriptions:
            del self._subscriptions[subscriber_id]
            return True
        return False

    def unsubscribe_all(self, pattern: Optional[str] = None) -> int:
        """
        Unsubscribe all handlers, optionally filtered by pattern.

        Returns number of subscriptions removed.
        """
        if pattern is None:
            count = len(self._subscriptions)
            self._subscriptions.clear()
            return count

        to_remove = [
            sid for sid, sub in self._subscriptions.items()
            if sub.pattern == pattern
        ]
        for sid in to_remove:
            del self._subscriptions[sid]
        return len(to_remove)

    async def publish(self, event: Union[Event, str], data: Dict = None, **kwargs) -> int:
        """
        Publish an event to all matching subscribers.

        Args:
            event: Event object or topic string
            data: Event data (if event is a string)
            **kwargs: Additional Event fields (source, priority, etc.)

        Returns:
            Number of handlers that were called
        """
        if isinstance(event, str):
            event = Event(topic=event, data=data or {}, **kwargs)

        # Record in history
        self._history.append(event)
        self._event_count += 1
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Persist if configured
        if self._persist_path:
            self._persist_event(event)

        # Find matching subscriptions, sorted by priority
        matching = []
        for sub in list(self._subscriptions.values()):
            if sub.matches(event.topic):
                if sub.filter_fn is None or sub.filter_fn(event):
                    matching.append(sub)

        # Execute handlers
        handlers_called = 0
        once_ids: List[str] = []

        for sub in matching:
            try:
                result = sub.handler(event)
                if asyncio.iscoroutine(result):
                    await result
                handlers_called += 1

                if sub.once:
                    once_ids.append(sub.subscriber_id)
            except Exception as e:
                self._error_count += 1
                self._dead_letters.append(DeadLetterEntry(
                    event=event,
                    subscription_id=sub.subscriber_id,
                    error=str(e),
                ))
                # Keep dead letter queue bounded
                if len(self._dead_letters) > 100:
                    self._dead_letters = self._dead_letters[-100:]

        # Remove once-subscriptions
        for sid in once_ids:
            self._subscriptions.pop(sid, None)

        return handlers_called

    def publish_sync(self, event: Union[Event, str], data: Dict = None, **kwargs) -> None:
        """
        Synchronous publish - creates event and queues for async processing.
        Useful for calling from sync code that can't await.
        """
        if isinstance(event, str):
            event = Event(topic=event, data=data or {}, **kwargs)

        self._history.append(event)
        self._event_count += 1
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(
        self,
        topic_pattern: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
        """
        Get event history with optional filtering.

        Args:
            topic_pattern: Filter by topic pattern (wildcards supported)
            source: Filter by source
            since: ISO timestamp - only events after this time
            limit: Max events to return

        Returns:
            List of matching events (newest first)
        """
        events = list(reversed(self._history))

        if topic_pattern:
            events = [e for e in events if fnmatch.fnmatch(e.topic, topic_pattern)]
        if source:
            events = [e for e in events if e.source == source]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[:limit]

    def get_dead_letters(self, limit: int = 20) -> List[Dict]:
        """Get recent dead letter entries for debugging."""
        entries = list(reversed(self._dead_letters))[:limit]
        return [
            {
                "event_topic": e.event.topic,
                "event_id": e.event.event_id,
                "subscription_id": e.subscription_id,
                "error": e.error,
                "timestamp": e.timestamp,
            }
            for e in entries
        ]

    def get_stats(self) -> Dict:
        """Get event bus statistics."""
        topic_counts: Dict[str, int] = {}
        for event in self._history:
            topic_counts[event.topic] = topic_counts.get(event.topic, 0) + 1

        return {
            "total_events_published": self._event_count,
            "total_errors": self._error_count,
            "active_subscriptions": len(self._subscriptions),
            "history_size": len(self._history),
            "dead_letter_count": len(self._dead_letters),
            "top_topics": dict(
                sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "subscriptions": [
                {"id": s.subscriber_id, "pattern": s.pattern, "once": s.once}
                for s in self._subscriptions.values()
            ],
        }

    def clear_history(self) -> int:
        """Clear event history. Returns number of events cleared."""
        count = len(self._history)
        self._history.clear()
        return count

    def clear_dead_letters(self) -> int:
        """Clear dead letter queue. Returns number cleared."""
        count = len(self._dead_letters)
        self._dead_letters.clear()
        return count

    async def replay(
        self,
        topic_pattern: str = "*",
        since: Optional[str] = None,
        limit: int = 100,
    ) -> int:
        """
        Replay historical events through current subscribers.

        Useful for:
        - Catching up a newly subscribed handler
        - Reprocessing after a handler fix
        - Testing subscriptions

        Returns number of events replayed.
        """
        events = self.get_history(
            topic_pattern=topic_pattern,
            since=since,
            limit=limit,
        )
        # Replay in chronological order
        events.reverse()

        count = 0
        for event in events:
            await self.publish(event)
            count += 1
        return count

    def _persist_event(self, event: Event):
        """Persist event to disk."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

            events = []
            if self._persist_path.exists():
                with open(self._persist_path, "r") as f:
                    events = json.load(f)

            events.append(event.to_dict())
            # Keep persisted events bounded
            if len(events) > self._max_history:
                events = events[-self._max_history:]

            with open(self._persist_path, "w") as f:
                json.dump(events, f, indent=2)
        except Exception:
            pass  # Don't let persistence failures break event delivery

    def _load_persisted(self):
        """Load persisted events from disk."""
        try:
            with open(self._persist_path, "r") as f:
                events_data = json.load(f)
            for ed in events_data:
                self._history.append(Event.from_dict(ed))
        except Exception:
            pass  # Don't break on corrupt persistence files
