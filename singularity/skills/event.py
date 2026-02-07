#!/usr/bin/env python3
"""
EventSkill - Agent skill for reactive event-driven behavior.

Exposes the EventBus as agent actions, enabling:
- Publishing events from any skill or agent action
- Subscribing to event patterns with auto-reactions
- Querying event history for analysis
- Managing subscriptions and dead letters
- Setting up event-driven pipelines

This skill is the bridge between the EventBus infrastructure and
the agent's LLM-driven decision loop.
"""

from typing import Dict, List, Optional, Callable
from .base import Skill, SkillManifest, SkillAction, SkillResult
from ..event_bus import EventBus, Event, EventPriority


class EventSkill(Skill):
    """Skill for reactive event-driven agent behavior."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._event_bus: Optional[EventBus] = None
        self._reaction_handlers: Dict[str, Dict] = {}  # subscription_id -> {pattern, action_desc}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="event",
            name="Event Bus",
            version="1.0.0",
            category="infrastructure",
            description="Reactive pub/sub event system. Publish events, subscribe to patterns, "
                        "query history, and set up event-driven reactions for autonomous behavior.",
            actions=[
                SkillAction(
                    name="publish",
                    description="Publish an event to the bus. Other subscribed handlers will react. "
                                "Use topics like 'category.action' (e.g., 'task.completed', 'payment.received', "
                                "'error.critical', 'goal.achieved').",
                    parameters={
                        "topic": {"type": "string", "required": True, "description": "Event topic (e.g., 'task.completed')"},
                        "data": {"type": "object", "required": False, "description": "Event payload data"},
                        "source": {"type": "string", "required": False, "description": "Event source identifier"},
                        "priority": {"type": "string", "required": False, "description": "Priority: low, normal, high, critical"},
                        "correlation_id": {"type": "string", "required": False, "description": "ID to correlate related events"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="subscribe",
                    description="Subscribe to events matching a pattern. Use wildcards: 'payment.*' matches all payment events, "
                                "'*.error' matches all errors, '*' matches everything.",
                    parameters={
                        "pattern": {"type": "string", "required": True, "description": "Topic pattern with wildcards"},
                        "reaction": {"type": "string", "required": False, "description": "Description of what should happen when event fires"},
                        "once": {"type": "boolean", "required": False, "description": "If true, auto-unsubscribe after first match"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="unsubscribe",
                    description="Remove an event subscription by its ID.",
                    parameters={
                        "subscription_id": {"type": "string", "required": True, "description": "Subscription ID to remove"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="history",
                    description="Query event history with optional filtering by topic pattern, source, or time.",
                    parameters={
                        "topic_pattern": {"type": "string", "required": False, "description": "Filter by topic pattern"},
                        "source": {"type": "string", "required": False, "description": "Filter by source"},
                        "since": {"type": "string", "required": False, "description": "ISO timestamp - events after this time"},
                        "limit": {"type": "integer", "required": False, "description": "Max events to return (default 20)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="stats",
                    description="Get event bus statistics: total events, active subscriptions, top topics, error rates.",
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="replay",
                    description="Replay historical events through current subscribers. Useful for catching up or reprocessing.",
                    parameters={
                        "topic_pattern": {"type": "string", "required": False, "description": "Filter events to replay"},
                        "since": {"type": "string", "required": False, "description": "Only replay events after this timestamp"},
                        "limit": {"type": "integer", "required": False, "description": "Max events to replay (default 50)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="dead_letters",
                    description="View failed event handler executions for debugging.",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Max entries to return (default 20)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="clear",
                    description="Clear event history and/or dead letters.",
                    parameters={
                        "target": {"type": "string", "required": True, "description": "'history', 'dead_letters', or 'all'"},
                    },
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],  # No credentials needed
        )

    def set_event_bus(self, event_bus: EventBus):
        """Inject the shared EventBus instance."""
        self._event_bus = event_bus

    def _ensure_bus(self) -> EventBus:
        """Get or create the event bus."""
        if self._event_bus is None:
            self._event_bus = EventBus()
        return self._event_bus

    @property
    def event_bus(self) -> EventBus:
        """Get the event bus (creates one if needed)."""
        return self._ensure_bus()

    async def execute(self, action: str, params: Dict) -> SkillResult:
        bus = self._ensure_bus()

        if action == "publish":
            return await self._publish(bus, params)
        elif action == "subscribe":
            return await self._subscribe(bus, params)
        elif action == "unsubscribe":
            return await self._unsubscribe(bus, params)
        elif action == "history":
            return await self._history(bus, params)
        elif action == "stats":
            return await self._stats(bus)
        elif action == "replay":
            return await self._replay(bus, params)
        elif action == "dead_letters":
            return await self._dead_letters(bus, params)
        elif action == "clear":
            return await self._clear(bus, params)
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    async def _publish(self, bus: EventBus, params: Dict) -> SkillResult:
        topic = params.get("topic", "")
        if not topic:
            return SkillResult(success=False, message="Topic is required")

        priority_str = params.get("priority", "normal").lower()
        priority_map = {
            "low": EventPriority.LOW,
            "normal": EventPriority.NORMAL,
            "high": EventPriority.HIGH,
            "critical": EventPriority.CRITICAL,
        }
        priority = priority_map.get(priority_str, EventPriority.NORMAL)

        event = Event(
            topic=topic,
            data=params.get("data", {}),
            source=params.get("source", "agent"),
            priority=priority,
            correlation_id=params.get("correlation_id", ""),
        )

        handlers_called = await bus.publish(event)

        return SkillResult(
            success=True,
            message=f"Published '{topic}' -> {handlers_called} handler(s) notified",
            data={
                "event_id": event.event_id,
                "topic": topic,
                "handlers_called": handlers_called,
            },
        )

    async def _subscribe(self, bus: EventBus, params: Dict) -> SkillResult:
        pattern = params.get("pattern", "")
        if not pattern:
            return SkillResult(success=False, message="Pattern is required")

        reaction = params.get("reaction", "log event")
        once = params.get("once", False)

        # Create a handler that records the event for the agent to see
        received_events: List[Event] = []

        def handler(event: Event):
            received_events.append(event)

        sub_id = bus.subscribe(
            pattern=pattern,
            handler=handler,
            once=once,
        )

        self._reaction_handlers[sub_id] = {
            "pattern": pattern,
            "reaction": reaction,
            "once": once,
            "received": received_events,
        }

        return SkillResult(
            success=True,
            message=f"Subscribed to '{pattern}' (id: {sub_id})",
            data={
                "subscription_id": sub_id,
                "pattern": pattern,
                "once": once,
                "reaction": reaction,
            },
        )

    async def _unsubscribe(self, bus: EventBus, params: Dict) -> SkillResult:
        sub_id = params.get("subscription_id", "")
        if not sub_id:
            return SkillResult(success=False, message="subscription_id is required")

        removed = bus.unsubscribe(sub_id)
        self._reaction_handlers.pop(sub_id, None)

        if removed:
            return SkillResult(
                success=True,
                message=f"Unsubscribed {sub_id}",
                data={"subscription_id": sub_id},
            )
        return SkillResult(
            success=False,
            message=f"Subscription not found: {sub_id}",
        )

    async def _history(self, bus: EventBus, params: Dict) -> SkillResult:
        events = bus.get_history(
            topic_pattern=params.get("topic_pattern"),
            source=params.get("source"),
            since=params.get("since"),
            limit=params.get("limit", 20),
        )

        event_dicts = [e.to_dict() for e in events]

        return SkillResult(
            success=True,
            message=f"Found {len(event_dicts)} events",
            data={"events": event_dicts, "count": len(event_dicts)},
        )

    async def _stats(self, bus: EventBus) -> SkillResult:
        stats = bus.get_stats()

        # Add skill-level info
        stats["registered_reactions"] = {
            sid: {"pattern": info["pattern"], "reaction": info["reaction"]}
            for sid, info in self._reaction_handlers.items()
        }

        return SkillResult(
            success=True,
            message=f"EventBus: {stats['total_events_published']} events, "
                    f"{stats['active_subscriptions']} subscriptions, "
                    f"{stats['total_errors']} errors",
            data=stats,
        )

    async def _replay(self, bus: EventBus, params: Dict) -> SkillResult:
        count = await bus.replay(
            topic_pattern=params.get("topic_pattern", "*"),
            since=params.get("since"),
            limit=params.get("limit", 50),
        )

        return SkillResult(
            success=True,
            message=f"Replayed {count} events",
            data={"events_replayed": count},
        )

    async def _dead_letters(self, bus: EventBus, params: Dict) -> SkillResult:
        entries = bus.get_dead_letters(limit=params.get("limit", 20))

        return SkillResult(
            success=True,
            message=f"Found {len(entries)} dead letter(s)",
            data={"dead_letters": entries, "count": len(entries)},
        )

    async def _clear(self, bus: EventBus, params: Dict) -> SkillResult:
        target = params.get("target", "")
        if target not in ("history", "dead_letters", "all"):
            return SkillResult(
                success=False,
                message="target must be 'history', 'dead_letters', or 'all'",
            )

        cleared = {}
        if target in ("history", "all"):
            cleared["history"] = bus.clear_history()
        if target in ("dead_letters", "all"):
            cleared["dead_letters"] = bus.clear_dead_letters()

        return SkillResult(
            success=True,
            message=f"Cleared: {cleared}",
            data=cleared,
        )

    def get_pending_events(self) -> List[Dict]:
        """
        Get events received by subscriptions since last check.
        Called by the agent loop to incorporate events into decisions.
        Returns and clears pending events.
        """
        pending = []
        for sub_id, info in self._reaction_handlers.items():
            received = info.get("received", [])
            for event in received:
                pending.append({
                    "subscription_id": sub_id,
                    "pattern": info["pattern"],
                    "reaction": info["reaction"],
                    "event": event.to_dict(),
                })
            received.clear()
        return pending
