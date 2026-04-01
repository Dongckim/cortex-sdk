"""Sliding window context event store."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ContextEvent:
    """A single context event from a VLM response.

    Attributes:
        timestamp: Unix timestamp of the event.
        event_type: Type of event (e.g., "text", "object", "scene").
        description: Human-readable description.
        summary: Compressed summary (generated later).
        embedding: Vector embedding for retrieval.
        metadata: Additional data (scores, frame stats, etc.).
    """

    timestamp: float
    event_type: str
    description: str
    summary: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Seconds since this event was created."""
        return time.time() - self.timestamp

    @property
    def age_label(self) -> str:
        """Human-readable age string."""
        age = self.age_seconds
        if age < 10:
            return "just now"
        elif age < 60:
            return f"{int(age)}s ago"
        elif age < 3600:
            return f"{int(age / 60)}m ago"
        else:
            return f"{int(age / 3600)}h ago"


class ContextStore:
    """FIFO sliding window store for context events.

    Maintains a fixed-size window of recent events with
    optional time-to-live (TTL) expiration.

    Args:
        max_events: Maximum events in the window. Default is 20.
        ttl_seconds: Time-to-live for events. Default is 300 (5 min).
    """

    def __init__(
        self,
        max_events: int = 20,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._events: deque[ContextEvent] = deque(maxlen=max_events)
        self._ttl = ttl_seconds
        self._total_added = 0

    def add(self, event: ContextEvent) -> None:
        """Add an event to the store.

        Args:
            event: Context event to add.
        """
        self._events.append(event)
        self._total_added += 1
        self._expire()
        logger.debug(
            "event added: type=%s desc='%s' (total=%d, active=%d)",
            event.event_type,
            event.description[:50],
            self._total_added,
            len(self._events),
        )

    def _expire(self) -> None:
        """Remove events older than TTL."""
        now = time.time()
        while self._events and (now - self._events[0].timestamp) > self._ttl:
            expired = self._events.popleft()
            logger.debug("event expired: '%s'", expired.description[:50])

    @property
    def events(self) -> list[ContextEvent]:
        """All active events, oldest first."""
        self._expire()
        return list(self._events)

    @property
    def recent(self) -> list[ContextEvent]:
        """Events from the last 30 seconds."""
        self._expire()
        now = time.time()
        return [e for e in self._events if (now - e.timestamp) < 30]

    @property
    def size(self) -> int:
        """Number of active events."""
        self._expire()
        return len(self._events)

    @property
    def total_added(self) -> int:
        """Total events ever added."""
        return self._total_added

    def clear(self) -> None:
        """Remove all events."""
        self._events.clear()

    def get_by_type(self, event_type: str) -> list[ContextEvent]:
        """Get events filtered by type.

        Args:
            event_type: Event type to filter by.

        Returns:
            Matching events.
        """
        self._expire()
        return [e for e in self._events if e.event_type == event_type]
