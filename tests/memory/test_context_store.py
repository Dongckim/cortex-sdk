"""Tests for ContextStore."""

import time

import pytest

from cortex.memory.context_store import ContextEvent, ContextStore


@pytest.fixture
def store() -> ContextStore:
    return ContextStore(max_events=5, ttl_seconds=2.0)


def _make_event(desc: str, event_type: str = "object") -> ContextEvent:
    return ContextEvent(
        timestamp=time.time(),
        event_type=event_type,
        description=desc,
    )


def test_add_event(store: ContextStore) -> None:
    store.add(_make_event("desk with notebook"))
    assert store.size == 1
    assert store.total_added == 1


def test_fifo_eviction(store: ContextStore) -> None:
    for i in range(7):
        store.add(_make_event(f"event {i}"))
    assert store.size == 5  # max_events=5
    assert store.events[0].description == "event 2"


def test_ttl_expiration(store: ContextStore) -> None:
    store.add(_make_event("old event"))
    time.sleep(2.5)
    assert store.size == 0


def test_events_ordered(store: ContextStore) -> None:
    store.add(_make_event("first"))
    store.add(_make_event("second"))
    assert store.events[0].description == "first"
    assert store.events[1].description == "second"


def test_get_by_type(store: ContextStore) -> None:
    store.add(_make_event("text here", event_type="text"))
    store.add(_make_event("a cup", event_type="object"))
    store.add(_make_event("more text", event_type="text"))
    assert len(store.get_by_type("text")) == 2
    assert len(store.get_by_type("object")) == 1


def test_clear(store: ContextStore) -> None:
    store.add(_make_event("something"))
    store.clear()
    assert store.size == 0


def test_age_label() -> None:
    event = ContextEvent(
        timestamp=time.time() - 90,
        event_type="object",
        description="test",
    )
    assert "1m ago" == event.age_label


def test_recent(store: ContextStore) -> None:
    store.add(_make_event("recent"))
    assert len(store.recent) == 1
