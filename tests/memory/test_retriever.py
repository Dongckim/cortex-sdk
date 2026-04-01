"""Tests for Retriever."""

import time

import pytest

from cortex.memory.context_store import ContextEvent
from cortex.memory.retriever import Retriever


@pytest.fixture
def retriever() -> Retriever:
    return Retriever()


def _event(desc: str, etype: str = "object") -> ContextEvent:
    return ContextEvent(
        timestamp=time.time(),
        event_type=etype,
        description=desc,
    )


@pytest.fixture
def events() -> list[ContextEvent]:
    return [
        _event("a red notebook on the desk"),
        _event("person walking in background"),
        _event("coffee cup next to laptop"),
        _event("text on whiteboard: meeting at 3pm"),
        _event("dark room with monitor glowing"),
    ]


def test_search_returns_relevant(
    retriever: Retriever, events: list[ContextEvent]
) -> None:
    results = retriever.search("notebook", events)
    assert len(results) > 0
    assert "notebook" in results[0][0].description


def test_search_text_query(
    retriever: Retriever, events: list[ContextEvent]
) -> None:
    results = retriever.search("whiteboard text", events)
    assert "whiteboard" in results[0][0].description


def test_search_top_k(
    retriever: Retriever, events: list[ContextEvent]
) -> None:
    results = retriever.search("desk", events, top_k=2)
    assert len(results) == 2


def test_search_scores_descending(
    retriever: Retriever, events: list[ContextEvent]
) -> None:
    results = retriever.search("cup coffee", events)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_query(
    retriever: Retriever, events: list[ContextEvent]
) -> None:
    results = retriever.search("", events)
    assert len(results) == 0


def test_search_empty_events(retriever: Retriever) -> None:
    results = retriever.search("hello", [])
    assert len(results) == 0


def test_search_no_match(retriever: Retriever) -> None:
    events = [_event("apple banana cherry")]
    results = retriever.search("xyz123", events)
    assert len(results) == 1
    assert results[0][1] == 0.0  # no similarity
