"""Tests for ContextInjector."""

import time

import pytest

from cortex.memory.context_store import ContextEvent, ContextStore
from cortex.memory.injector import ContextInjector


@pytest.fixture
def store() -> ContextStore:
    return ContextStore(max_events=20, ttl_seconds=300)


@pytest.fixture
def injector(store: ContextStore) -> ContextInjector:
    return ContextInjector(store)


def _event(desc: str, age: float = 0, etype: str = "object") -> ContextEvent:
    return ContextEvent(
        timestamp=time.time() - age,
        event_type=etype,
        description=desc,
    )


def test_empty_context(injector: ContextInjector) -> None:
    assert injector.build_context() == ""


def test_recent_context(
    injector: ContextInjector, store: ContextStore
) -> None:
    store.add(_event("a notebook on desk", age=5))
    context = injector.build_context()
    assert "notebook" in context
    assert "Recent" in context


def test_mid_context(
    injector: ContextInjector, store: ContextStore
) -> None:
    store.add(_event("coffee cup visible", age=60))
    context = injector.build_context()
    assert "Earlier" in context


def test_old_context(
    injector: ContextInjector, store: ContextStore
) -> None:
    store.add(_event("whiteboard notes", age=180))
    context = injector.build_context()
    assert "Background" in context


def test_query_retrieval(
    injector: ContextInjector, store: ContextStore
) -> None:
    store.add(_event("red notebook on desk", age=5))
    store.add(_event("person walking by", age=10))
    context = injector.build_context(query="notebook")
    assert "notebook" in context


def test_inject_prompt(
    injector: ContextInjector, store: ContextStore
) -> None:
    store.add(_event("laptop screen showing code", age=5))
    prompt = injector.inject_prompt("What do you see?")
    assert "[Context]" in prompt
    assert "What do you see?" in prompt
    assert "laptop" in prompt


def test_inject_prompt_with_system(
    injector: ContextInjector, store: ContextStore
) -> None:
    store.add(_event("desk scene", age=5))
    prompt = injector.inject_prompt(
        "Describe this",
        system_context="You are a visual assistant.",
    )
    assert "visual assistant" in prompt
    assert "Describe this" in prompt


def test_inject_prompt_empty_store(injector: ContextInjector) -> None:
    prompt = injector.inject_prompt("Hello")
    assert prompt == "Hello"


def test_context_stats(
    injector: ContextInjector, store: ContextStore
) -> None:
    store.add(_event("event1", age=5))
    store.add(_event("event2", age=60))
    stats = injector.context_stats
    assert stats["total_events"] == 2
    assert stats["recent_count"] == 1
    assert stats["mid_count"] == 1
    assert stats["estimated_tokens"] > 0
