"""Tests for Summarizer."""

import time

import pytest

from cortex.memory.context_store import ContextEvent
from cortex.memory.summarizer import Summarizer


@pytest.fixture
def summarizer() -> Summarizer:
    return Summarizer()


def _event(desc: str, age: float = 0, etype: str = "object") -> ContextEvent:
    return ContextEvent(
        timestamp=time.time() - age,
        event_type=etype,
        description=desc,
    )


def test_summarize_short(summarizer: Summarizer) -> None:
    event = _event("a red cup on the desk")
    assert summarizer.summarize_event(event) == "a red cup on the desk"


def test_summarize_first_sentence(summarizer: Summarizer) -> None:
    event = _event("A notebook is open. There is writing on it. Blue pen.")
    assert summarizer.summarize_event(event) == "A notebook is open."


def test_summarize_long_truncates(summarizer: Summarizer) -> None:
    event = _event("x" * 100)
    result = summarizer.summarize_event(event)
    assert len(result) <= 60
    assert result.endswith("...")


def test_summarize_empty(summarizer: Summarizer) -> None:
    event = _event("", etype="scene")
    assert summarizer.summarize_event(event) == "scene"


def test_summarize_group(summarizer: Summarizer) -> None:
    events = [
        _event("cup", etype="object"),
        _event("text", etype="text"),
        _event("book", etype="object"),
    ]
    result = summarizer.summarize_group(events)
    assert "3 events" in result
    assert "object" in result
    assert "text" in result


def test_summarize_group_empty(summarizer: Summarizer) -> None:
    assert summarizer.summarize_group([]) == ""


def test_hierarchical_recent(summarizer: Summarizer) -> None:
    events = [_event("just happened", age=5)]
    result = summarizer.hierarchical_summary(events)
    assert len(result["recent"]) == 1
    assert len(result["mid"]) == 0
    assert len(result["old"]) == 0


def test_hierarchical_mid(summarizer: Summarizer) -> None:
    events = [_event("a minute ago", age=60)]
    result = summarizer.hierarchical_summary(events)
    assert len(result["recent"]) == 0
    assert len(result["mid"]) == 1


def test_hierarchical_old(summarizer: Summarizer) -> None:
    events = [_event("long ago", age=180)]
    result = summarizer.hierarchical_summary(events)
    assert len(result["old"]) == 1
    assert "1 events" in result["old"][0]


def test_hierarchical_mixed(summarizer: Summarizer) -> None:
    events = [
        _event("ancient", age=300),
        _event("mid range", age=60),
        _event("fresh", age=5),
    ]
    result = summarizer.hierarchical_summary(events)
    assert len(result["recent"]) == 1
    assert len(result["mid"]) == 1
    assert len(result["old"]) == 1


def test_estimate_tokens(summarizer: Summarizer) -> None:
    summary = {"recent": ["hello world"], "mid": ["test"], "old": []}
    tokens = summarizer.estimate_tokens(summary)
    assert tokens > 0
