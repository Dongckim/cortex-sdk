"""Tests for RequestClassifier."""

import pytest

from cortex.optimizer.classifier import RequestClassifier
from cortex.optimizer.hybrid_roi import RequestType


@pytest.fixture
def classifier() -> RequestClassifier:
    return RequestClassifier()


# Voice classification tests

def test_voice_text_command(classifier: RequestClassifier) -> None:
    result = classifier.classify_voice("read the sign")
    assert result == RequestType.TEXT_RECOGNITION


def test_voice_translate(classifier: RequestClassifier) -> None:
    result = classifier.classify_voice("translate this menu")
    assert result == RequestType.TEXT_RECOGNITION


def test_voice_object_command(classifier: RequestClassifier) -> None:
    result = classifier.classify_voice("what is this?")
    assert result == RequestType.OBJECT_SCENE


def test_voice_describe(classifier: RequestClassifier) -> None:
    result = classifier.classify_voice("describe what you see")
    assert result == RequestType.OBJECT_SCENE


def test_voice_navigation(classifier: RequestClassifier) -> None:
    result = classifier.classify_voice("where do I turn?")
    assert result == RequestType.NAVIGATION


def test_voice_unknown(classifier: RequestClassifier) -> None:
    result = classifier.classify_voice("hello world")
    assert result == RequestType.GENERAL


def test_voice_confidence(classifier: RequestClassifier) -> None:
    classifier.classify_voice("read the text on the sign")
    assert classifier.confidence > 0.5


def test_voice_unknown_low_confidence(classifier: RequestClassifier) -> None:
    classifier.classify_voice("banana smoothie")
    assert classifier.confidence <= 0.3


# Implicit classification tests

def test_implicit_text_stationary(classifier: RequestClassifier) -> None:
    result = classifier.classify_implicit(has_text=True, is_moving=False)
    assert result == RequestType.TEXT_RECOGNITION


def test_implicit_no_text_moving(classifier: RequestClassifier) -> None:
    result = classifier.classify_implicit(has_text=False, is_moving=True)
    assert result == RequestType.NAVIGATION


def test_implicit_no_text_stationary(classifier: RequestClassifier) -> None:
    result = classifier.classify_implicit(has_text=False, is_moving=False)
    assert result == RequestType.OBJECT_SCENE


def test_implicit_text_moving(classifier: RequestClassifier) -> None:
    result = classifier.classify_implicit(has_text=True, is_moving=True)
    assert result == RequestType.GENERAL
