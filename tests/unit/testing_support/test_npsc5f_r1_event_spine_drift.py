# © Artur Czarnecki. All rights reserved.

"""Unit tests for NPSC-5F/R1 event-spine drift classification."""

from __future__ import annotations

import pytest

from testing_support.npsc5f_r1_event_spine_drift import classify_event_spine_r1_protected_change

pytestmark = pytest.mark.unit


def test_event_spine_classifier_runtime_event_is_a() -> None:
    assert (
        classify_event_spine_r1_protected_change("intergrax/runtime/events/runtime_event.py")
        == "A"
    )


def test_event_spine_classifier_event_bus_is_b() -> None:
    assert classify_event_spine_r1_protected_change("intergrax/runtime/events/event_bus.py") == "B"


def test_event_spine_classifier_rejects_non_spine_path() -> None:
    with pytest.raises(ValueError, match="not an event-spine"):
        classify_event_spine_r1_protected_change("intergrax/runtime/events/event_catalog.py")
