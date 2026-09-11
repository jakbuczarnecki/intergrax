# © Artur Czarnecki. All rights reserved.

"""Unit tests for NPSC-5F/R1 protected production drift classification."""

from __future__ import annotations

import pytest

from testing_support.npsc5f_r1_protected_drift import (
    classify_r1_protected_drift,
    is_r1_protected_production_path,
)

pytestmark = pytest.mark.unit


def test_r1_drift_classifier_allows_docs_change() -> None:
    assert classify_r1_protected_drift(["docs/project/maintainers/qualification/x.md"]) == []


def test_r1_drift_classifier_allows_session_d_test_runner_change() -> None:
    assert (
        classify_r1_protected_drift(
            ["testing_support/execution_qualification/coordinator.py"],
        )
        == []
    )


def test_r1_drift_classifier_ignores_unrelated_execution_surface() -> None:
    path = "intergrax/runtime/execution/runtime.py"
    assert not is_r1_protected_production_path(path)
    assert classify_r1_protected_drift([path]) == []


def test_r1_drift_classifier_flags_event_bus_change() -> None:
    path = "intergrax/runtime/events/event_bus.py"
    assert is_r1_protected_production_path(path)
    assert classify_r1_protected_drift([path]) == [path]


def test_r1_drift_classifier_flags_persistence_contract_change() -> None:
    path = "intergrax/runtime/events/persistence_contract.py"
    assert classify_r1_protected_drift([path]) == [path]


def test_r1_drift_classifier_flags_evidence_durability_change() -> None:
    path = "intergrax/runtime/events/evidence_durability.py"
    assert classify_r1_protected_drift([path]) == [path]


def test_r1_drift_classifier_flags_stores_subtree() -> None:
    path = "intergrax/runtime/events/stores/sqlite_runtime_event_store.py"
    assert is_r1_protected_production_path(path)
    assert classify_r1_protected_drift([path]) == [path]
