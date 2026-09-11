# © Artur Czarnecki. All rights reserved.

"""Unit tests for NPSC-5F/R3 protected production drift classification."""

from __future__ import annotations

import pytest

from testing_support.npsc5f_r3_protected_drift import (
    classify_r3_protected_drift,
    is_r3_protected_production_path,
)

pytestmark = pytest.mark.unit


def test_r3_drift_classifier_allows_docs_change() -> None:
    assert classify_r3_protected_drift(["docs/project/maintainers/qualification/x.md"]) == []


def test_r3_drift_classifier_allows_session_c_scale_surface() -> None:
    path = "intergrax/runtime/resilience/local_dependency_concurrency_admission.py"
    assert not is_r3_protected_production_path(path)
    assert classify_r3_protected_drift([path]) == []


def test_r3_drift_classifier_allows_unrelated_execution_surface() -> None:
    path = "intergrax/runtime/execution/local_execution_capacity_admission.py"
    assert classify_r3_protected_drift([path]) == []


def test_r3_drift_classifier_allows_r2_journal_read_contract() -> None:
    path = "intergrax/runtime/events/unified_run_journal.py"
    assert not is_r3_protected_production_path(path)
    assert classify_r3_protected_drift([path]) == []


def test_r3_drift_classifier_blocks_export_boundary() -> None:
    path = "intergrax/runtime/observability/export_boundary.py"
    assert is_r3_protected_production_path(path)
    assert classify_r3_protected_drift([path]) == [path]


def test_r3_drift_classifier_blocks_journal_export() -> None:
    path = "intergrax/runtime/observability/journal_export.py"
    assert classify_r3_protected_drift([path]) == [path]


def test_r3_drift_classifier_blocks_export_bridge() -> None:
    path = "intergrax/runtime/observability/export_bridge.py"
    assert classify_r3_protected_drift([path]) == [path]


def test_r3_drift_classifier_does_not_block_export_routing() -> None:
    path = "intergrax/runtime/observability/export_routing.py"
    assert not is_r3_protected_production_path(path)
    assert classify_r3_protected_drift([path]) == []
