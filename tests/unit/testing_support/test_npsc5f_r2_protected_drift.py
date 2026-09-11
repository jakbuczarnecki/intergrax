# © Artur Czarnecki. All rights reserved.

"""Unit tests for NPSC-5F/R2 protected production drift classification."""

from __future__ import annotations

import pytest

from testing_support.npsc5f_r2_protected_drift import (
    classify_r2_protected_drift,
    is_r2_protected_production_path,
)

pytestmark = pytest.mark.unit


def test_r2_drift_classifier_allows_docs_change() -> None:
    assert classify_r2_protected_drift(["docs/project/maintainers/qualification/x.md"]) == []


def test_r2_drift_classifier_allows_session_d_qualification_file() -> None:
    assert (
        classify_r2_protected_drift(
            ["docs/project/maintainers/qualification/EXECUTION_CERTIFICATION_ACCELERATION_R2.md"],
        )
        == []
    )


def test_r2_drift_classifier_allows_unrelated_execution_surface() -> None:
    path = "intergrax/runtime/execution/local_execution_capacity_admission.py"
    assert not is_r2_protected_production_path(path)
    assert classify_r2_protected_drift([path]) == []


def test_r2_drift_classifier_blocks_unified_run_journal() -> None:
    path = "intergrax/runtime/events/unified_run_journal.py"
    assert is_r2_protected_production_path(path)
    assert classify_r2_protected_drift([path]) == [path]


def test_r2_drift_classifier_blocks_execution_position() -> None:
    path = "intergrax/runtime/events/execution_position.py"
    assert classify_r2_protected_drift([path]) == [path]


def test_r2_drift_classifier_blocks_persistence_read_contract() -> None:
    path = "intergrax/runtime/events/persistence_contract.py"
    assert classify_r2_protected_drift([path]) == [path]


def test_r2_drift_classifier_blocks_store_pagination_surface() -> None:
    path = "intergrax/runtime/events/stores/sqlite_runtime_event_store.py"
    assert is_r2_protected_production_path(path)
    assert classify_r2_protected_drift([path]) == [path]


def test_r2_drift_classifier_does_not_block_journal_export_for_r3_redaction() -> None:
    path = "intergrax/runtime/observability/journal_export.py"
    assert not is_r2_protected_production_path(path)
    assert classify_r2_protected_drift([path]) == []
