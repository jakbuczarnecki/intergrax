# © Artur Czarnecki. All rights reserved.

"""Unit tests for NPSC-5F/R4 protected production drift classification."""

from __future__ import annotations

import pytest

from testing_support.npsc5f_r4_protected_drift import (
    classify_r4_protected_drift,
    is_r4_protected_production_path,
)

pytestmark = pytest.mark.unit


def test_r4_drift_classifier_allows_docs_change() -> None:
    assert classify_r4_protected_drift(["docs/project/maintainers/qualification/x.md"]) == []


def test_r4_drift_classifier_allows_runtime_event_taxonomy() -> None:
    path = "intergrax/runtime/events/runtime_event.py"
    assert not is_r4_protected_production_path(path)
    assert classify_r4_protected_drift([path]) == []


def test_r4_drift_classifier_allows_asof_projection_surface() -> None:
    path = "intergrax/runtime/events/asof_projection.py"
    assert classify_r4_protected_drift([path]) == []


def test_r4_drift_classifier_blocks_reconstruction_service() -> None:
    path = "intergrax/runtime/observability/historical_reconstruction.py"
    assert is_r4_protected_production_path(path)
    assert classify_r4_protected_drift([path]) == [path]


def test_r4_drift_classifier_blocks_reconstruction_contract() -> None:
    path = "intergrax/contracts/historical_reconstruction.py"
    assert classify_r4_protected_drift([path]) == [path]


def test_r4_drift_classifier_blocks_qualification_gate() -> None:
    path = "tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_asof_bitemporal.py"
    assert classify_r4_protected_drift([path]) == [path]
