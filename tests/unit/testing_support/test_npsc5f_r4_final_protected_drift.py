# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 Final — protected production drift classification and sentinel."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.npsc5f_r4_protected_drift import (
    R4_IMPLEMENTATION_SHA,
    classify_r4_protected_drift,
    collect_r4_protected_production_drift,
    is_r4_protected_production_path,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_r4_final_drift_classifier_allows_docs_change() -> None:
    assert classify_r4_protected_drift(["docs/project/maintainers/qualification/x.md"]) == []


def test_r4_final_drift_classifier_allows_runtime_event_taxonomy() -> None:
    path = "intergrax/runtime/events/runtime_event.py"
    assert not is_r4_protected_production_path(path)
    assert classify_r4_protected_drift([path]) == []


def test_r4_final_drift_classifier_allows_asof_projection_surface() -> None:
    path = "intergrax/runtime/events/asof_projection.py"
    assert classify_r4_protected_drift([path]) == []


def test_r4_final_drift_classifier_allows_execution_reconstruction_diag() -> None:
    path = "intergrax/runtime/diagnostics/execution_reconstruction.py"
    assert classify_r4_protected_drift([path]) == []


def test_r4_final_drift_classifier_blocks_reconstruction_service() -> None:
    path = "intergrax/runtime/observability/historical_reconstruction.py"
    assert is_r4_protected_production_path(path)
    assert classify_r4_protected_drift([path]) == [path]


def test_r4_final_drift_classifier_blocks_reconstruction_contract() -> None:
    path = "intergrax/contracts/historical_reconstruction.py"
    assert classify_r4_protected_drift([path]) == [path]


def test_r4_final_drift_classifier_blocks_qualification_gate() -> None:
    path = "tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_asof_bitemporal.py"
    assert classify_r4_protected_drift([path]) == [path]


def test_r4_final_implementation_sha_recorded() -> None:
    assert R4_IMPLEMENTATION_SHA == "37fb051c7f164d705f628760436b8ea10ee0289f"


def test_r4_final_no_unqualified_protected_drift_since_implementation() -> None:
    drift = collect_r4_protected_production_drift(_REPO_ROOT)
    assert drift == [], f"R4 protected production drift since implementation: {drift}"


def test_npsc5f_r4_final_protected_drift() -> None:
    """R4 Final drift sentinel — unqualified protected production drift must be empty."""
    assert collect_r4_protected_production_drift(_REPO_ROOT) == []
