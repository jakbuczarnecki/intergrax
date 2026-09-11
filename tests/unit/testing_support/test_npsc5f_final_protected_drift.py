# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — Evidence Plane drift tri-classifier and sentinel."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.npsc5f_final_evidence_plane_drift import (
    EvidencePlaneDriftClass,
    NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA,
    breaking_evidence_plane_drift,
    classify_evidence_plane_drift_path,
    collect_breaking_evidence_plane_production_drift,
    is_evidence_plane_protected_production_path,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_final_drift_classifier_unrelated_execution_surface() -> None:
    path = "intergrax/runtime/execution/retry/service.py"
    assert classify_evidence_plane_drift_path(path) is EvidencePlaneDriftClass.UNRELATED


def test_final_drift_classifier_unrelated_diagnostics() -> None:
    path = "intergrax/runtime/diagnostics/execution_reconstruction.py"
    assert classify_evidence_plane_drift_path(path) is EvidencePlaneDriftClass.UNRELATED


def test_final_drift_classifier_unrelated_application_layer() -> None:
    path = "applications/local_workspace_application/foo.py"
    assert classify_evidence_plane_drift_path(path) is EvidencePlaneDriftClass.UNRELATED


def test_final_drift_classifier_qualified_compatible_qualification_doc() -> None:
    path = "docs/project/maintainers/qualification/NPSC_5F_FINAL_EVIDENCE_PLANE_QUALIFICATION_AND_FREEZE.md"
    assert classify_evidence_plane_drift_path(path) is EvidencePlaneDriftClass.QUALIFIED_COMPATIBLE


def test_final_drift_classifier_qualified_compatible_testing_support() -> None:
    path = "testing_support/npsc5f_final_evidence_plane_drift.py"
    assert classify_evidence_plane_drift_path(path) is EvidencePlaneDriftClass.QUALIFIED_COMPATIBLE


def test_final_drift_classifier_breaking_export_boundary() -> None:
    path = "intergrax/runtime/observability/export_boundary.py"
    assert is_evidence_plane_protected_production_path(path)
    assert classify_evidence_plane_drift_path(path) is EvidencePlaneDriftClass.BREAKING


def test_final_drift_classifier_breaking_runtime_event_taxonomy() -> None:
    path = "intergrax/runtime/events/runtime_event.py"
    assert classify_evidence_plane_drift_path(path) is EvidencePlaneDriftClass.BREAKING


def test_final_drift_classifier_breaking_historical_reconstruction_contract() -> None:
    path = "intergrax/contracts/historical_reconstruction.py"
    assert classify_evidence_plane_drift_path(path) is EvidencePlaneDriftClass.BREAKING


def test_final_drift_classifier_does_not_flag_whole_runtime_sentinel() -> None:
    assert classify_evidence_plane_drift_path("intergrax/runtime/coordinator.py") is EvidencePlaneDriftClass.UNRELATED


def test_final_baseline_sha_recorded() -> None:
    assert NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA == "3bec620ab56417a469487347f68045bf3dec6bd5"


def test_final_no_breaking_protected_drift_since_baseline() -> None:
    drift = collect_breaking_evidence_plane_production_drift(_REPO_ROOT)
    assert drift == [], f"Evidence Plane BREAKING drift since baseline: {drift}"


def test_final_breaking_classifier_lists_only_protected_paths() -> None:
    paths = [
        "docs/foo.md",
        "intergrax/runtime/events/event_bus.py",
        "testing_support/npsc5f_final_evidence_plane_drift.py",
    ]
    assert breaking_evidence_plane_drift(paths) == ["intergrax/runtime/events/event_bus.py"]


def test_npsc5f_final_protected_drift() -> None:
    """Final drift sentinel — BREAKING Evidence Plane production drift must be empty."""
    assert collect_breaking_evidence_plane_production_drift(_REPO_ROOT) == []
