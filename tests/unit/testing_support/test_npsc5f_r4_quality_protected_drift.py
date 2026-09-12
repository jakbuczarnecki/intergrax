# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 Final reconstruction quality — protected production drift classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.npsc5f_r4_quality_protected_drift import (
    R4_QUALITY_IMPLEMENTATION_SHA,
    classify_r4_quality_protected_drift,
    collect_r4_quality_protected_production_drift,
    is_r4_quality_protected_production_path,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_r4_quality_drift_classifier_allows_historical_reconstruction() -> None:
    path = "intergrax/runtime/observability/historical_reconstruction.py"
    assert classify_r4_quality_protected_drift([path]) == []


def test_r4_quality_drift_classifier_blocks_execution_reconstruction() -> None:
    path = "intergrax/runtime/diagnostics/execution_reconstruction.py"
    assert is_r4_quality_protected_production_path(path)
    assert classify_r4_quality_protected_drift([path]) == [path]


def test_r4_quality_drift_classifier_allows_qualification_gate() -> None:
    path = "tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_quality.py"
    assert classify_r4_quality_protected_drift([path]) == []


def test_r4_quality_implementation_sha_recorded() -> None:
    assert R4_QUALITY_IMPLEMENTATION_SHA == "84e704eec611e7b24eb82b0be4fe98172c512739"


def test_r4_quality_no_unqualified_protected_drift_since_implementation() -> None:
    drift = collect_r4_quality_protected_production_drift(_REPO_ROOT)
    assert drift == [], f"R4 quality protected production drift since implementation: {drift}"
