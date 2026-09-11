# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 Final — historical reconstruction as-of bitemporal qualification and freeze."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.npsc5f_r4_protected_drift import (
    R4_IMPLEMENTATION_SHA,
    collect_r4_protected_production_drift,
)
from testing_support.npsc5f_r4_regression_matrix import (
    MANDATORY_REGRESSION_SUITES,
    run_mandatory_regression_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

R3_FINAL_SHA = "0346face3ef68d8f21504822a26f8f45f2384cf9"
R3_IMPLEMENTATION_SHA = R3_FINAL_SHA
R2_FINAL_SHA = "76c92847f67da22d97943b55896a88c814d7e39d"
R1_FINAL_SHA = "455c09f342f995ac0a6fcb03ffef2f4d3e36a447"
NPSC_5F_P0_SHA = "7811371da1069b661987b050a4c9bf42c02bda69"
NPSC_5E_FINAL_SHA = "fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7"


@pytest.mark.gate
def test_mandatory_regression_matrix_passes() -> None:
    """One ``uv run pytest`` for the full matrix (no per-suite subprocess fan-out)."""
    proc = run_mandatory_regression_matrix(_REPO_ROOT)
    assert proc.returncode == 0, (
        f"R4 Final mandatory regression matrix failed:\n{proc.stdout}\n{proc.stderr}"
    )


def test_r4_final_mandatory_suite_labels_recorded() -> None:
    labels = [label for label, _ in MANDATORY_REGRESSION_SUITES]
    assert "R4 implementation gate" in labels
    assert "R4 Final drift sentinel" in labels
    assert "NPSC-5E Final" in labels


def test_r4_final_canonical_predecessor_shas_recorded() -> None:
    assert R4_IMPLEMENTATION_SHA == "37fb051c7f164d705f628760436b8ea10ee0289f"
    assert R3_FINAL_SHA.startswith("0346fac")
    assert R2_FINAL_SHA.startswith("76c9284")
    assert R1_FINAL_SHA.startswith("455c09f")
    assert NPSC_5F_P0_SHA.startswith("7811371")
    assert NPSC_5E_FINAL_SHA.startswith("fabdcfe")


def test_r4_final_no_unqualified_protected_drift_since_implementation() -> None:
    drift = collect_r4_protected_production_drift(_REPO_ROOT)
    assert drift == [], f"R4 protected production drift since implementation: {drift}"


@pytest.mark.gate
def test_npsc5f_r4_final_qualification_gate() -> None:
    assert R4_IMPLEMENTATION_SHA == "37fb051c7f164d705f628760436b8ea10ee0289f"
    assert R3_IMPLEMENTATION_SHA == R3_FINAL_SHA
