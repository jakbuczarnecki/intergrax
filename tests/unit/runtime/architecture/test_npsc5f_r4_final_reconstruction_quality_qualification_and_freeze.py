# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 Final — reconstruction quality qualification and freeze."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.npsc5f_r4_quality_protected_drift import (
    R4_QUALITY_IMPLEMENTATION_SHA,
    collect_r4_quality_protected_production_drift,
)
from testing_support.npsc5f_r4_quality_regression_matrix import (
    MANDATORY_REGRESSION_SUITES,
    run_mandatory_regression_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

R3_FINAL_SHA = "0346face3ef68d8f21504822a26f8f45f2384cf9"
R2_FINAL_SHA = "76c92847f67da22d97943b55896a88c814d7e39d"
R1_FINAL_SHA = "455c09f342f995ac0a6fcb03ffef2f4d3e36a447"


@pytest.mark.gate
def test_r4_quality_mandatory_regression_matrix() -> None:
    proc = run_mandatory_regression_matrix(_REPO_ROOT)
    assert proc.returncode == 0, (
        f"R4 reconstruction quality Final regression matrix failed:\n{proc.stdout}\n{proc.stderr}"
    )


def test_r4_quality_final_mandatory_suite_labels_recorded() -> None:
    labels = [label for label, _ in MANDATORY_REGRESSION_SUITES]
    assert "R4 reconstruction quality gate" in labels
    assert "R4 quality Final drift sentinel" in labels
    assert "Execution reconstruction" in labels


def test_r4_quality_final_canonical_predecessor_shas_recorded() -> None:
    assert R4_QUALITY_IMPLEMENTATION_SHA == "84e704eec611e7b24eb82b0be4fe98172c512739"
    assert R3_FINAL_SHA.startswith("0346fac")
    assert R2_FINAL_SHA.startswith("76c9284")
    assert R1_FINAL_SHA.startswith("455c09f")


def test_r4_quality_final_no_unqualified_protected_drift_since_implementation() -> None:
    drift = collect_r4_quality_protected_production_drift(_REPO_ROOT)
    assert drift == [], f"R4 quality protected production drift since implementation: {drift}"


@pytest.mark.gate
def test_npsc5f_r4_final_reconstruction_quality_qualification_gate() -> None:
    doc = (
        _REPO_ROOT
        / "docs/project/maintainers/qualification/NPSC_5F_R4_FINAL_RECONSTRUCTION_QUALITY_QUALIFICATION_AND_FREEZE.md"
    )
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "FROZEN / PASS" in text
    assert R4_QUALITY_IMPLEMENTATION_SHA in text
