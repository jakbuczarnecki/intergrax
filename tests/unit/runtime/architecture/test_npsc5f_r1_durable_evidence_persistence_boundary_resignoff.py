# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R1 — durable evidence re-signoff after ``EvidencePersistencePort`` boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.npsc5f_r1_protected_drift import (
    R1_POST_R2_QUALIFIED_BASELINE_SHA,
    collect_r1_protected_production_drift,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_R1_RESIGNOFF_QUAL_PATH = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "NPSC_5F_R1_DURABLE_EVIDENCE_RE_SIGNOFF_PERSISTENCE_BOUNDARY.md"
)


def test_npsc5f_r1_resignoff_qualified_baseline_has_no_protected_drift() -> None:
    drift = collect_r1_protected_production_drift(
        _REPO_ROOT,
        from_sha=R1_POST_R2_QUALIFIED_BASELINE_SHA,
    )
    assert drift == [], f"unqualified R1 protected drift since re-signoff baseline: {drift}"


def test_npsc5f_r1_resignoff_qualification_record_exists() -> None:
    assert _R1_RESIGNOFF_QUAL_PATH.is_file()
    text = _R1_RESIGNOFF_QUAL_PATH.read_text(encoding="utf-8")
    assert "EvidencePersistencePort" in text
    assert R1_POST_R2_QUALIFIED_BASELINE_SHA in text
