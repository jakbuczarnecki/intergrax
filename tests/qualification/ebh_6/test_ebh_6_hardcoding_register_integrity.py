# © Artur Czarnecki. All rights reserved.

"""Integrity gates for EBH-6 CORE_HARDCODING_FINDINGS_REGISTER SSOT."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._findings_register import (
    FindingStatus,
    HardcodingFindingClassification,
    all_hardcoding_findings,
    count_prohibited_unresolved,
    count_unclassified,
    findings_by_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTER_MD = _REPO_ROOT / "docs/project/maintainers/qualification/EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md"


def test_ebh6_findings_unclassified_zero() -> None:
    assert count_unclassified() == 0


def test_ebh6_prohibited_unresolved_zero() -> None:
    assert count_prohibited_unresolved() == 0


def test_ebh6_finding_ids_unique() -> None:
    findings = all_hardcoding_findings()
    ids = [row.finding_id for row in findings]
    assert len(ids) == len(set(ids))


def test_ebh6_sqlite_findings_cover_annex() -> None:
    sqlite_ids = [row.finding_id for row in all_hardcoding_findings() if row.finding_id.startswith("EBH6-SQLITE-")]
    assert len(sqlite_ids) >= 30


def test_ebh6_layer_coverage_findings_complete() -> None:
    layer_rows = [row for row in all_hardcoding_findings() if row.finding_id.startswith("EBH6-LAYER-")]
    assert len(layer_rows) == 16
    assert all(row.status == FindingStatus.OPEN for row in layer_rows)
    assert all(row.classification == HardcodingFindingClassification.DISCOVERY_REQUIRED for row in layer_rows)


def test_ebh6_tracked_debt_findings_ratified() -> None:
    by_id = findings_by_id()
    q2 = by_id["EBH6-DEBT-Q2-D2"]
    bnd = by_id["EBH6-DEBT-BND-COMPAT-R3"]
    assert q2.status == FindingStatus.RATIFIED
    assert bnd.status == FindingStatus.RATIFIED
    assert q2.classification != HardcodingFindingClassification.PROHIBITED_CORE


def test_ebh6_register_md_references_ssot() -> None:
    text = _REGISTER_MD.read_text(encoding="utf-8")
    assert "_findings_register.py" in text
    assert "unclassified" in text.lower()
