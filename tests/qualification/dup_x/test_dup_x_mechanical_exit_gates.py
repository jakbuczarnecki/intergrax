# © Artur Czarnecki. All rights reserved.

"""DUP-X mechanical exit criteria — roadmap §3.0.3 measurement @ handoff (not stage CLOSED)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.dup_x._closed_world_inventory import assert_dup_x_closed_world_integrity
from tests.qualification.dup_x._findings_register import (
    DUP_X_AUDITED_HEAD,
    DUP_X_CANDIDATE_RECORDS,
    FindingKind,
    count_open_bypass_unresolved,
    count_open_gaps,
    count_unresolved_duplicate_blockers,
    count_unresolved_shadow_authority,
    findings_by_id,
    retained_parallel_without_dossier,
)
from tests.qualification.ebh_5._closed_world_discovery import COMPOSITION_ROOT_HINTS
from tests.qualification.ebh_6._config_x_reconciliation import assert_ebh6_config_x_fully_reconciled

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_MANDATORY_DUP_X_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/DUP_X_C0_BASELINE_AUDIT_RECORD.md",
    "docs/project/maintainers/qualification/DUP_X_CLOSED_WORLD_INVENTORY.md",
    "docs/project/maintainers/qualification/DUP_X_CLASS_COVERAGE_MATRIX.md",
    "docs/project/maintainers/qualification/DUP_X_FINDINGS_REGISTER.md",
    "docs/project/maintainers/qualification/DUP_X_OPERATOR_DECISION_REGISTER.md",
    "docs/project/maintainers/qualification/DUP_X_CLOSURE_MATRIX.md",
    "docs/project/maintainers/qualification/DUP_X_PARENT_CERTIFICATION_DOSSIER.md",
    "docs/project/maintainers/qualification/DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PACKAGE.md",
    "docs/project/maintainers/qualification/DUP_X_CLASS_D_INTEGRATION_SELECTOR_CALLER_MATRIX.md",
    "docs/project/maintainers/qualification/DUP_X_TIER3_APPLICATION_HOST_DEDUPE_DISPOSITION.md",
    "docs/project/maintainers/qualification/DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md",
    "docs/project/maintainers/qualification/DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md",
    "docs/project/maintainers/qualification/DUP_X_B4_NPSC_NEGATIVE_BOUNDARY_EVIDENCE_RECORD.md",
)


def test_dup_x_section_13_measure_unresolved_duplicate_blockers() -> None:
    """§3.0.3 zero — recorded count (stage not CLOSED)."""
    assert count_unresolved_duplicate_blockers() == 0


def test_dup_x_section_13_measure_shadow_authority() -> None:
    assert count_unresolved_shadow_authority() == 0


def test_dup_x_section_13_measure_open_bypass() -> None:
    assert count_open_bypass_unresolved() == 0


def test_dup_x_section_13_open_gaps_registered() -> None:
    """Gap rows remain in SSOT with stable IDs after Wave 1 closure (open count must be zero)."""
    by_id = findings_by_id()
    for gap_id in ("DUPX-GAP-CLASS-D", "DUPX-GAP-L3-TIER3-DEDUPE"):
        assert gap_id in by_id
        assert by_id[gap_id].kind is FindingKind.GAP
    assert count_open_gaps() == 0


def test_dup_x_section_13_retained_parallel_delta_proof() -> None:
    assert not retained_parallel_without_dossier()


def test_dup_x_section_13_candidates_classified() -> None:
    assert len(DUP_X_CANDIDATE_RECORDS) == 15


def test_dup_x_exit_closed_world_replay() -> None:
    assert_dup_x_closed_world_integrity()


def test_dup_x_exit_config_x_reconciled() -> None:
    assert_ebh6_config_x_fully_reconciled()


def test_dup_x_exit_mandatory_docs_present() -> None:
    for rel in _MANDATORY_DUP_X_DOCS:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_dup_x_exit_composition_hints_normative_in_config_x() -> None:
    from tests.qualification.config_x._config_x_discovery import discover_composition_root_paths

    union = discover_composition_root_paths()
    for hint in COMPOSITION_ROOT_HINTS:
        if (_REPO_ROOT / hint).is_file():
            assert hint in union, hint


def test_dup_x_exit_audited_head_documented() -> None:
    c0 = (_REPO_ROOT / _MANDATORY_DUP_X_DOCS[0]).read_text(encoding="utf-8")
    assert DUP_X_AUDITED_HEAD in c0


def test_dup_x_exit_operator_gates_not_self_closed() -> None:
    reg = (_REPO_ROOT / "docs/project/maintainers/qualification/DUP_X_OPERATOR_DECISION_REGISTER.md").read_text(
        encoding="utf-8"
    )
    assert "OD-DUPX-08" in reg and "APPROVED" in reg
    assert "OD-DUPX-02" in reg
    assert "OD-DUPX-01" in reg
    dossier = (_REPO_ROOT / "docs/project/maintainers/qualification/DUP_X_PARENT_CERTIFICATION_DOSSIER.md").read_text(
        encoding="utf-8"
    )
    assert "CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)" in dossier
