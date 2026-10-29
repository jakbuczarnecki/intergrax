# © Artur Czarnecki. All rights reserved.

"""DUP-X Wave 1 — parent closure mechanical gates (register SSOT; not Protocol PASS)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.dup_x._findings_register import (
    DUP_X_CANDIDATE_RECORDS,
    DUP_X_FINDING_RECORDS,
    DUP_X_WAVE1_IMPLEMENTATION_HEAD,
    DupXCandidateRecord,
    DupXFindingRecord,
    FindingKind,
    FindingSeverity,
    FindingStatus,
    count_open_candidates,
    count_open_gaps,
    findings_by_id,
    findings_with_unknown_reachability,
    retained_parallel_without_dossier,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_WAVE1_GAP_IDS: tuple[str, ...] = (
    "DUPX-GAP-CLASS-D",
    "DUPX-GAP-L3-TIER3-DEDUPE",
)

_WAVE1_DELTA_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/DUP_X_CLASS_D_INTEGRATION_SELECTOR_CALLER_MATRIX.md",
    "docs/project/maintainers/qualification/DUP_X_TIER3_APPLICATION_HOST_DEDUPE_DISPOSITION.md",
    "docs/project/maintainers/qualification/DUP_X_RESPONSIBILITY_DELTA_observability_export_routing.md",
)


def test_dup_x_parent_closure_zero_open_gaps() -> None:
    assert count_open_gaps() == 0


def test_dup_x_parent_closure_zero_open_candidates() -> None:
    assert count_open_candidates() == 0


def test_dup_x_parent_closure_no_unknown_reachability() -> None:
    assert not findings_with_unknown_reachability()


def test_dup_x_parent_closure_gap_records_present_with_expected_ids() -> None:
    by_id = findings_by_id()
    for gap_id in _WAVE1_GAP_IDS:
        assert gap_id in by_id
        row = by_id[gap_id]
        assert row.kind is FindingKind.GAP
        assert row.status is FindingStatus.PASS


def test_dup_x_parent_closure_wave1_delta_docs_committed() -> None:
    for rel in _WAVE1_DELTA_DOCS:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_dup_x_parent_closure_retained_parallel_dossiers_complete() -> None:
    assert not retained_parallel_without_dossier()


def test_dup_x_parent_closure_wave1_head_documented_in_operator_register() -> None:
    reg = (_REPO_ROOT / "docs/project/maintainers/qualification/DUP_X_OPERATOR_DECISION_REGISTER.md").read_text(
        encoding="utf-8"
    )
    assert DUP_X_WAVE1_IMPLEMENTATION_HEAD in reg


def test_dup_x_parent_closure_negative_open_gap_injected_state_fails_gate() -> None:
    injected = DUP_X_FINDING_RECORDS + (
        DupXFindingRecord(
            finding_id="DUPX-GAP-INJECTED-TEST",
            severity=FindingSeverity.GAP,
            dup_classes=("DUP-10",),
            kind=FindingKind.GAP,
            status=FindingStatus.OPEN,
            source_paths=("tests/injected",),
            symbols=("injected",),
            production_reachability="unknown — injected",
            canonical_owners=("tests/injected",),
            remediation="test-only",
        ),
    )
    assert count_open_gaps(injected) == 1
    assert findings_with_unknown_reachability(injected) == ("DUPX-GAP-INJECTED-TEST",)


def test_dup_x_parent_closure_negative_open_candidate_injected_state_fails_gate() -> None:
    injected = DUP_X_CANDIDATE_RECORDS + (
        DupXCandidateRecord(
            candidate_id="DUPX-CAND-INJECTED-TEST",
            dup_classes=("DUP-10",),
            status=FindingStatus.OPEN,
            disposition="injected",
            parallel_retained=False,
            delta_dossier=None,
        ),
    )
    assert count_open_candidates(injected) == 1


def test_dup_x_cand_09_observability_routing_uses_operator_wiring_not_shadow_slug() -> None:
    lkw_obs = _REPO_ROOT / "applications/local_workspace_application/host/observability_wiring.py"
    text = lkw_obs.read_text(encoding="utf-8")
    assert "ObservabilityExportBackendRegistry" in text
    assert "build_observability_export_integration" in text
    assert "read_integration_slug_from_env" not in text
