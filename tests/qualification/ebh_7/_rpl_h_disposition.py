# © Artur Czarnecki. All rights reserved.

"""EBH-7 RPL-H disposition — structural inventory vs behavioral qualification."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification
from tests.qualification.ebh_7._certification_status import CertificationStatus, is_behavioral_certification
from tests.qualification.ebh_7._constants import EBH7_RPL_MATRIX_ROWS


@dataclass(frozen=True, slots=True)
class RplHDispositionRow:
    concern_id: str
    rpl_h_at_head: ExtensibilityClassification
    materialization_status: CertificationStatus
    qualification_status: CertificationStatus
    existing_evidence: str
    missing_structural: str
    missing_behavioral: str
    remediation: str


def rpl_h_disposition_rows() -> tuple[RplHDispositionRow, ...]:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import normative_rpl_row
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

    rows: list[RplHDispositionRow] = []
    for matrix_row in CROSS_LAYER_REPLACEABILITY_MATRIX:
        norm = normative_rpl_row(matrix_row.concern_id)
        qualified = norm.semantic_verdict is RplSemanticVerdict.QUALIFIED
        q_status = CertificationStatus.QUALIFIED if qualified else CertificationStatus.QUALIFICATION_PENDING
        rows.append(
            RplHDispositionRow(
                concern_id=matrix_row.concern_id,
                rpl_h_at_head=matrix_row.rpl_h,
                materialization_status=CertificationStatus.MATERIALIZED,
                qualification_status=q_status,
                existing_evidence=norm.positive_test if qualified else norm.defect,
                missing_structural="PASS" if matrix_row.rpl_b.value == "PASS" else "RPL-B OPEN",
                missing_behavioral="PASS" if qualified else norm.defect,
                remediation=norm.task_id if not qualified else "N/A",
            )
        )
    return tuple(rows)


def assert_ebh7_rpl_h_disposition_integrity() -> None:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import all_normative_rpl_rows
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

    assert len(CROSS_LAYER_REPLACEABILITY_MATRIX) == EBH7_RPL_MATRIX_ROWS
    rows = rpl_h_disposition_rows()
    assert len(rows) == EBH7_RPL_MATRIX_ROWS
    norm_qualified = sum(1 for r in all_normative_rpl_rows() if r.semantic_verdict is RplSemanticVerdict.QUALIFIED)
    qualified = sum(1 for r in rows if is_behavioral_certification(r.qualification_status))
    assert qualified == norm_qualified


def assert_ebh7_rpl_h_behavioral_qualification() -> None:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import all_normative_rpl_rows
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

    rows = rpl_h_disposition_rows()
    for norm in all_normative_rpl_rows():
        disp = next(r for r in rows if r.concern_id == norm.concern_id)
        if norm.semantic_verdict is RplSemanticVerdict.QUALIFIED:
            assert is_behavioral_certification(disp.qualification_status), norm.concern_id
        else:
            assert not is_behavioral_certification(disp.qualification_status), norm.concern_id


def assert_ebh7_rpl_h_semantic_qualification_for_b6() -> None:
    """B6 readiness — agent semantic verdicts must match normative SSOT (fail-closed)."""
    from tests.qualification.ebh_7._rpl_h_normative_evidence import (
        assert_ebh7_rpl_h_normative_evidence_integrity,
        normative_rpl_verdict_counts,
    )
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import rpl_semantic_verdict_counts

    assert_ebh7_rpl_h_normative_evidence_integrity()
    agent = rpl_semantic_verdict_counts()
    norm = normative_rpl_verdict_counts()
    assert agent == norm, f"Agent vs normative RPL mismatch: agent={agent} norm={norm}"
