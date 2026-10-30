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
    rows: list[RplHDispositionRow] = []
    for matrix_row in CROSS_LAYER_REPLACEABILITY_MATRIX:
        qualified = matrix_row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
        q_status = CertificationStatus.QUALIFIED if qualified else CertificationStatus.QUALIFICATION_PENDING
        rows.append(
            RplHDispositionRow(
                concern_id=matrix_row.concern_id,
                rpl_h_at_head=matrix_row.rpl_h,
                materialization_status=CertificationStatus.MATERIALIZED,
                qualification_status=q_status,
                existing_evidence=(matrix_row.behavioral_proof_ref if qualified else "NONE qualified @ HEAD"),
                missing_structural="PASS" if matrix_row.rpl_b.value == "PASS" else "RPL-B OPEN",
                missing_behavioral="PASS" if matrix_row.rpl_e.value == "PASS" else "RPL-E OPEN",
                remediation="QUALIFY per EBH-5 matrix" if not qualified else "N/A",
            )
        )
    return tuple(rows)


def assert_ebh7_rpl_h_disposition_integrity() -> None:
    assert len(CROSS_LAYER_REPLACEABILITY_MATRIX) == EBH7_RPL_MATRIX_ROWS
    rows = rpl_h_disposition_rows()
    assert len(rows) == EBH7_RPL_MATRIX_ROWS
    qualified = sum(1 for r in rows if is_behavioral_certification(r.qualification_status))
    not_yet = sum(1 for r in rows if r.rpl_h_at_head == ExtensibilityClassification.REPLACEABLE_NOT_YET)
    assert qualified + not_yet == EBH7_RPL_MATRIX_ROWS


def assert_ebh7_rpl_h_behavioral_qualification() -> None:
    rows = rpl_h_disposition_rows()
    pending = [r.concern_id for r in rows if not is_behavioral_certification(r.qualification_status)]
    assert pending == [], f"RPL-H behavioral qualification OPEN for: {pending[:5]}... ({len(pending)} total)"
