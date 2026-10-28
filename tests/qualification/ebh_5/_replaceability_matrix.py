# © Artur Czarnecki. All rights reserved.

"""CROSS_LAYER_REPLACEABILITY_MATRIX SSOT (RPL-A..H)."""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from tests.qualification.ebh_5._behavioral_proof_registry import (
    QUALIFIED_CONCERN_ALTERNATE_REF,
    QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE,
    QUALIFIED_CONCERN_REGRESSION_GATES,
)
from tests.qualification.ebh_5._rpl_f_evidence_registry import rpl_f_gate_for_qualified
from tests.qualification.ebh_5._vendor_backend_inventory import (
    concerns_with_extensibility_yes,
)
from tests.qualification.ebh_5._vendor_backend_types import (
    ExtensibilityClassification,
    ExtensibilityPromised,
    ReplaceabilityMatrixRow,
    RplGateStatus,
)


def _qualified_row(
    concern_id: str,
    *,
    behavioral: str,
    alternate: str,
    gates: tuple[str, ...],
    rpl_f: RplGateStatus = RplGateStatus.NA,
) -> ReplaceabilityMatrixRow:
    return ReplaceabilityMatrixRow(
        concern_id=concern_id,
        rpl_a=RplGateStatus.PASS,
        rpl_b=RplGateStatus.PASS,
        rpl_c=RplGateStatus.PASS,
        rpl_d=RplGateStatus.PASS,
        rpl_e=RplGateStatus.PASS,
        rpl_f=rpl_f,
        rpl_g=RplGateStatus.PASS,
        rpl_h=ExtensibilityClassification.REPLACEABLE_QUALIFIED,
        alternate_implementation_ref=alternate,
        behavioral_proof_ref=behavioral,
        regression_gate_ids=gates,
    )


def _not_yet_row(concern_id: str, *, gates: tuple[str, ...] = ("ebh5-matrix-integrity",)) -> ReplaceabilityMatrixRow:
    return ReplaceabilityMatrixRow(
        concern_id=concern_id,
        rpl_a=RplGateStatus.PASS,
        rpl_b=RplGateStatus.FAIL,
        rpl_c=RplGateStatus.PASS,
        rpl_d=RplGateStatus.PASS,
        rpl_e=RplGateStatus.FAIL,
        rpl_f=RplGateStatus.NA,
        rpl_g=RplGateStatus.PASS,
        rpl_h=ExtensibilityClassification.REPLACEABLE_NOT_YET,
        alternate_implementation_ref="PENDING",
        behavioral_proof_ref="PENDING",
        regression_gate_ids=gates,
    )


def _qualified_override(concern_id: str) -> ReplaceabilityMatrixRow:
    return _qualified_row(
        concern_id,
        behavioral=QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE[concern_id],
        alternate=QUALIFIED_CONCERN_ALTERNATE_REF[concern_id],
        gates=QUALIFIED_CONCERN_REGRESSION_GATES[concern_id],
        rpl_f=rpl_f_gate_for_qualified(concern_id),
    )


_QUALIFIED_OVERRIDES: dict[str, ReplaceabilityMatrixRow] = {
    concern_id: _qualified_override(concern_id) for concern_id in QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE
}


@lru_cache(maxsize=1)
def replaceability_matrix_rows() -> tuple[ReplaceabilityMatrixRow, ...]:
    rows: list[ReplaceabilityMatrixRow] = []
    for concern in concerns_with_extensibility_yes():
        if concern.concern_id in _QUALIFIED_OVERRIDES:
            rows.append(_QUALIFIED_OVERRIDES[concern.concern_id])
            continue
        if concern.extensibility_classification == ExtensibilityClassification.REPLACEABLE_QUALIFIED:
            rows.append(
                _qualified_row(
                    concern.concern_id,
                    behavioral=concern.behavioral_qualification_evidence_id,
                    alternate=concern.structural_replaceability_evidence_id,
                    gates=("test_ebh_5_l2_integrity_gates",),
                ),
            )
            continue
        if concern.extensibility_promised == ExtensibilityPromised.YES:
            rows.append(_not_yet_row(concern.concern_id))
            continue
        if concern.extensibility_promised == ExtensibilityPromised.PARTIAL:
            rows.append(_not_yet_row(concern.concern_id))
    return tuple(rows)


CROSS_LAYER_REPLACEABILITY_MATRIX: Final[tuple[ReplaceabilityMatrixRow, ...]] = replaceability_matrix_rows()


def matrix_by_concern_id() -> dict[str, ReplaceabilityMatrixRow]:
    return {row.concern_id: row for row in CROSS_LAYER_REPLACEABILITY_MATRIX}


def undocumented_qualified_rows() -> tuple[str, ...]:
    from tests.qualification.ebh_5._proof_validation import undocumented_qualified_rows as _invalid

    return _invalid(CROSS_LAYER_REPLACEABILITY_MATRIX)
