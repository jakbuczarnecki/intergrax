# © Artur Czarnecki. All rights reserved.

"""Risk-based RPL-H Package C semantic review — evidence-derived verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
from tests.qualification.ebh_7._rpl_h_semantic_evidence import (
    RplSemanticVerdict,
    all_rpl_concern_semantic_evidence,
    rpl_concern_semantic_evidence,
)


@dataclass(frozen=True, slots=True)
class RplSemanticReviewRow:
    concern_id: str
    risk_tier: str
    mechanical_rpl_h: str
    semantic_verdict: RplSemanticVerdict
    evidence_inspected: str
    notes: str


def _risk_tier(concern_id: str) -> str:
    if concern_id.startswith(("integration.", "llm.", "execution.", "plugin.", "tool.", "agent.")):
        return "HIGH"
    return "LOW"


def rpl_semantic_review_rows() -> tuple[RplSemanticReviewRow, ...]:
    matrix_by_id = {row.concern_id: row for row in CROSS_LAYER_REPLACEABILITY_MATRIX}
    rows: list[RplSemanticReviewRow] = []
    for evidence in all_rpl_concern_semantic_evidence():
        matrix_row = matrix_by_id[evidence.concern_id]
        rows.append(
            RplSemanticReviewRow(
                concern_id=evidence.concern_id,
                risk_tier=_risk_tier(evidence.concern_id),
                mechanical_rpl_h=matrix_row.rpl_h.value,
                semantic_verdict=evidence.verdict,
                evidence_inspected=evidence.positive_test,
                notes=evidence.notes,
            )
        )
    return tuple(rows)


def assert_ebh7_rpl_h_semantic_review_complete() -> None:
    rows = rpl_semantic_review_rows()
    assert len(rows) == len(CROSS_LAYER_REPLACEABILITY_MATRIX)
    for row in rows:
        assert row.evidence_inspected.strip() != ""
        assert row.semantic_verdict in RplSemanticVerdict
        ev = rpl_concern_semantic_evidence(row.concern_id)
        assert ev.verdict is row.semantic_verdict


def rpl_semantic_partial_concern_ids() -> tuple[str, ...]:
    return tuple(
        sorted(r.concern_id for r in rpl_semantic_review_rows() if r.semantic_verdict is RplSemanticVerdict.PARTIAL)
    )


def rpl_semantic_blocked_concern_ids() -> tuple[str, ...]:
    return tuple(
        sorted(r.concern_id for r in rpl_semantic_review_rows() if r.semantic_verdict is RplSemanticVerdict.BLOCKED)
    )


# deprecated — see _rpl_h_semantic_evidence.LEGACY_BEHAVIORAL_QUALIFIED
LEGACY_QUALIFIED: Final[frozenset[str]] = frozenset()
PACKAGE_C_V2_QUALIFIED: Final[frozenset[str]] = frozenset()  # removed — no static allowlist
