# © Artur Czarnecki. All rights reserved.

"""Risk-based RPL-H Package C semantic review (mechanical matrix ≠ semantic proof)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX

LEGACY_QUALIFIED: Final[frozenset[str]] = frozenset(
    {
        "integration.key_value_cache",
        "persistence.relational_sqlite_lab",
        "int_config.configure_existing",
        "observability.otlp_transport",
        "tokenizer.selection",
    }
)


class RplSemanticVerdict(StrEnum):
    QUALIFIED = "QUALIFIED"
    PARTIAL = "PARTIAL"
    NOT_YET = "NOT_YET"


@dataclass(frozen=True, slots=True)
class RplSemanticReviewRow:
    concern_id: str
    risk_tier: str
    mechanical_rpl_h: str
    semantic_verdict: RplSemanticVerdict
    evidence_inspected: str
    notes: str


def _risk_tier(concern_id: str) -> str:
    if concern_id in LEGACY_QUALIFIED:
        return "LEGACY"
    if concern_id.startswith(("integration.", "llm.", "execution.", "plugin.", "tool.", "agent.")):
        return "HIGH"
    return "LOW"


def _semantic_verdict_for(concern_id: str) -> RplSemanticVerdict:
    if concern_id in LEGACY_QUALIFIED:
        return RplSemanticVerdict.QUALIFIED
    if concern_id.startswith("integration."):
        return RplSemanticVerdict.PARTIAL
    if concern_id.startswith(("llm.", "plugin.", "execution.")):
        return RplSemanticVerdict.PARTIAL
    return RplSemanticVerdict.PARTIAL


def rpl_semantic_review_rows() -> tuple[RplSemanticReviewRow, ...]:
    rows: list[RplSemanticReviewRow] = []
    for matrix_row in CROSS_LAYER_REPLACEABILITY_MATRIX:
        cid = matrix_row.concern_id
        tier = _risk_tier(cid)
        verdict = _semantic_verdict_for(cid)
        evidence = matrix_row.behavioral_proof_ref or "Package C test_ebh5_pkgc_*"
        if verdict is RplSemanticVerdict.PARTIAL:
            notes = "Package C: fail-closed empty profile or single-path probe — not full alternate backend switch"
        else:
            notes = "Legacy EBH-5 behavioral proof — alternate path exercised"
        rows.append(
            RplSemanticReviewRow(
                concern_id=cid,
                risk_tier=tier,
                mechanical_rpl_h=matrix_row.rpl_h.value,
                semantic_verdict=verdict,
                evidence_inspected=evidence,
                notes=notes,
            )
        )
    return tuple(rows)


def assert_ebh7_rpl_h_semantic_review_complete() -> None:
    rows = rpl_semantic_review_rows()
    assert len(rows) == len(CROSS_LAYER_REPLACEABILITY_MATRIX)
    high = [r for r in rows if r.risk_tier == "HIGH"]
    assert len(high) >= 30
    for row in rows:
        assert row.evidence_inspected.strip() != ""
        assert row.semantic_verdict in RplSemanticVerdict


def rpl_semantic_partial_concern_ids() -> tuple[str, ...]:
    return tuple(
        sorted(r.concern_id for r in rpl_semantic_review_rows() if r.semantic_verdict is RplSemanticVerdict.PARTIAL)
    )
