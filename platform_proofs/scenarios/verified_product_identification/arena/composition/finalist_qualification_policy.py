"""Composition-level finalist qualification presets and decision mapping."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    BASELINE_CANDIDATE_ID,
    QWEN_CANDIDATE_ID,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidate_selection import (
    EmbeddingArenaCandidateSelection,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaDecision,
    FinalistQualificationGate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.finalist_qualification import (
    FinalistQualificationSelection,
)

FINALIST_BGE_QWEN_CANDIDATE_SELECTION = EmbeddingArenaCandidateSelection(
    candidate_ids=(BASELINE_CANDIDATE_ID, QWEN_CANDIDATE_ID),
)

FINALIST_BGE_QWEN_QUALIFICATION_SELECTION = FinalistQualificationSelection(
    baseline_candidate_id=BASELINE_CANDIDATE_ID,
    challenger_candidate_id=QWEN_CANDIDATE_ID,
)


def resolve_finalist_qualification_selection(
    selection: EmbeddingArenaCandidateSelection,
) -> FinalistQualificationSelection:
    if selection.candidate_ids == FINALIST_BGE_QWEN_CANDIDATE_SELECTION.candidate_ids:
        return FINALIST_BGE_QWEN_QUALIFICATION_SELECTION
    msg = (
        "unsupported finalist qualification candidate selection "
        f"{selection.candidate_ids!r}; no typed baseline/challenger mapping configured"
    )
    raise ValueError(msg)


def map_finalist_gate_to_decision(
    gate: FinalistQualificationGate,
    qualification_selection: FinalistQualificationSelection,
) -> EmbeddingArenaDecision:
    if qualification_selection is FINALIST_BGE_QWEN_QUALIFICATION_SELECTION:
        return _map_bge_qwen_finalist_gate_to_decision(gate)
    msg = (
        "unsupported finalist qualification selection for decision mapping: "
        f"baseline={qualification_selection.baseline_candidate_id!r}, "
        f"challenger={qualification_selection.challenger_candidate_id!r}"
    )
    raise ValueError(msg)


def _map_bge_qwen_finalist_gate_to_decision(
    gate: FinalistQualificationGate,
) -> EmbeddingArenaDecision:
    if gate is FinalistQualificationGate.CHALLENGER_CLEAR_WIN:
        return EmbeddingArenaDecision.PROMOTE_QWEN3_0_6B_CANDIDATE
    if gate is FinalistQualificationGate.BASELINE_CLEAR_WIN:
        return EmbeddingArenaDecision.KEEP_BGE_M3
    if gate is FinalistQualificationGate.QUALITY_REGRESSION:
        return EmbeddingArenaDecision.KEEP_BGE_M3
    if gate is FinalistQualificationGate.RUNTIME_REJECTED:
        return EmbeddingArenaDecision.NO_CLEAR_WINNER
    if gate is FinalistQualificationGate.AMBIGUOUS:
        return EmbeddingArenaDecision.MORE_EVIDENCE_REQUIRED
    return EmbeddingArenaDecision.NO_CLEAR_WINNER
