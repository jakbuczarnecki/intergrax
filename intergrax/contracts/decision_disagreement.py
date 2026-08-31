# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed disagreement artifact contracts (DS-DELIB-03).

Structured record of factual positions and conflicts between competing proposal
branches. Describes what disagreements exist — not how to resolve them.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_record import DecisionLineageRef
from intergrax.contracts.evidence_claims import (
    EvidenceReferenceId,
    validate_evidence_reference_id,
)


def _validate_canonical_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


def _lineage_ref_sort_key(ref: DecisionLineageRef) -> tuple[int, str]:
    return (ref.version.value, ref.branch_id)


def _canonicalize_lineage_refs(
    refs: tuple[DecisionLineageRef, ...],
    *,
    field_name: str,
    minimum: int,
) -> tuple[DecisionLineageRef, ...]:
    if minimum < 0:
        raise ValueError("minimum must be >= 0")
    normalized: list[DecisionLineageRef] = []
    seen: set[tuple[int, str]] = set()
    for ref in refs:
        if type(ref) is not DecisionLineageRef:
            raise TypeError(f"{field_name} must contain DecisionLineageRef")
        key = _lineage_ref_sort_key(ref)
        if key in seen:
            raise ValueError(f"{field_name} must not contain duplicates")
        seen.add(key)
        normalized.append(ref)
    if len(normalized) < minimum:
        raise ValueError(f"{field_name} must contain at least {minimum} entries")
    return tuple(sorted(normalized, key=_lineage_ref_sort_key))


def _canonicalize_evidence_refs(
    refs: tuple[EvidenceReferenceId, ...],
    *,
    field_name: str,
) -> tuple[EvidenceReferenceId, ...]:
    normalized: list[EvidenceReferenceId] = []
    seen: set[EvidenceReferenceId] = set()
    for item in refs:
        evidence_id = validate_evidence_reference_id(item)
        if evidence_id in seen:
            raise ValueError(f"{field_name} must not contain duplicates")
        seen.add(evidence_id)
        normalized.append(evidence_id)
    return tuple(sorted(normalized, key=str))


def _proposal_ref_keys(
    proposal_refs: tuple[DecisionLineageRef, ...],
) -> set[tuple[int, str]]:
    return {_lineage_ref_sort_key(ref) for ref in proposal_refs}


def _require_known_proposal_refs(
    refs: tuple[DecisionLineageRef, ...],
    *,
    field_name: str,
    known: set[tuple[int, str]],
) -> None:
    for ref in refs:
        if _lineage_ref_sort_key(ref) not in known:
            raise ValueError(f"{field_name} must reference known proposal refs")


def _require_canonical_lineage_refs(
    refs: tuple[DecisionLineageRef, ...],
    *,
    field_name: str,
    minimum: int,
) -> None:
    canonical = _canonicalize_lineage_refs(
        refs,
        field_name=field_name,
        minimum=minimum,
    )
    if refs != canonical:
        raise ValueError(f"{field_name} must be in canonical order without duplicates")


def _require_canonical_evidence_refs(
    refs: tuple[EvidenceReferenceId, ...],
    *,
    field_name: str,
) -> None:
    canonical = _canonicalize_evidence_refs(refs, field_name=field_name)
    if refs != canonical:
        raise ValueError(f"{field_name} must be in canonical order without duplicates")


@dataclass(frozen=True, slots=True)
class DisagreementPosition:
    """Auditable stance summary for one competing proposal branch."""

    proposal_ref: DecisionLineageRef
    summary: str
    evidence_refs: tuple[EvidenceReferenceId, ...] = ()

    def __post_init__(self) -> None:
        if type(self.proposal_ref) is not DecisionLineageRef:
            raise TypeError("DisagreementPosition.proposal_ref must be DecisionLineageRef")
        _validate_canonical_string(self.summary, "DisagreementPosition.summary")
        _require_canonical_evidence_refs(
            self.evidence_refs,
            field_name="DisagreementPosition.evidence_refs",
        )


@dataclass(frozen=True, slots=True)
class DisagreementConflict:
    """One explicit conflict dimension across competing proposal branches."""

    dimension: str
    proposal_refs: tuple[DecisionLineageRef, ...]
    summary: str

    def __post_init__(self) -> None:
        _validate_canonical_string(self.dimension, "DisagreementConflict.dimension")
        _validate_canonical_string(self.summary, "DisagreementConflict.summary")
        _require_canonical_lineage_refs(
            self.proposal_refs,
            field_name="DisagreementConflict.proposal_refs",
            minimum=2,
        )


@dataclass(frozen=True, slots=True)
class UnresolvedQuestion:
    """Open question surfaced during deliberation without resolution semantics."""

    question: str
    related_proposal_refs: tuple[DecisionLineageRef, ...] = ()
    evidence_refs: tuple[EvidenceReferenceId, ...] = ()

    def __post_init__(self) -> None:
        _validate_canonical_string(self.question, "UnresolvedQuestion.question")
        _require_canonical_lineage_refs(
            self.related_proposal_refs,
            field_name="UnresolvedQuestion.related_proposal_refs",
            minimum=0,
        )
        _require_canonical_evidence_refs(
            self.evidence_refs,
            field_name="UnresolvedQuestion.evidence_refs",
        )


def _canonicalize_positions(
    positions: tuple[DisagreementPosition, ...],
) -> tuple[DisagreementPosition, ...]:
    if not positions:
        raise ValueError("positions must be non-empty")
    normalized: list[DisagreementPosition] = []
    seen: set[tuple[int, str]] = set()
    for position in positions:
        if type(position) is not DisagreementPosition:
            raise TypeError("positions must contain DisagreementPosition")
        key = _lineage_ref_sort_key(position.proposal_ref)
        if key in seen:
            raise ValueError("positions must not contain duplicate proposal_ref")
        seen.add(key)
        normalized.append(position)
    return tuple(sorted(normalized, key=lambda item: _lineage_ref_sort_key(item.proposal_ref)))


def _canonicalize_conflicts(
    conflicts: tuple[DisagreementConflict, ...],
) -> tuple[DisagreementConflict, ...]:
    if not conflicts:
        raise ValueError("conflicts must be non-empty")
    normalized: list[DisagreementConflict] = []
    for conflict in conflicts:
        if type(conflict) is not DisagreementConflict:
            raise TypeError("conflicts must contain DisagreementConflict")
        normalized.append(conflict)
    return tuple(
        sorted(
            normalized,
            key=lambda item: (
                item.dimension,
                item.summary,
                tuple(_lineage_ref_sort_key(ref) for ref in item.proposal_refs),
            ),
        ),
    )


def _canonicalize_unresolved_questions(
    unresolved_questions: tuple[UnresolvedQuestion, ...],
) -> tuple[UnresolvedQuestion, ...]:
    normalized: list[UnresolvedQuestion] = []
    seen_questions: set[str] = set()
    for question in unresolved_questions:
        if type(question) is not UnresolvedQuestion:
            raise TypeError("unresolved_questions must contain UnresolvedQuestion")
        if question.question in seen_questions:
            raise ValueError("unresolved_questions must not contain duplicate question")
        seen_questions.add(question.question)
        normalized.append(question)
    return tuple(sorted(normalized, key=lambda item: item.question))


def _require_canonical_positions(
    positions: tuple[DisagreementPosition, ...],
) -> None:
    canonical = _canonicalize_positions(positions)
    if positions != canonical:
        raise ValueError("positions must be in canonical order without duplicates")


def _require_canonical_conflicts(
    conflicts: tuple[DisagreementConflict, ...],
) -> None:
    canonical = _canonicalize_conflicts(conflicts)
    if conflicts != canonical:
        raise ValueError("conflicts must be in canonical order")


def _require_canonical_unresolved_questions(
    unresolved_questions: tuple[UnresolvedQuestion, ...],
) -> None:
    canonical = _canonicalize_unresolved_questions(unresolved_questions)
    if unresolved_questions != canonical:
        raise ValueError("unresolved_questions must be in canonical order without duplicates")


@dataclass(frozen=True, slots=True)
class DecisionDisagreementArtifact:
    """Structured disagreement across at least two distinct proposal branches.

    Supporting deliberation artifact — not a CandidateDecision and not
    authoritative resolution output.
    """

    proposal_refs: tuple[DecisionLineageRef, ...]
    positions: tuple[DisagreementPosition, ...]
    conflicts: tuple[DisagreementConflict, ...]
    unresolved_questions: tuple[UnresolvedQuestion, ...] = ()

    def __post_init__(self) -> None:
        _require_canonical_lineage_refs(
            self.proposal_refs,
            field_name="DecisionDisagreementArtifact.proposal_refs",
            minimum=2,
        )
        known = _proposal_ref_keys(self.proposal_refs)

        _require_canonical_positions(self.positions)
        for position in self.positions:
            if _lineage_ref_sort_key(position.proposal_ref) not in known:
                raise ValueError(
                    "DisagreementPosition.proposal_ref must reference known proposal refs",
                )

        _require_canonical_conflicts(self.conflicts)
        for conflict in self.conflicts:
            _require_known_proposal_refs(
                conflict.proposal_refs,
                field_name="DisagreementConflict.proposal_refs",
                known=known,
            )

        _require_canonical_unresolved_questions(self.unresolved_questions)
        for question in self.unresolved_questions:
            _require_known_proposal_refs(
                question.related_proposal_refs,
                field_name="UnresolvedQuestion.related_proposal_refs",
                known=known,
            )


def disagreement_position(
    *,
    proposal_ref: DecisionLineageRef,
    summary: str,
    evidence_refs: tuple[EvidenceReferenceId, ...] = (),
) -> DisagreementPosition:
    """Build one position with canonical evidence ordering."""
    return DisagreementPosition(
        proposal_ref=proposal_ref,
        summary=summary,
        evidence_refs=_canonicalize_evidence_refs(
            evidence_refs,
            field_name="DisagreementPosition.evidence_refs",
        ),
    )


def disagreement_conflict(
    *,
    dimension: str,
    proposal_refs: tuple[DecisionLineageRef, ...],
    summary: str,
) -> DisagreementConflict:
    """Build one conflict with canonical proposal ordering."""
    return DisagreementConflict(
        dimension=dimension,
        proposal_refs=_canonicalize_lineage_refs(
            proposal_refs,
            field_name="DisagreementConflict.proposal_refs",
            minimum=2,
        ),
        summary=summary,
    )


def unresolved_question(
    *,
    question: str,
    related_proposal_refs: tuple[DecisionLineageRef, ...] = (),
    evidence_refs: tuple[EvidenceReferenceId, ...] = (),
) -> UnresolvedQuestion:
    """Build one unresolved question with canonical ref ordering."""
    return UnresolvedQuestion(
        question=question,
        related_proposal_refs=_canonicalize_lineage_refs(
            related_proposal_refs,
            field_name="UnresolvedQuestion.related_proposal_refs",
            minimum=0,
        ),
        evidence_refs=_canonicalize_evidence_refs(
            evidence_refs,
            field_name="UnresolvedQuestion.evidence_refs",
        ),
    )


def decision_disagreement_artifact(
    *,
    proposal_refs: tuple[DecisionLineageRef, ...],
    positions: tuple[DisagreementPosition, ...],
    conflicts: tuple[DisagreementConflict, ...],
    unresolved_questions: tuple[UnresolvedQuestion, ...] = (),
) -> DecisionDisagreementArtifact:
    """Build one disagreement artifact with canonical ordering throughout."""
    return DecisionDisagreementArtifact(
        proposal_refs=_canonicalize_lineage_refs(
            proposal_refs,
            field_name="DecisionDisagreementArtifact.proposal_refs",
            minimum=2,
        ),
        positions=_canonicalize_positions(positions),
        conflicts=_canonicalize_conflicts(conflicts),
        unresolved_questions=_canonicalize_unresolved_questions(
            unresolved_questions,
        ),
    )
