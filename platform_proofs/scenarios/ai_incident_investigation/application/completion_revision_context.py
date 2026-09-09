# © Artur Czarnecki. All rights reserved.

"""Typed authoritative completion-alignment revision context (S2) for model revision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    CompletionAlignmentMismatchReason,
    CompletionAlignmentState,
    CompletionAlignmentStatus,
    assess_completion_alignment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    HypothesisId,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    DIAGNOSIS_CLAIM_KIND,
)


class ClaimHypothesisBindingLike(Protocol):
    claim_id: str
    hypothesis_id: Literal["H1", "H2", "H3"]


@dataclass(frozen=True, slots=True)
class CompletionAlignmentRevisionContext:
    """Immutable S2 projection of authoritative S1 facts for completion-alignment revision."""

    mismatch_reason: CompletionAlignmentMismatchReason
    supported_hypothesis_id: HypothesisId | None
    supported_resolution: ClaimResolution | None

    def __post_init__(self) -> None:
        if (
            self.mismatch_reason
            is CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS
        ):
            if self.supported_hypothesis_id is None:
                raise ValueError(
                    "unresolved_with_supported_diagnosis requires supported_hypothesis_id"
                )
            if self.supported_resolution is not ClaimResolution.SUPPORTED:
                raise ValueError(
                    "unresolved_with_supported_diagnosis requires supported_resolution=SUPPORTED"
                )
            return
        if (
            self.mismatch_reason
            is CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
        ):
            if self.supported_hypothesis_id is not None:
                raise ValueError(
                    "supported_diagnosis_without_supported_state forbids supported_hypothesis_id"
                )
            if self.supported_resolution is not None:
                raise ValueError(
                    "supported_diagnosis_without_supported_state forbids supported_resolution"
                )
            return
        raise ValueError(f"unsupported mismatch_reason: {self.mismatch_reason}")


def _supported_diagnosis_claims(
    resolved_claim_set: EvidenceClaimSet,
) -> tuple[EvidenceBackedClaim, ...]:
    return tuple(
        claim
        for claim in resolved_claim_set.claims
        if str(claim.claim_kind) == DIAGNOSIS_CLAIM_KIND
        and claim.resolution is ClaimResolution.SUPPORTED
    )


def _hypothesis_for_claim(
    claim: EvidenceBackedClaim,
    bindings: Sequence[ClaimHypothesisBindingLike],
) -> HypothesisId | None:
    for binding in bindings:
        if str(binding.claim_id) == str(claim.claim_id):
            return HypothesisId(binding.hypothesis_id)
    return None


def resolve_authoritative_supported_hypothesis(
    resolved_claim_set: EvidenceClaimSet,
    bindings: Sequence[ClaimHypothesisBindingLike],
) -> HypothesisId | None:
    """Return the authoritative supported diagnosis hypothesis, if exactly one exists."""
    supported_claims = _supported_diagnosis_claims(resolved_claim_set)
    if not supported_claims:
        return None
    latest = supported_claims[-1]
    hypotheses = {
        hypothesis
        for claim in supported_claims
        for hypothesis in (_hypothesis_for_claim(claim, bindings),)
        if hypothesis is not None
    }
    if len(hypotheses) > 1:
        raise ValueError("multiple authoritative supported diagnosis hypotheses")
    latest_hypothesis = _hypothesis_for_claim(latest, bindings)
    if latest_hypothesis is None:
        raise ValueError("supported diagnosis claim missing hypothesis binding")
    return latest_hypothesis


def build_completion_alignment_revision_context(
    *,
    resolved_claim_set: EvidenceClaimSet,
    bindings: Sequence[ClaimHypothesisBindingLike],
    prior_completion_mode: str,
) -> CompletionAlignmentRevisionContext | None:
    """Project critic-resolved S1 state into typed revision context when misaligned."""
    supported_claims = _supported_diagnosis_claims(resolved_claim_set)
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=prior_completion_mode,
            has_supported_diagnosis=bool(supported_claims),
        )
    )
    if assessment.status is not CompletionAlignmentStatus.MISALIGNED:
        return None
    if assessment.mismatch_reason is None:
        return None

    if (
        assessment.mismatch_reason
        is CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS
    ):
        supported_hypothesis_id = resolve_authoritative_supported_hypothesis(
            resolved_claim_set,
            bindings,
        )
        if supported_hypothesis_id is None:
            raise ValueError(
                "unresolved_with_supported_diagnosis requires authoritative supported hypothesis"
            )
        return CompletionAlignmentRevisionContext(
            mismatch_reason=assessment.mismatch_reason,
            supported_hypothesis_id=supported_hypothesis_id,
            supported_resolution=ClaimResolution.SUPPORTED,
        )

    if (
        assessment.mismatch_reason
        is CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
    ):
        return CompletionAlignmentRevisionContext(
            mismatch_reason=assessment.mismatch_reason,
            supported_hypothesis_id=None,
            supported_resolution=None,
        )

    return None


def render_completion_alignment_revision_context(
    context: CompletionAlignmentRevisionContext,
) -> tuple[str, ...]:
    """Render typed authoritative revision facts for model-facing revision messages."""
    lines = [
        "Authoritative validation state (critic-resolved; not model-proposed prior reasoning):",
    ]
    if context.supported_hypothesis_id is not None:
        lines.append(f"- supported hypothesis: {context.supported_hypothesis_id.value}")
    if context.supported_resolution is not None:
        lines.append(f"- resolution: {context.supported_resolution.value}")
    if (
        context.mismatch_reason
        is CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS
    ):
        lines.append(
            "- alignment mismatch: unresolved completion intent conflicts with "
            "authoritative supported diagnosis"
        )
    elif (
        context.mismatch_reason
        is CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
    ):
        lines.append(
            "- alignment mismatch: supported diagnosis completion intent lacks "
            "authoritative supported diagnosis state"
        )
    lines.append(
        "Re-evaluate the structured completion intent against this authoritative state."
    )
    return tuple(lines)
