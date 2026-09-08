# © Artur Czarnecki. All rights reserved.

"""Provider-neutral evidence phase context for reasoning completion intent validation."""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationStopReason
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_NEED_MORE_EVIDENCE,
    DIAGNOSIS_KIND,
)

if TYPE_CHECKING:
    from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
        CompletionIntent,
    )

_GATHERING_CLOSED_STOP_REASONS: frozenset[ToolInvocationStopReason] = frozenset(
    {
        "planner_final_answer",
        "max_iterations",
        "legacy_single_pass",
    }
)


@dataclass(frozen=True, slots=True)
class EvidencePhaseContext:
    """Typed evidence-gathering phase state consumed by reasoning."""

    stop_reason: ToolInvocationStopReason
    additional_evidence_gathering_allowed: bool


class CompletionIntentPhaseValidationError(Exception):
    """Raised when model completion intent violates the active evidence phase contract."""


def derive_evidence_phase_context(
    stop_reason: ToolInvocationStopReason,
) -> EvidencePhaseContext:
    """Derive phase context deterministically from evidence-gathering termination."""
    return EvidencePhaseContext(
        stop_reason=stop_reason,
        additional_evidence_gathering_allowed=(
            stop_reason not in _GATHERING_CLOSED_STOP_REASONS
        ),
    )


def render_completion_intent_contract_lines(
    phase_context: EvidencePhaseContext,
) -> tuple[str, ...]:
    """Render completion-intent prompt guidance from typed phase context."""
    if phase_context.additional_evidence_gathering_allowed:
        need_more_evidence_line = (
            "- need_more_evidence: only when additional allowed evidence-gathering work "
            "remains possible; still provide non-empty claim_proposals describing the "
            "current provisional assessment."
        )
        phase_status_line = (
            f"Evidence phase: additional evidence gathering remains allowed "
            f"(stop_reason={phase_context.stop_reason})."
        )
    else:
        need_more_evidence_line = (
            "- need_more_evidence: not available — evidence gathering is complete; "
            "choose supported_diagnosis or unresolved."
        )
        phase_status_line = (
            f"Evidence phase: evidence gathering is complete "
            f"(stop_reason={phase_context.stop_reason}); additional evidence gathering "
            "is not allowed."
        )
    return (
        "Completion intent contract:",
        phase_status_line,
        "- claim_proposals must always be non-empty; include diagnosis claim proposals for "
        "each hypothesis you assess.",
        "- supported_diagnosis: only when gathered evidence supports a final diagnosis "
        "strongly enough for the scenario contract.",
        "- unresolved: only after available investigation is exhausted; must set unresolved_reason "
        "to a non-empty string and information_gaps to a non-empty list.",
        need_more_evidence_line,
        (
            "Claim proposal contract: always emit at least one claim_proposal with "
            f"claim_kind={str(DIAGNOSIS_KIND)!s} for each hypothesis under active consideration. "
            "Do not emit evidence_id fields — the platform binds evidence relations deterministically."
        ),
    )


def validate_completion_intent_for_phase(
    completion_intent: CompletionIntent,
    phase_context: EvidencePhaseContext,
) -> None:
    """Fail closed when completion intent contradicts the evidence phase contract."""
    intent_value = (
        completion_intent.value
        if hasattr(completion_intent, "value")
        else str(completion_intent)
    )
    if (
        intent_value == COMPLETION_NEED_MORE_EVIDENCE
        and not phase_context.additional_evidence_gathering_allowed
    ):
        raise CompletionIntentPhaseValidationError(
            "need_more_evidence_invalid_after_evidence_gathering_closed"
        )
