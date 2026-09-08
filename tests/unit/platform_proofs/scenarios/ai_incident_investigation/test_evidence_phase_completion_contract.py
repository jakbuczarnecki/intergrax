# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationStopReason
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_phase_context import (
    CompletionIntentPhaseValidationError,
    derive_evidence_phase_context,
    validate_completion_intent_for_phase,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    ClaimProposal,
    CompletionIntent,
    HypothesisDisposition,
    HypothesisProposal,
    IncidentReasoningProposal,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    DIAGNOSIS_KIND,
)

pytestmark = pytest.mark.unit


def _proposal(*, completion_intent: CompletionIntent) -> IncidentReasoningProposal:
    unresolved_fields: dict[str, object] = {}
    if completion_intent is CompletionIntent.UNRESOLVED:
        unresolved_fields = {
            "unresolved_reason": "telemetry unavailable",
            "information_gaps": ("decisive telemetry",),
        }
    return IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H1",
                disposition=HypothesisDisposition.PLAUSIBLE,
                summary="Workload-throughput correlation observed.",
            ),
        ),
        preferred_hypothesis_id="H1",
        uncertainty_class="high",
        claim_proposals=(
            ClaimProposal(
                hypothesis_id="H1",
                statement="Overload hypothesis H1 pending distinguishing evidence.",
                claim_kind=str(DIAGNOSIS_KIND),
            ),
        ),
        completion_intent=completion_intent,
        action_objective="assess gathered evidence",
        **unresolved_fields,
    )


def test_planner_final_answer_rejects_need_more_evidence() -> None:
    phase_context = derive_evidence_phase_context("planner_final_answer")
    with pytest.raises(
        CompletionIntentPhaseValidationError,
        match="need_more_evidence_invalid_after_evidence_gathering_closed",
    ):
        validate_completion_intent_for_phase(
            CompletionIntent.NEED_MORE_EVIDENCE,
            phase_context,
        )


def test_planner_final_answer_allows_supported_diagnosis() -> None:
    phase_context = derive_evidence_phase_context("planner_final_answer")
    validate_completion_intent_for_phase(
        _proposal(completion_intent=CompletionIntent.SUPPORTED_DIAGNOSIS).completion_intent,
        phase_context,
    )


def test_planner_final_answer_allows_unresolved() -> None:
    phase_context = derive_evidence_phase_context("planner_final_answer")
    validate_completion_intent_for_phase(
        _proposal(completion_intent=CompletionIntent.UNRESOLVED).completion_intent,
        phase_context,
    )


def test_active_gathering_allows_need_more_evidence() -> None:
    phase_context = derive_evidence_phase_context("empty_tool_calls")
    validate_completion_intent_for_phase(
        CompletionIntent.NEED_MORE_EVIDENCE,
        phase_context,
    )


@pytest.mark.parametrize(
    ("stop_reason", "additional_gathering_allowed"),
    [
        ("planner_final_answer", False),
        ("max_iterations", False),
        ("legacy_single_pass", False),
        ("empty_tool_calls", True),
    ],
)
def test_derive_evidence_phase_context_is_deterministic(
    stop_reason: ToolInvocationStopReason,
    additional_gathering_allowed: bool,
) -> None:
    first = derive_evidence_phase_context(stop_reason)
    second = derive_evidence_phase_context(stop_reason)
    assert first == second
    assert first.stop_reason == stop_reason
    assert first.additional_evidence_gathering_allowed is additional_gathering_allowed
