# © Artur Czarnecki. All rights reserved.

"""Canonical evaluator-loop revision proof for completion alignment (DS-E2E-15F.1)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    COMPLETION_ALIGNMENT_MISMATCH_SIGNAL,
    COMPLETION_ALIGNMENT_REVISION_GUIDANCE,
    CompletionAlignmentState,
    CompletionAlignmentStatus,
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
    assess_completion_alignment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
    PreReconciliationValidationError,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
    IncidentReasoningProposal,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    EVALUATOR_LOOP_MAX_ITERATIONS,
    OUTCOME_RESOLVED,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
    COMPLETION_SUPPORTED_DIAGNOSIS,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.lab_planner_llm import (
    FixtureDrivenIncidentInvestigationLLM,
    _evidence_ids_from_messages,
    build_fixture_reasoning_proposal,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)

pytestmark = pytest.mark.unit

_FULL_EVIDENCE_SEQUENCE: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_TELEMETRY_READ,
)
_ALL_EVIDENCE_IDS = {
    str(WORKLOAD_EVIDENCE_ID),
    str(THROUGHPUT_EVIDENCE_ID),
    str(STAFFING_PRELIMINARY_EVIDENCE_ID),
    str(STAFFING_ATTENDANCE_EVIDENCE_ID),
    str(COMPARISON_EVIDENCE_ID),
    str(TELEMETRY_EVIDENCE_ID),
}


def _has_full_evidence(messages: Sequence[ChatMessage]) -> bool:
    return _ALL_EVIDENCE_IDS.issubset(_evidence_ids_from_messages(messages))


class _AlignmentRevisionLLM(FixtureDrivenIncidentInvestigationLLM):
    """Gather full evidence, then emit 15E residual before canonical revision repair."""

    def __init__(self, *, persist_misalignment: bool = False) -> None:
        super().__init__(
            initial_sequence=_FULL_EVIDENCE_SEQUENCE,
            revision_sequence=(),
        )
        self._persist_misalignment = persist_misalignment
        self.revision_messages: tuple[ChatMessage, ...] = ()

    def generate_structured(self, messages, output_model, **kwargs):  # type: ignore[no-untyped-def]
        if output_model is not IncidentReasoningProposal:
            return super().generate_structured(messages, output_model, **kwargs)
        is_revision = self._detect_reasoning_phase(messages) == "revision"
        if is_revision:
            self.revision_messages = tuple(messages)
        evidence_ids = _evidence_ids_from_messages(messages)
        proposal = build_fixture_reasoning_proposal(
            evidence_ids=evidence_ids,
            is_revision=is_revision,
        )
        if _has_full_evidence(messages) and (not is_revision or self._persist_misalignment):
            proposal = proposal.model_copy(
                update={
                    "completion_intent": CompletionIntent.UNRESOLVED,
                    "uncertainty_class": "decisive_gap",
                    "information_gaps": (
                        "completion intent not reconciled with supported claim state",
                    ),
                    "unresolved_reason": (
                        "Completion intent left unresolved despite supported diagnosis state."
                    ),
                }
            )
        return LLMStructuredResult(
            parsed=proposal,
            response=build_adapter_response(content=""),
        )


@pytest.mark.asyncio
async def test_alignment_misalignment_routes_through_evaluator_loop_to_revision() -> None:
    llm = _AlignmentRevisionLLM()
    fixture_bundle = build_fixture_runtime_bundle(llm_adapter_override=llm)
    baseline_bundle = build_fixture_runtime_bundle(
        llm_adapter_override=FixtureDrivenIncidentInvestigationLLM(
            initial_sequence=_FULL_EVIDENCE_SEQUENCE,
            revision_sequence=(),
        ),
    )

    baseline = await execute_resolved_skeleton(baseline_bundle.bundle)
    result = await execute_resolved_skeleton(fixture_bundle.bundle)

    assert baseline.revision_pass is False
    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_verdict_passed
    assert result.revision_pass is True
    assert result.evaluator_loop_iterations >= 1
    assert result.evaluator_loop_iterations <= EVALUATOR_LOOP_MAX_ITERATIONS

    assert llm.revision_messages
    system_content = next(
        message.content or ""
        for message in llm.revision_messages
        if message.role == "system"
    )
    assert "Investigation phase: revision" in system_content
    assert UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR in system_content
    assert COMPLETION_ALIGNMENT_MISMATCH_SIGNAL in system_content
    assert COMPLETION_ALIGNMENT_REVISION_GUIDANCE in system_content
    assert "Authoritative validation state" in system_content
    assert "supported hypothesis: H3" in system_content
    assert "resolution: supported" in system_content
    prior_index = system_content.index("Prior model reasoning")
    authoritative_index = system_content.index("Authoritative validation state")
    critic_index = system_content.index("Critic feedback requiring incremental correction")
    assert prior_index < authoritative_index < critic_index
    assert "set completion_mode" not in system_content.lower()
    assert COMPLETION_SUPPORTED_DIAGNOSIS not in COMPLETION_ALIGNMENT_REVISION_GUIDANCE


@pytest.mark.asyncio
async def test_alignment_revision_failure_exhausts_evaluator_budget() -> None:
    fixture_bundle = build_fixture_runtime_bundle(
        llm_adapter_override=_AlignmentRevisionLLM(persist_misalignment=True),
    )
    with pytest.raises(PreReconciliationValidationError) as exc_info:
        await execute_resolved_skeleton(
            fixture_bundle.bundle,
            evaluator_loop_max_iterations=EVALUATOR_LOOP_MAX_ITERATIONS,
        )
    assert UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR in exc_info.value.diagnostic.validation_errors


@pytest.mark.asyncio
async def test_alignment_zero_evaluator_budget_fails_closed() -> None:
    fixture_bundle = build_fixture_runtime_bundle(
        llm_adapter_override=_AlignmentRevisionLLM(persist_misalignment=True),
    )
    with pytest.raises(PreReconciliationValidationError):
        await execute_resolved_skeleton(
            fixture_bundle.bundle,
            evaluator_loop_max_iterations=1,
        )


def test_unknown_completion_mode_fails_closed_as_misaligned() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode="unsupported_mode",
            has_supported_diagnosis=False,
        )
    )
    assert assessment.status is CompletionAlignmentStatus.MISALIGNED


def test_alignment_guidance_signal_is_non_prescriptive() -> None:
    assert "set completion_mode" not in COMPLETION_ALIGNMENT_REVISION_GUIDANCE.lower()
    assert COMPLETION_SUPPORTED_DIAGNOSIS not in COMPLETION_ALIGNMENT_REVISION_GUIDANCE
