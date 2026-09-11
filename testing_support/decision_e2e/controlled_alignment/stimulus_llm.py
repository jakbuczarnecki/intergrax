# © Artur Czarnecki. All rights reserved.

"""Fixture LLM adapters that inject controlled completion-alignment mismatch."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
    IncidentReasoningProposal,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
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

FULL_EVIDENCE_TOOL_SEQUENCE: tuple[str, ...] = (
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


class ModelOvercommitStimulusLLM(FixtureDrivenIncidentInvestigationLLM):
    """
    Gather full observational evidence, then overcommit on attempt 0:

    completion intent SUPPORTED_DIAGNOSIS with only partial claim backing.
    Canonical fixture repair on evaluator revision (attempt 1).
    """

    def __init__(self, *, persist_misalignment: bool = False) -> None:
        super().__init__(
            initial_sequence=FULL_EVIDENCE_TOOL_SEQUENCE,
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
        if _has_full_evidence(messages) and (not is_revision or self._persist_misalignment):
            partial = build_fixture_reasoning_proposal(
                evidence_ids={
                    str(WORKLOAD_EVIDENCE_ID),
                    str(THROUGHPUT_EVIDENCE_ID),
                    str(STAFFING_PRELIMINARY_EVIDENCE_ID),
                },
                is_revision=False,
            )
            proposal = partial.model_copy(
                update={"completion_intent": CompletionIntent.SUPPORTED_DIAGNOSIS}
            )
        else:
            proposal = build_fixture_reasoning_proposal(
                evidence_ids=evidence_ids,
                is_revision=is_revision,
            )
        return LLMStructuredResult(
            parsed=proposal,
            response=build_adapter_response(content=""),
        )


class ForwardMismatchStimulusLLM(FixtureDrivenIncidentInvestigationLLM):
    """UNRESOLVED completion intent despite supported diagnosis state (MODEL_UNDERCOMMIT)."""

    def __init__(self, *, persist_misalignment: bool = False) -> None:
        super().__init__(
            initial_sequence=FULL_EVIDENCE_TOOL_SEQUENCE,
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


__all__ = [
    "ForwardMismatchStimulusLLM",
    "FULL_EVIDENCE_TOOL_SEQUENCE",
    "ModelOvercommitStimulusLLM",
]
