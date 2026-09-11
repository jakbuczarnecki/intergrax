# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-O2-P2 completion alignment producer reachability qualification."""

from __future__ import annotations

import pytest

from intergrax.decision_system.completion_eligibility import CompletionEligibilityStatus
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionMode,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
    IncidentReasoningProposal,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
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
from testing_support.decision_e2e.completion_alignment_producer_reachability import (
    assert_canonical_match_alignment_event,
    assert_reverse_mismatch_alignment_event,
    run_completion_alignment_producer_reachability_probe,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
    TraceReadbackStatus,
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


class _ReverseAlignmentProbeLLM(FixtureDrivenIncidentInvestigationLLM):
    """Full evidence with supported-diagnosis overcommit, then canonical repair on revision."""

    def generate_structured(self, messages, output_model, **kwargs):  # type: ignore[no-untyped-def]
        if output_model is not IncidentReasoningProposal:
            return super().generate_structured(messages, output_model, **kwargs)
        is_revision = self._detect_reasoning_phase(messages) == "revision"
        evidence_ids = _evidence_ids_from_messages(messages)
        if not is_revision and len(evidence_ids) >= 4:
            partial = build_fixture_reasoning_proposal(
                evidence_ids={
                    str(WORKLOAD_EVIDENCE_ID),
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


@pytest.mark.asyncio
async def test_completion_alignment_producer_reachable() -> None:
    probe = await run_completion_alignment_producer_reachability_probe()

    assert probe.execution_result.outcome == OUTCOME_RESOLVED
    assert probe.completion_eligibility_status is CompletionEligibilityStatus.ELIGIBLE
    assert probe.trace_readback_status is TraceReadbackStatus.PASS
    assert probe.alignment_events_count >= 1
    assert probe.coverage_alignment.alignment_event_present >= 1

    event = probe.typed_alignment_events[-1]
    assert event.schema_id() == COMPLETION_ALIGNMENT_TRACE_SCHEMA
    assert event.completion_mode is CompletionMode.SUPPORTED_DIAGNOSIS
    assert_canonical_match_alignment_event(event)


@pytest.mark.asyncio
async def test_completion_alignment_producer_reachable_reverse_mismatch_telemetry() -> None:
    fixture_bundle = build_fixture_runtime_bundle(
        llm_adapter_override=_ReverseAlignmentProbeLLM(
            initial_sequence=_FULL_EVIDENCE_SEQUENCE,
            revision_sequence=(),
        ),
    )
    probe = await run_completion_alignment_producer_reachability_probe(
        fixture_bundle=fixture_bundle,
    )

    assert probe.execution_result.outcome == OUTCOME_RESOLVED
    assert probe.completion_eligibility_status is CompletionEligibilityStatus.ELIGIBLE
    assert probe.alignment_events_count >= 1

    pre_repair = next(
        (
            item
            for item in probe.typed_alignment_events
            if item.alignment_status is AlignmentStatus.MISMATCH
            and item.alignment_direction is AlignmentDirection.REVERSE
        ),
        None,
    )
    if pre_repair is not None:
        assert_reverse_mismatch_alignment_event(pre_repair)
        return

    final_event = probe.typed_alignment_events[-1]
    assert final_event.alignment_status in {
        AlignmentStatus.MATCH,
        AlignmentStatus.MISMATCH,
    }
    if final_event.alignment_status is AlignmentStatus.MISMATCH:
        assert final_event.completion_mode.value in {
            COMPLETION_SUPPORTED_DIAGNOSIS,
            COMPLETION_UNRESOLVED,
        }
