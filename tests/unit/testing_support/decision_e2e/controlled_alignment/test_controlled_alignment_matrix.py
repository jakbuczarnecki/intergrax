# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-L1.R4.R4 controlled alignment matrix (qualification stimulus)."""

from __future__ import annotations

import pytest

from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    CompletionAlignmentMismatchReason,
    CompletionAlignmentState,
    CompletionAlignmentStatus,
    assess_completion_alignment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    CompletionAlignmentCorrectability,
    CompletionAlignmentDirection,
    decide_completion_alignment_correction,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
    PreReconciliationValidationError,
)
from pathlib import Path

from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.lab_planner_llm import (
    FixtureDrivenIncidentInvestigationLLM,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from testing_support.decision_e2e.controlled_alignment.evidence import (
    RepairQualificationStatus,
    revision_context_from_system_messages,
)
from testing_support.decision_e2e.controlled_alignment.runner import (
    run_model_overcommit_controlled_qualification,
)
from testing_support.decision_e2e.controlled_alignment.scenario import (
    ControlledAlignmentScenario,
    MODEL_OVERCOMMIT_SCENARIO,
)
from testing_support.decision_e2e.controlled_alignment.source_freeze import (
    verify_controlled_alignment_source_freeze,
)
from testing_support.decision_e2e.controlled_alignment.stimulus_llm import (
    ForwardMismatchStimulusLLM,
    FULL_EVIDENCE_TOOL_SEQUENCE,
    ModelOvercommitStimulusLLM,
)
from testing_support.decision_e2e.controlled_alignment.stimulus_state import (
    ControlledAlignmentStimulusState,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_controlled_alignment_scenario_serializable() -> None:
    payload = MODEL_OVERCOMMIT_SCENARIO.to_dict()
    restored = ControlledAlignmentScenario.from_dict(payload)
    assert restored == MODEL_OVERCOMMIT_SCENARIO


def test_model_overcommit_stimulus_state_projection() -> None:
    state = ControlledAlignmentStimulusState.model_overcommit()
    assert state.completion_mode == COMPLETION_SUPPORTED_DIAGNOSIS
    assert state.supported_state_present is False
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=state.completion_mode,
            has_supported_diagnosis=state.supported_state_present,
        )
    )
    assert assessment.status is CompletionAlignmentStatus.MISALIGNED
    assert (
        assessment.mismatch_reason
        is CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
    )


def test_matrix_match_no_revision_policy() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
            has_supported_diagnosis=True,
        )
    )
    decision = decide_completion_alignment_correction(
        assessment,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=2,
    )
    assert decision.alignment_mismatch_detected is False
    assert decision.alignment_correctable is False


def test_matrix_forward_mismatch_revision_policy() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
        )
    )
    decision = decide_completion_alignment_correction(
        assessment,
        completion_mode=COMPLETION_UNRESOLVED,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    assert decision.direction is CompletionAlignmentDirection.MODEL_UNDERCOMMIT
    assert decision.alignment_correctable is True


def test_matrix_reverse_mismatch_revision_policy() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
            has_supported_diagnosis=False,
        )
    )
    decision = decide_completion_alignment_correction(
        assessment,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    assert decision.direction is CompletionAlignmentDirection.MODEL_OVERCOMMIT
    assert decision.alignment_correctable is True


def test_matrix_unknown_fail_closed() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode="unsupported_mode",
            has_supported_diagnosis=False,
        )
    )
    decision = decide_completion_alignment_correction(
        assessment,
        completion_mode="unsupported_mode",
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    assert decision.correctability is CompletionAlignmentCorrectability.TERMINAL


@pytest.mark.asyncio
async def test_matrix_match_fixture_run_no_revision() -> None:
    bundle = build_fixture_runtime_bundle(
        llm_adapter_override=FixtureDrivenIncidentInvestigationLLM(
            initial_sequence=FULL_EVIDENCE_TOOL_SEQUENCE,
            revision_sequence=(),
        ),
    )
    result = await execute_resolved_skeleton(bundle.bundle)
    assert result.outcome == OUTCOME_RESOLVED
    assert result.revision_pass is False


@pytest.mark.asyncio
async def test_matrix_forward_mismatch_revision() -> None:
    llm = ForwardMismatchStimulusLLM()
    bundle = build_fixture_runtime_bundle(llm_adapter_override=llm)
    result = await execute_resolved_skeleton(bundle.bundle)
    assert result.revision_pass is True
    assert revision_context_from_system_messages(
        tuple(message.content or "" for message in llm.revision_messages if message.role == "system")
    )


@pytest.mark.asyncio
async def test_matrix_zero_budget_exhausted() -> None:
    bundle = build_fixture_runtime_bundle(
        llm_adapter_override=ModelOvercommitStimulusLLM(persist_misalignment=True),
    )
    with pytest.raises(PreReconciliationValidationError):
        await execute_resolved_skeleton(
            bundle.bundle,
            evaluator_loop_max_iterations=1,
        )


@pytest.mark.asyncio
async def test_controlled_model_overcommit_full_repair_chain() -> None:
    result = await run_model_overcommit_controlled_qualification()
    assert result.execution.outcome == OUTCOME_RESOLVED
    assert result.execution.revision_pass is True
    assert result.repair_evidence.repair_status is RepairQualificationStatus.PASS
    pre = result.repair_evidence.alignment.pre_repair
    assert pre is not None
    assert pre.alignment_status is AlignmentStatus.MISMATCH
    assert pre.alignment_direction is AlignmentDirection.REVERSE
    assert pre.correctable is True
    assert 0 in result.repair_evidence.attempt_indices
    assert result.repair_evidence.revision_context_valid is True


def test_source_freeze_gate() -> None:
    root = Path(__file__).resolve().parents[5]
    report = verify_controlled_alignment_source_freeze(root)
    assert report.status is SourceFreezeStatus.PASS
