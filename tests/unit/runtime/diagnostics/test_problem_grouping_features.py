# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticCertainty,
    DiagnosticFinding,
    DiagnosticFindingKind,
    DiagnosticLimitation,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    build_deterministic_problem_signature,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
    LifecycleViolationTransition,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingInput,
    normalize_assessment,
)
from intergrax.runtime.diagnostics.problem_grouping_features import (
    CAUSAL_SOURCE_REF_KIND_MESSAGE_BUS_TASK,
    CAUSAL_TARGET_REF_KIND_RUNTIME_EXECUTION,
    MAX_TEXT_EVIDENCE_CHARS,
    ProblemGroupingCausalFeature,
    ProblemGroupingComponentFeature,
    ProblemGroupingExecutionFeature,
    ProblemGroupingFailureFeature,
    ProblemGroupingFeatureIntegrityError,
    ProblemGroupingFeatureSet,
    ProblemGroupingIntegrationFeature,
    ProblemGroupingOperationFeature,
    ProblemGroupingTextEvidence,
    ProblemGroupingTextEvidenceSourceKind,
    REPRESENTATION_VERSION_V2,
    project_assessment_features,
)
from intergrax.runtime.events.event_taxonomy import EventCategory
from intergrax.runtime.events.runtime_event import RuntimeEventType

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"


def _finding(*, claim: str = "A lifecycle event was recorded after canonical run closure.") -> DiagnosticFinding:
    event_id = mint_event_id()
    return DiagnosticFinding(
        kind=DiagnosticFindingKind.EVENT_AFTER_TERMINAL,
        scope=LifecycleAnomalyScope.EXECUTION,
        attempt_id=None,
        certainty=DiagnosticCertainty.PROVEN,
        claim=claim,
        source_anomaly_kind=LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
        supporting_event_ids=(event_id,),
        supporting_evidence_ids=(),
        supporting_positions=(),
    )


def _limitation() -> DiagnosticLimitation:
    event_id = mint_event_id()
    return DiagnosticLimitation(
        kind=DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED,
        factual_message="Runtime history is truncated; conclusions requiring the unseen tail cannot be proven.",
        source_anomaly_kind=LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED,
        supporting_event_ids=(event_id,),
        supporting_evidence_ids=(),
        supporting_positions=(),
    )


def _assessment(
    *,
    findings: tuple[DiagnosticFinding, ...] = (),
    limitations: tuple[DiagnosticLimitation, ...] = (),
) -> DiagnosticAssessment:
    return DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        findings=findings,
        limitations=limitations,
    )


def test_project_assessment_features_links_structural_signature_and_text() -> None:
    assessment = _assessment(findings=(_finding(),), limitations=(_limitation(),))
    features = project_assessment_features(assessment)

    assert features.subject_ref.tenant_id == assessment.tenant_id
    assert features.subject_ref.task_id == assessment.task_id
    assert features.subject_ref.run_id == assessment.run_id
    assert features.representation_version == "2"
    assert len(features.text_evidence) == 2
    assert features.text_evidence[0].source_kind is ProblemGroupingTextEvidenceSourceKind.OPERATOR_CLAIM
    assert features.text_evidence[1].source_kind is ProblemGroupingTextEvidenceSourceKind.FACTUAL_LIMITATION
    assert features.structural_signature.findings
    assert features.structural_signature.limitations


def test_grouping_input_preserves_subject_identity() -> None:
    assessment = _assessment(findings=(_finding(),))
    subject = normalize_assessment(assessment)
    features = project_assessment_features(assessment, subject=subject)
    grouping_input = ProblemGroupingInput(subject=subject, features=features)

    assert grouping_input.subject.tenant_id == assessment.tenant_id
    assert grouping_input.subject.task_id == assessment.task_id
    assert grouping_input.subject.run_id == assessment.run_id
    assert grouping_input.features is not None
    assert grouping_input.features.subject_ref == subject.ref


def test_project_rejects_oversized_text() -> None:
    oversized = "x" * (MAX_TEXT_EVIDENCE_CHARS + 1)
    assessment = _assessment(findings=(_finding(claim=oversized),))
    with pytest.raises(ProblemGroupingFeatureIntegrityError):
        project_assessment_features(assessment)


def test_default_projector_yields_empty_extended_feature_tuples() -> None:
    assessment = _assessment(findings=(_finding(),))
    features = project_assessment_features(assessment)

    assert features.execution_context == ()
    assert features.component_context == ()
    assert features.operation_context == ()
    assert features.integration_context == ()
    assert features.failure_context == ()
    assert features.causal_context == ()


def test_feature_set_carries_all_typed_categories_simultaneously() -> None:
    assessment = _assessment(findings=(_finding(),))
    subject = normalize_assessment(assessment)
    event_id = mint_event_id()
    evidence_id = mint_event_id()

    features = ProblemGroupingFeatureSet(
        subject_ref=subject.ref,
        representation_version=REPRESENTATION_VERSION_V2,
        structural_signature=build_deterministic_problem_signature(subject),
        execution_context=(
            ProblemGroupingExecutionFeature(
                phase=ExecutionPhase.STEP_EXECUTION,
                event_category=EventCategory.TOOL,
                event_type=RuntimeEventType.TOOL_FAILED.value,
                is_retry_related=False,
                supporting_event_ids=(event_id,),
            ),
        ),
        component_context=(
            ProblemGroupingComponentFeature(
                source_layer="tool",
                source_component="mcp_connector",
                supporting_event_ids=(event_id,),
            ),
        ),
        operation_context=(
            ProblemGroupingOperationFeature(
                agent_id="agent-a",
                tool_id="tool-b",
                capability="search",
                supporting_event_ids=(event_id,),
            ),
        ),
        integration_context=(
            ProblemGroupingIntegrationFeature(
                provider="celery",
                integration_id="queue-main",
                namespace="applications.hosting",
                supporting_event_ids=(event_id,),
            ),
        ),
        failure_context=(
            ProblemGroupingFailureFeature(
                problem_kind="platform.tool_failure",
                severity="error",
                error_code="TOOL_TIMEOUT",
                exception_type="TimeoutError",
                supporting_event_ids=(event_id,),
            ),
        ),
        causal_context=(
            ProblemGroupingCausalFeature(
                relation_kind="transport_task.triggered_execution",
                source_ref_kind=CAUSAL_SOURCE_REF_KIND_MESSAGE_BUS_TASK,
                target_ref_kind=CAUSAL_TARGET_REF_KIND_RUNTIME_EXECUTION,
                source_provider="celery",
                supporting_evidence_ids=(evidence_id,),
            ),
        ),
        text_evidence=(
            ProblemGroupingTextEvidence(
                source_kind=ProblemGroupingTextEvidenceSourceKind.PROBLEM_SIGNAL_SAFE_MESSAGE,
                text="Tool invocation timed out after policy limit.",
                supporting_event_ids=(event_id,),
                supporting_evidence_ids=(),
            ),
        ),
    )
    grouping_input = ProblemGroupingInput(subject=subject, features=features)

    assert grouping_input.features is features
    assert grouping_input.features.structural_signature.findings
    assert grouping_input.features.execution_context[0].phase is ExecutionPhase.STEP_EXECUTION
    assert grouping_input.features.component_context[0].source_component == "mcp_connector"
    assert grouping_input.features.operation_context[0].tool_id == "tool-b"
    assert grouping_input.features.integration_context[0].provider == "celery"
    assert grouping_input.features.failure_context[0].error_code == "TOOL_TIMEOUT"
    assert grouping_input.features.causal_context[0].relation_kind == (
        "transport_task.triggered_execution"
    )
    assert grouping_input.features.text_evidence[0].source_kind is (
        ProblemGroupingTextEvidenceSourceKind.PROBLEM_SIGNAL_SAFE_MESSAGE
    )


def test_multiple_component_and_failure_features_coexist() -> None:
    assessment = _assessment(findings=(_finding(),))
    subject = normalize_assessment(assessment)
    event_a = mint_event_id()
    event_b = mint_event_id()

    features = ProblemGroupingFeatureSet(
        subject_ref=subject.ref,
        representation_version=REPRESENTATION_VERSION_V2,
        structural_signature=build_deterministic_problem_signature(subject),
        component_context=(
            ProblemGroupingComponentFeature(
                source_layer="agent",
                source_component="planner",
                supporting_event_ids=(event_a,),
            ),
            ProblemGroupingComponentFeature(
                source_layer="tool",
                source_component="browser",
                supporting_event_ids=(event_b,),
            ),
        ),
        failure_context=(
            ProblemGroupingFailureFeature(
                problem_kind="platform.tool_failure",
                severity="error",
                supporting_event_ids=(event_a,),
            ),
            ProblemGroupingFailureFeature(
                problem_kind="platform.integration_failure",
                severity="warning",
                error_code="HTTP_503",
                supporting_event_ids=(event_b,),
            ),
        ),
    )

    assert len(features.component_context) == 2
    assert len(features.failure_context) == 2


def test_instance_ids_remain_provenance_not_semantic_identity() -> None:
    assessment = _assessment(findings=(_finding(),))
    subject = normalize_assessment(assessment)
    event_id = mint_event_id()
    evidence_id = mint_event_id()

    features = ProblemGroupingFeatureSet(
        subject_ref=subject.ref,
        representation_version=REPRESENTATION_VERSION_V2,
        structural_signature=build_deterministic_problem_signature(subject),
        execution_context=(
            ProblemGroupingExecutionFeature(supporting_event_ids=(event_id,)),
        ),
        causal_context=(
            ProblemGroupingCausalFeature(
                relation_kind="transport_task.triggered_execution",
                source_ref_kind=CAUSAL_SOURCE_REF_KIND_MESSAGE_BUS_TASK,
                target_ref_kind=CAUSAL_TARGET_REF_KIND_RUNTIME_EXECUTION,
                supporting_evidence_ids=(evidence_id,),
            ),
        ),
    )

    assert features.subject_ref.task_id == subject.task_id
    assert features.subject_ref.run_id == subject.run_id
    assert features.execution_context[0].supporting_event_ids == (event_id,)
    assert features.causal_context[0].supporting_evidence_ids == (evidence_id,)
    assert not hasattr(features.execution_context[0], "task_id")
    assert not hasattr(features.execution_context[0], "run_id")
    assert not hasattr(features.causal_context[0], "attempt_id")


def test_rejects_blank_component_identifiers() -> None:
    assessment = _assessment(findings=(_finding(),))
    subject = normalize_assessment(assessment)

    with pytest.raises(ProblemGroupingFeatureIntegrityError):
        ProblemGroupingFeatureSet(
            subject_ref=subject.ref,
            representation_version=REPRESENTATION_VERSION_V2,
            structural_signature=build_deterministic_problem_signature(subject),
            component_context=(
                ProblemGroupingComponentFeature(
                    source_layer="  ",
                    source_component="planner",
                ),
            ),
        )


def test_rejects_operation_feature_without_dimensions() -> None:
    assessment = _assessment(findings=(_finding(),))
    subject = normalize_assessment(assessment)

    with pytest.raises(ProblemGroupingFeatureIntegrityError):
        ProblemGroupingFeatureSet(
            subject_ref=subject.ref,
            representation_version=REPRESENTATION_VERSION_V2,
            structural_signature=build_deterministic_problem_signature(subject),
            operation_context=(ProblemGroupingOperationFeature(),),
        )
