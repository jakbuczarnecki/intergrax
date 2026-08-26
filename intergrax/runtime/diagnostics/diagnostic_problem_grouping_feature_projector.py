# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Production feature projector: typed diagnostic/platform facts → ProblemGroupingFeatureSet (DIAG-5C-C).

Performs no grouping decisions, no persistence queries, and no payload parsing.
"""

from __future__ import annotations

from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessment
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstruction
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubject
from intergrax.runtime.diagnostics.problem_grouping_features import (
    CAUSAL_SOURCE_REF_KIND_MESSAGE_BUS_TASK,
    CAUSAL_TARGET_REF_KIND_RUNTIME_EXECUTION,
    ProblemGroupingCausalFeature,
    ProblemGroupingComponentFeature,
    ProblemGroupingExecutionFeature,
    ProblemGroupingFailureFeature,
    ProblemGroupingFeatureIntegrityError,
    ProblemGroupingFeatureSet,
    ProblemGroupingFeatureSourceFacts,
    ProblemGroupingIntegrationFeature,
    ProblemGroupingOperationFeature,
    ProblemGroupingRepresentationVersion,
    ProblemGroupingTextEvidence,
    ProblemGroupingTextEvidenceSourceKind,
    REPRESENTATION_VERSION_V2,
    _bounded_text,
    project_assessment_features,
)
from intergrax.runtime.events.execution_position import PositionedRuntimeEvent
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence
from intergrax.runtime.observability.problem_signal import PlatformProblemSignal

_RETRY_RELATED_EVENT_TYPES: frozenset[RuntimeEventType] = frozenset(
    {
        RuntimeEventType.RETRY_SCHEDULED,
        RuntimeEventType.RETRY_STARTED,
    }
)

_GROUPING_FAILURE_ANOMALY_EVENT_TYPES: frozenset[RuntimeEventType] = frozenset(
    {
        RuntimeEventType.PLAN_FAILED,
        RuntimeEventType.CONTEXT_VALIDATION_FAILED,
        RuntimeEventType.INGESTION_FAILED,
        RuntimeEventType.SKILL_IMPORT_FAILED,
        RuntimeEventType.STEP_FAILED,
        RuntimeEventType.TOOL_DENIED,
        RuntimeEventType.TOOL_FAILED,
        RuntimeEventType.VALIDATION_FAILED,
        RuntimeEventType.INTERRUPT_ESCALATED,
        RuntimeEventType.HUMAN_APPROVAL_TIMEOUT,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.RUNTIME_HANDLER_FAILED,
        RuntimeEventType.GUARDRAIL_BLOCKED,
        RuntimeEventType.BUDGET_EXCEEDED,
    }
)


def select_positioned_events_for_grouping(
    positioned_events: tuple[PositionedRuntimeEvent, ...],
    assessment: DiagnosticAssessment,
) -> tuple[PositionedRuntimeEvent, ...]:
    """
    Deterministic runtime-event selection for execution/operation projection.

    Include a positioned event when any of the following holds (evaluated in
    ``positioned_events`` source order):

    1. ``event.event_type`` is a failure/anomaly-relevant ``RuntimeEventType``
       (members of ``_GROUPING_FAILURE_ANOMALY_EVENT_TYPES``).
    2. ``event.event_type`` is retry-related (``RETRY_SCHEDULED`` or ``RETRY_STARTED``).
    3. ``event.event_id`` appears in ``DiagnosticFinding.supporting_event_ids`` or
       ``DiagnosticLimitation.supporting_event_ids`` on the assessment.

    Informational events that match none of the above are excluded.
    """
    referenced_event_ids: set[EventId] = set()
    for finding in assessment.findings:
        referenced_event_ids.update(finding.supporting_event_ids)
    for limitation in assessment.limitations:
        referenced_event_ids.update(limitation.supporting_event_ids)

    selected: list[PositionedRuntimeEvent] = []
    for row in positioned_events:
        event = row.event
        if (
            event.event_type in _GROUPING_FAILURE_ANOMALY_EVENT_TYPES
            or event.event_type in _RETRY_RELATED_EVENT_TYPES
            or event.event_id in referenced_event_ids
        ):
            selected.append(row)
    return tuple(selected)


class DiagnosticProblemGroupingFeatureProjector:
    """Concrete production projector — representation v2, no persistence access."""

    @property
    def representation_version(self) -> ProblemGroupingRepresentationVersion:
        return REPRESENTATION_VERSION_V2

    def project(
        self,
        assessment: DiagnosticAssessment,
        subject: ProblemGroupingSubject,
        *,
        source_facts: ProblemGroupingFeatureSourceFacts | None = None,
    ) -> ProblemGroupingFeatureSet:
        base = project_assessment_features(assessment, subject=subject)
        if source_facts is None:
            return base
        if source_facts.reconstruction is None and not source_facts.problem_signals:
            return base

        execution_context: list[ProblemGroupingExecutionFeature] = []
        operation_context: list[ProblemGroupingOperationFeature] = []
        integration_context: list[ProblemGroupingIntegrationFeature] = []
        causal_context: list[ProblemGroupingCausalFeature] = []
        component_context: list[ProblemGroupingComponentFeature] = []
        failure_context: list[ProblemGroupingFailureFeature] = []
        text_evidence = list(base.text_evidence)

        reconstruction = source_facts.reconstruction
        if reconstruction is not None:
            execution_context.extend(
                _execution_features_from_reconstruction(reconstruction, assessment)
            )
            operation_context.extend(
                _operation_features_from_reconstruction(reconstruction, assessment)
            )
            integration_context.extend(
                _integration_features_from_causal_evidence(reconstruction.causal_evidence)
            )
            causal_context.extend(
                _causal_features_from_evidence(reconstruction.causal_evidence)
            )

        for signal in source_facts.problem_signals:
            component = _component_feature_from_signal(signal)
            if component is not None:
                component_context.append(component)
            failure = _failure_feature_from_signal(signal)
            if failure is not None:
                failure_context.append(failure)
            operation = _operation_feature_from_signal(signal)
            if operation is not None:
                operation_context.append(operation)
            text = _text_evidence_from_signal(signal)
            if text is not None:
                text_evidence.append(text)

        return ProblemGroupingFeatureSet(
            subject_ref=base.subject_ref,
            representation_version=REPRESENTATION_VERSION_V2,
            structural_signature=base.structural_signature,
            execution_context=_dedupe_tuple(execution_context),
            component_context=_dedupe_tuple(component_context),
            operation_context=_dedupe_tuple(operation_context),
            integration_context=_dedupe_tuple(integration_context),
            failure_context=_dedupe_tuple(failure_context),
            causal_context=_dedupe_tuple(causal_context),
            text_evidence=_dedupe_tuple(text_evidence),
        )


def _execution_features_from_reconstruction(
    reconstruction: ExecutionReconstruction,
    assessment: DiagnosticAssessment,
) -> tuple[ProblemGroupingExecutionFeature, ...]:
    features: list[ProblemGroupingExecutionFeature] = []
    for row in select_positioned_events_for_grouping(
        reconstruction.positioned_events,
        assessment,
    ):
        event = row.event
        features.append(
            ProblemGroupingExecutionFeature(
                phase=event.phase,
                event_category=event.event_category,
                event_type=event.event_type,
                is_retry_related=event.event_type in _RETRY_RELATED_EVENT_TYPES,
                supporting_event_ids=(event.event_id,),
            )
        )
    return tuple(features)


def _operation_features_from_reconstruction(
    reconstruction: ExecutionReconstruction,
    assessment: DiagnosticAssessment,
) -> tuple[ProblemGroupingOperationFeature, ...]:
    features: list[ProblemGroupingOperationFeature] = []
    for row in select_positioned_events_for_grouping(
        reconstruction.positioned_events,
        assessment,
    ):
        event = row.event
        if not any((event.agent_id, event.node_id, event.step_id)):
            continue
        features.append(
            ProblemGroupingOperationFeature(
                agent_id=event.agent_id,
                node_id=event.node_id,
                step_id=event.step_id,
                supporting_event_ids=(event.event_id,),
            )
        )
    return tuple(features)


def _causal_features_from_evidence(
    causal_evidence: tuple[PlatformCausalEvidence, ...],
) -> tuple[ProblemGroupingCausalFeature, ...]:
    return tuple(
        ProblemGroupingCausalFeature(
            relation_kind=evidence.relation_kind,
            source_ref_kind=CAUSAL_SOURCE_REF_KIND_MESSAGE_BUS_TASK,
            target_ref_kind=CAUSAL_TARGET_REF_KIND_RUNTIME_EXECUTION,
            source_provider=evidence.source.provider,
            supporting_evidence_ids=(evidence.evidence_id,),
        )
        for evidence in causal_evidence
    )


def _integration_features_from_causal_evidence(
    causal_evidence: tuple[PlatformCausalEvidence, ...],
) -> tuple[ProblemGroupingIntegrationFeature, ...]:
    features: list[ProblemGroupingIntegrationFeature] = []
    for evidence in causal_evidence:
        provider = evidence.source.provider
        if not provider or not provider.strip():
            continue
        features.append(
            ProblemGroupingIntegrationFeature(
                provider=provider,
                supporting_evidence_ids=(evidence.evidence_id,),
            )
        )
    return tuple(features)


def _component_feature_from_signal(
    signal: PlatformProblemSignal,
) -> ProblemGroupingComponentFeature | None:
    layer = _non_empty(signal.source_layer)
    component = _non_empty(signal.source_component)
    if layer is None or component is None:
        return None
    return ProblemGroupingComponentFeature(
        source_layer=layer,
        source_component=component,
        supporting_event_ids=_optional_supporting_event_ids(signal.event_id),
    )


def _failure_feature_from_signal(
    signal: PlatformProblemSignal,
) -> ProblemGroupingFailureFeature | None:
    problem_kind = _non_empty(signal.problem_kind)
    severity = _non_empty(signal.severity)
    if problem_kind is None or severity is None:
        return None
    return ProblemGroupingFailureFeature(
        problem_kind=problem_kind,
        severity=severity,
        error_code=_non_empty(signal.error_code),
        exception_type=_non_empty(signal.exception_type) if signal.exception_type else None,
        supporting_event_ids=_optional_supporting_event_ids(signal.event_id),
    )


def _operation_feature_from_signal(
    signal: PlatformProblemSignal,
) -> ProblemGroupingOperationFeature | None:
    agent_id = _non_empty(signal.agent_id)
    tool_id = _non_empty(signal.tool_id)
    capability = _non_empty(signal.capability)
    operation = None
    if signal.application_attributes is not None:
        operation = _non_empty(signal.application_attributes.operation or "")
    if not any((agent_id, tool_id, capability, operation)):
        return None
    return ProblemGroupingOperationFeature(
        agent_id=agent_id,
        tool_id=tool_id,
        capability=capability,
        operation=operation,
        supporting_event_ids=_optional_supporting_event_ids(signal.event_id),
    )


def _text_evidence_from_signal(
    signal: PlatformProblemSignal,
) -> ProblemGroupingTextEvidence | None:
    if not signal.safe_message or not signal.safe_message.strip():
        return None
    return ProblemGroupingTextEvidence(
        source_kind=ProblemGroupingTextEvidenceSourceKind.PROBLEM_SIGNAL_SAFE_MESSAGE,
        text=_bounded_text(signal.safe_message, field_name="problem_signal.safe_message"),
        supporting_event_ids=_optional_supporting_event_ids(signal.event_id),
        supporting_evidence_ids=(),
    )


def _optional_supporting_event_ids(event_id: str) -> tuple[EventId, ...]:
    if not event_id or not event_id.strip():
        return ()
    try:
        return (validate_event_id(event_id),)
    except (TypeError, ValueError) as exc:
        raise ProblemGroupingFeatureIntegrityError(
            "problem signal event_id is not a valid EventId"
        ) from exc


def _non_empty(value: str) -> str | None:
    if type(value) is not str:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _dedupe_tuple[T](items: list[T]) -> tuple[T, ...]:
    seen: set[T] = set()
    result: list[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)
