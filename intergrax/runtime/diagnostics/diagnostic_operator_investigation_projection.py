# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic operator investigation projections (DIAG R7)."""

from __future__ import annotations

from intergrax.contracts.diagnostic_analyzer import DiagnosticExtensionCertainty
from intergrax.contracts.diagnostic_investigation import (
    DiagnosticEvidenceExplanationConfidence,
    DiagnosticImpactNodeHealth,
    DiagnosticInvestigationSeverity,
    DiagnosticRecommendationKind,
    DiagnosticRootCauseStatus,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticFindingKind,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.diagnostic_operator_investigation_read_models import (
    DiagnosticEvidenceExplanation,
    DiagnosticExecutionContextSummary,
    DiagnosticImpactGraph,
    DiagnosticImpactGraphNode,
    DiagnosticInvestigationView,
    DiagnosticRecommendation,
    DiagnosticStructuredInvestigationPayload,
    DiagnosticTimeline,
    DiagnosticTimelineEntry,
    DiagnosticTimelineEntryKind,
    FailureInvestigationSummary,
)
from intergrax.runtime.diagnostics.diagnostic_precision import FailureBoundary
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
    DiagnosticProblemSummary,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstruction
from intergrax.runtime.events.runtime_event import RuntimeEventType


def project_investigation_view(
    *,
    problem_detail: DiagnosticProblemDetail,
    occurrence: DiagnosticProblemOccurrenceView,
    reconstruction: ExecutionReconstruction | None,
) -> DiagnosticInvestigationView:
    summary = DiagnosticProblemSummary(
        problem_id=problem_detail.problem_id,
        tenant_id=problem_detail.tenant_id,
        status=problem_detail.status,
        first_seen_at=problem_detail.first_seen_at,
        last_seen_at=problem_detail.last_seen_at,
        occurrence_count=problem_detail.occurrence_count,
        grouping_provenance=problem_detail.grouping_provenance,
        occurrence_aggregate_health=problem_detail.occurrence_aggregate_health,
    )
    assessment = occurrence.assessment
    failure_boundary = _primary_failure_boundary(assessment)
    severity = _project_severity(problem_detail, assessment)
    evidence_summary = _project_evidence_summary(occurrence, assessment)
    timeline = _project_timeline(occurrence, reconstruction, assessment)
    impact_graph = _project_impact_graph(occurrence, assessment)
    failure_investigation = _project_failure_investigation(
        occurrence,
        assessment,
        evidence_summary,
        impact_graph,
    )
    recommendations = _project_recommendations(
        occurrence,
        assessment,
        failure_investigation,
    )
    execution_context = _project_execution_context(occurrence)
    limitations = _collect_investigation_limitations(occurrence, assessment, timeline)
    assistant_payload = _project_assistant_payload(
        problem_detail=problem_detail,
        severity=severity,
        failure_investigation=failure_investigation,
        evidence_summary=evidence_summary,
        recommendations=recommendations,
        timeline=timeline,
        limitations=limitations,
    )
    affected = _affected_execution_ids(assessment)

    return DiagnosticInvestigationView(
        problem=summary,
        problem_id=problem_detail.problem_id,
        occurrence=occurrence,
        severity=severity,
        assessment=assessment,
        failure_boundary=failure_boundary,
        execution_context=execution_context,
        decision_context=occurrence.decision_context,
        extension_enrichment=occurrence.extension_enrichment,
        evidence_summary=evidence_summary,
        timeline=timeline,
        affected_execution_ids=affected,
        failure_investigation=failure_investigation,
        impact_graph=impact_graph,
        recommendations=recommendations,
        assistant_payload=assistant_payload,
        investigation_limitations=limitations,
    )


def _primary_failure_boundary(
    assessment: DiagnosticAssessment | None,
) -> FailureBoundary | None:
    if assessment is None:
        return None
    for finding in assessment.findings:
        if finding.failure_boundary is not None:
            return finding.failure_boundary
    return None


def _project_severity(
    detail: DiagnosticProblemDetail,
    assessment: DiagnosticAssessment | None,
) -> DiagnosticInvestigationSeverity:
    if assessment is not None:
        for finding in assessment.findings:
            if finding.kind is DiagnosticFindingKind.EXECUTION_FAILED:
                return DiagnosticInvestigationSeverity.HIGH
    from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus

    if detail.status is ProblemStatus.OPEN:
        return DiagnosticInvestigationSeverity.MEDIUM
    return DiagnosticInvestigationSeverity.UNKNOWN


def _project_evidence_summary(
    occurrence: DiagnosticProblemOccurrenceView,
    assessment: DiagnosticAssessment | None,
) -> tuple[DiagnosticEvidenceExplanation, ...]:
    items: list[DiagnosticEvidenceExplanation] = []
    if assessment is not None:
        for finding in assessment.findings:
            confidence = DiagnosticEvidenceExplanationConfidence.PROVEN
            if finding.certainty.value == "insufficient_evidence":
                confidence = DiagnosticEvidenceExplanationConfidence.UNKNOWN
            refs = finding.supporting_event_ids + finding.supporting_evidence_ids
            items.append(
                DiagnosticEvidenceExplanation(
                    headline=finding.kind.value.replace("_", " "),
                    confidence=confidence,
                    evidence_refs=refs,
                    detail=finding.claim,
                    is_causal_claim=False,
                )
            )
        for limitation in assessment.limitations:
            items.append(
                DiagnosticEvidenceExplanation(
                    headline=limitation.kind.value.replace("_", " "),
                    confidence=DiagnosticEvidenceExplanationConfidence.PROVEN,
                    evidence_refs=limitation.supporting_event_ids
                    + limitation.supporting_evidence_ids,
                    detail=limitation.factual_message,
                    is_causal_claim=False,
                )
            )
    if occurrence.decision_context is not None:
        for entry in occurrence.decision_context.related_decisions:
            items.append(
                DiagnosticEvidenceExplanation(
                    headline=f"Related decision {entry.decision_id}",
                    confidence=DiagnosticEvidenceExplanationConfidence.SUPPORTED,
                    evidence_refs=(),
                    detail=(
                        "Decision context is correlated execution evidence only; "
                        "it does not prove causality."
                    ),
                    is_causal_claim=False,
                )
            )
    if occurrence.extension_enrichment is not None:
        for finding in occurrence.extension_enrichment.extension_findings:
            confidence = DiagnosticEvidenceExplanationConfidence.UNKNOWN
            if finding.certainty is DiagnosticExtensionCertainty.SUPPORTED:
                confidence = DiagnosticEvidenceExplanationConfidence.SUPPORTED
            items.append(
                DiagnosticEvidenceExplanation(
                    headline=finding.summary,
                    confidence=confidence,
                    evidence_refs=finding.evidence_refs,
                    detail=f"Extension finding ({finding.kind})",
                    is_causal_claim=False,
                )
            )
    if not items and occurrence.read_status is not DiagnosticOccurrenceReadStatus.AVAILABLE:
        items.append(
            DiagnosticEvidenceExplanation(
                headline="Diagnostic assessment unavailable",
                confidence=DiagnosticEvidenceExplanationConfidence.UNKNOWN,
                evidence_refs=(),
                detail=str(occurrence.unavailable_reason or "unknown"),
                is_causal_claim=False,
            )
        )
    return tuple(items)


def _project_timeline(
    occurrence: DiagnosticProblemOccurrenceView,
    reconstruction: ExecutionReconstruction | None,
    assessment: DiagnosticAssessment | None,
) -> DiagnosticTimeline:
    entries: list[DiagnosticTimelineEntry] = []
    limitations: list[str] = []
    truncated = False

    if reconstruction is not None:
        if not reconstruction.is_runtime_history_complete:
            truncated = True
            limitations.append(
                "Runtime history is truncated; timeline may omit events after the boundary."
            )
        for positioned in reconstruction.positioned_events:
            event = positioned.event
            entries.append(
                DiagnosticTimelineEntry(
                    kind=_timeline_kind_for_event(event.event_type),
                    label=f"{event.event_type.value} ({event.event_kind or 'runtime'})",
                    observed_at=event.timestamp,
                    execution_id=event.execution_id,
                    event_id=event.event_id,
                    sort_key=(int(event.timestamp.timestamp()), positioned.position.value),
                )
            )

    if occurrence.decision_context is not None:
        for entry in occurrence.decision_context.related_decisions:
            entries.append(
                DiagnosticTimelineEntry(
                    kind=DiagnosticTimelineEntryKind.DECISION_CONTEXT,
                    label=f"Decision {entry.decision_id} correlated",
                    observed_at=None,
                    sort_key=(0, 0),
                )
            )

    if occurrence.extension_enrichment is not None:
        for evidence in occurrence.extension_enrichment.contributed_evidence:
            entries.append(
                DiagnosticTimelineEntry(
                    kind=DiagnosticTimelineEntryKind.EXTENSION_EVIDENCE,
                    label=evidence.summary,
                    observed_at=None,
                    event_id=evidence.evidence_id,
                    sort_key=(1, 0),
                )
            )

    entries.append(
        DiagnosticTimelineEntry(
            kind=DiagnosticTimelineEntryKind.PROBLEM_OCCURRENCE,
            label="Problem occurrence recorded",
            observed_at=occurrence.observed_at,
            sort_key=(int(occurrence.observed_at.timestamp()), 10_000_000),
        )
    )

    if assessment is not None:
        for finding in assessment.findings:
            if finding.kind is DiagnosticFindingKind.EXECUTION_FAILED:
                for event_id in finding.supporting_event_ids:
                    entries.append(
                        DiagnosticTimelineEntry(
                            kind=DiagnosticTimelineEntryKind.EXECUTION_FAILURE,
                            label="Execution failure evidence",
                            observed_at=None,
                            execution_id=finding.execution_id,
                            event_id=event_id,
                            sort_key=(2, 0),
                        )
                    )

    ordered = tuple(sorted(entries, key=lambda item: item.sort_key))
    return DiagnosticTimeline(
        entries=ordered,
        is_truncated=truncated,
        limitations=tuple(limitations),
    )


def _timeline_kind_for_event(event_type: RuntimeEventType) -> DiagnosticTimelineEntryKind:
    if event_type is RuntimeEventType.EXECUTION_FAILED:
        return DiagnosticTimelineEntryKind.EXECUTION_FAILURE
    return DiagnosticTimelineEntryKind.RUNTIME_EVENT


def _project_impact_graph(
    occurrence: DiagnosticProblemOccurrenceView,
    assessment: DiagnosticAssessment | None,
) -> DiagnosticImpactGraph:
    lineage = occurrence.execution_lineage
    if lineage is None or not lineage.attempts:
        return DiagnosticImpactGraph(
            root_execution_id=None,
            nodes=(),
            completeness_limitations=("Execution lineage unavailable for impact projection.",),
        )

    failed_ids: set[str] = set()
    healthy_ids: set[str] = set()
    unknown_ids: set[str] = set()
    if assessment is not None and assessment.failure_boundary_analysis is not None:
        topology = assessment.failure_boundary_analysis.topology
        failed_ids = {str(x.execution_id) for x in topology.failed_boundaries}
        healthy_ids = {str(x) for x in topology.healthy_executions}
        unknown_ids = {str(x) for x in topology.unknown_scope}

    nodes: list[DiagnosticImpactGraphNode] = []
    root_id: ExecutionId | None = None
    for attempt in lineage.attempts:
        for segment in attempt.segments:
            for node in segment.executions:
                if root_id is None and node.parent_execution_id is None:
                    root_id = node.execution_id
                health = DiagnosticImpactNodeHealth.UNKNOWN
                eid = str(node.execution_id)
                if eid in failed_ids:
                    health = DiagnosticImpactNodeHealth.FAILED
                elif eid in healthy_ids:
                    health = DiagnosticImpactNodeHealth.HEALTHY
                elif eid in unknown_ids:
                    health = DiagnosticImpactNodeHealth.UNKNOWN
                nodes.append(
                    DiagnosticImpactGraphNode(
                        execution_id=node.execution_id,
                        parent_execution_id=node.parent_execution_id,
                        health=health,
                    )
                )

    limitations: list[str] = []
    if assessment is None or assessment.failure_boundary_analysis is None:
        limitations.append(
            "Impact health labels require failure boundary analysis; nodes may be unknown."
        )

    return DiagnosticImpactGraph(
        root_execution_id=root_id,
        nodes=tuple(nodes),
        completeness_limitations=tuple(limitations),
    )


def _project_failure_investigation(
    occurrence: DiagnosticProblemOccurrenceView,
    assessment: DiagnosticAssessment | None,
    evidence_summary: tuple[DiagnosticEvidenceExplanation, ...],
    impact_graph: DiagnosticImpactGraph,
) -> FailureInvestigationSummary:
    boundaries = ()
    if assessment is not None and assessment.failure_boundary_analysis is not None:
        boundaries = assessment.failure_boundary_analysis.topology.failed_boundaries
    legacy = _primary_failure_boundary(assessment)

    related_decisions: list[str] = []
    if occurrence.decision_context is not None:
        related_decisions = [
            str(entry.decision_id)
            for entry in occurrence.decision_context.related_decisions
        ]

    unknowns: list[str] = []
    if assessment is None:
        unknowns.append("No diagnostic assessment available for this occurrence.")
    else:
        unknowns.append("Root cause is not proven by available evidence.")

    if occurrence.read_status is not DiagnosticOccurrenceReadStatus.AVAILABLE:
        unknowns.append(
            f"Occurrence diagnostics unavailable: {occurrence.unavailable_reason}"
        )

    proven_evidence = tuple(
        item
        for item in evidence_summary
        if item.confidence is DiagnosticEvidenceExplanationConfidence.PROVEN
    )

    return FailureInvestigationSummary(
        failure_boundaries=boundaries,
        legacy_failure_boundary=legacy,
        impact_root_execution_id=impact_graph.root_execution_id,
        root_cause_status=DiagnosticRootCauseStatus.UNKNOWN,
        supporting_evidence=proven_evidence,
        related_decision_ids=tuple(related_decisions),
        affected_agent_ids=(),
        explicit_unknowns=tuple(unknowns),
    )


def _project_recommendations(
    occurrence: DiagnosticProblemOccurrenceView,
    assessment: DiagnosticAssessment | None,
    failure_investigation: FailureInvestigationSummary,
) -> tuple[DiagnosticRecommendation, ...]:
    recs: list[DiagnosticRecommendation] = []
    if assessment is not None:
        for limitation in assessment.limitations:
            if limitation.kind is DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED:
                recs.append(
                    DiagnosticRecommendation(
                        kind=DiagnosticRecommendationKind.INVESTIGATE_EVIDENCE_GAP,
                        recommendation=(
                            "Investigate runtime event persistence beyond the truncated "
                            "history boundary."
                        ),
                        reason=limitation.factual_message,
                        evidence_refs=limitation.supporting_event_ids,
                    )
                )
        for finding in assessment.findings:
            if finding.kind is DiagnosticFindingKind.EXECUTION_FAILED:
                recs.append(
                    DiagnosticRecommendation(
                        kind=DiagnosticRecommendationKind.REVIEW_EXTERNAL_DEPENDENCY,
                        recommendation=(
                            "Review delegate execution boundary and retry policy."
                        ),
                        reason="Proven execution failure evidence is recorded for this scope.",
                        evidence_refs=finding.supporting_event_ids,
                    )
                )

    if occurrence.extension_enrichment is not None:
        from intergrax.runtime.diagnostics.diagnostic_extension_read_models import (
            DiagnosticExtensionPluginStatus,
        )

        if any(
            f.plugin_status is DiagnosticExtensionPluginStatus.PLUGIN_UNAVAILABLE
            for f in occurrence.extension_enrichment.extension_findings
        ):
            recs.append(
                DiagnosticRecommendation(
                    kind=DiagnosticRecommendationKind.REVIEW_EXTENSION_PLUGIN,
                    recommendation="Review diagnostic extension plugin availability.",
                    reason="Extension enrichment degraded due to plugin failure.",
                )
            )

    if failure_investigation.related_decision_ids:
        recs.append(
            DiagnosticRecommendation(
                kind=DiagnosticRecommendationKind.REVIEW_DECISION_CONTEXT,
                recommendation=(
                    "Review related decision records for contextual facts "
                    "(not as proven cause)."
                ),
                reason="Decision correlation exists for this occurrence.",
            )
        )

    if occurrence.execution_lineage is None:
        recs.append(
            DiagnosticRecommendation(
                kind=DiagnosticRecommendationKind.REVIEW_EXECUTION_LINEAGE,
                recommendation=(
                    "Review execution lineage persistence for nested impact scope."
                ),
                reason="Lineage projection was unavailable during investigation read.",
            )
        )

    return tuple(recs)


def _project_execution_context(
    occurrence: DiagnosticProblemOccurrenceView,
) -> DiagnosticExecutionContextSummary | None:
    execution = occurrence.subject_ref.execution()
    if execution is None:
        return None
    return DiagnosticExecutionContextSummary(
        task_id=str(execution.task_id),
        run_id=str(execution.run_id),
        lineage=occurrence.execution_lineage,
    )


def _collect_investigation_limitations(
    occurrence: DiagnosticProblemOccurrenceView,
    assessment: DiagnosticAssessment | None,
    timeline: DiagnosticTimeline,
) -> tuple[str, ...]:
    items: list[str] = list(timeline.limitations)
    if occurrence.decision_context is not None:
        items.extend(occurrence.decision_context.limitations)
    if occurrence.extension_enrichment is not None:
        items.extend(occurrence.extension_enrichment.limitations)
    if assessment is not None:
        for limitation in assessment.limitations:
            items.append(limitation.factual_message)
    items.append("Timeline ordering is chronological evidence only - not causal order.")
    return tuple(dict.fromkeys(items))


def _affected_execution_ids(
    assessment: DiagnosticAssessment | None,
) -> tuple[ExecutionId, ...]:
    if assessment is None or assessment.failure_boundary_analysis is None:
        return ()
    return assessment.failure_boundary_analysis.topology.affected_executions


def _project_assistant_payload(
    *,
    problem_detail: DiagnosticProblemDetail,
    severity: DiagnosticInvestigationSeverity,
    failure_investigation: FailureInvestigationSummary,
    evidence_summary: tuple[DiagnosticEvidenceExplanation, ...],
    recommendations: tuple[DiagnosticRecommendation, ...],
    timeline: DiagnosticTimeline,
    limitations: tuple[str, ...],
) -> DiagnosticStructuredInvestigationPayload:
    proven = tuple(
        item.detail or item.headline
        for item in evidence_summary
        if item.confidence is DiagnosticEvidenceExplanationConfidence.PROVEN
    )
    supported = tuple(
        item.detail or item.headline
        for item in evidence_summary
        if item.confidence is DiagnosticEvidenceExplanationConfidence.SUPPORTED
    )
    what = "Execution diagnostic problem"
    if failure_investigation.legacy_failure_boundary is not None:
        what = (
            f"Execution {failure_investigation.legacy_failure_boundary.execution_id} "
            "recorded proven failure boundary evidence."
        )
    return DiagnosticStructuredInvestigationPayload(
        problem_id=str(problem_detail.problem_id),
        tenant_id=problem_detail.tenant_id,
        severity=severity,
        what_happened=what,
        failure_boundary_execution_ids=tuple(
            str(b.execution_id) for b in failure_investigation.failure_boundaries
        ),
        impact_root_execution_id=(
            str(failure_investigation.impact_root_execution_id)
            if failure_investigation.impact_root_execution_id is not None
            else None
        ),
        root_cause_status=failure_investigation.root_cause_status,
        proven_evidence=proven,
        supported_evidence=supported,
        unknowns=failure_investigation.explicit_unknowns + limitations,
        related_decision_ids=failure_investigation.related_decision_ids,
        recommendations=tuple(r.recommendation for r in recommendations),
        timeline_labels=tuple(entry.label for entry in timeline.entries),
    )


__all__ = ["project_investigation_view"]
