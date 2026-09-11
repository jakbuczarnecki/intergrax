# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic multi-agent failure boundary localization (DIAG R3)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.multi_agent_failure_localization import (
    DiagnosticFailureBoundary,
    DiagnosticFailureTopology,
    DiagnosticFailureTopologyCompleteness,
    FailureBoundaryAnalysis,
    FailureBoundaryCertainty,
    FailureBoundaryPrecision,
)
from typing import TYPE_CHECKING

from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageReadStatus,
    ReconstructedAttemptLineage,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstruction

from intergrax.runtime.diagnostics.diagnostic_precision import DiagnosticPrecision

if TYPE_CHECKING:
    from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFinding


def _lineage_execution_index(
    lineage: ReconstructedAttemptLineage,
) -> tuple[dict[ExecutionId, ExecutionId | None], frozenset[ExecutionId]]:
    parent_by_execution: dict[ExecutionId, ExecutionId | None] = {}
    for segment in lineage.segments:
        for admission in segment.admissions:
            parent_by_execution[admission.execution_id] = admission.parent_execution_id
    return parent_by_execution, frozenset(parent_by_execution)


def _ancestors(
    execution_id: ExecutionId,
    parent_by_execution: dict[ExecutionId, ExecutionId | None],
) -> tuple[ExecutionId, ...]:
    chain: list[ExecutionId] = []
    current = parent_by_execution.get(execution_id)
    while current is not None:
        chain.append(current)
        current = parent_by_execution.get(current)
    return tuple(chain)


def _deepest_failed_execution_ids(
    failed_ids: frozenset[ExecutionId],
    parent_by_execution: dict[ExecutionId, ExecutionId | None],
) -> frozenset[ExecutionId]:
    if not failed_ids:
        return frozenset()
    has_failed_child: set[ExecutionId] = set()
    for execution_id in failed_ids:
        parent_id = parent_by_execution.get(execution_id)
        if parent_id is not None and parent_id in failed_ids:
            has_failed_child.add(parent_id)
    return frozenset(
        execution_id for execution_id in failed_ids if execution_id not in has_failed_child
    )


def _boundary_from_finding(finding: DiagnosticFinding) -> DiagnosticFailureBoundary:  # noqa: F821
    event_ids = finding.supporting_event_ids
    evidence_refs = event_ids
    if finding.failure_boundary is not None:
        evidence_refs = (finding.failure_boundary.supporting_event_id,) + tuple(
            event_id
            for event_id in event_ids
            if event_id != finding.failure_boundary.supporting_event_id
        )
    execution_id = finding.execution_id
    if execution_id is None:
        raise ValueError("EXECUTION_FAILED finding missing execution_id")
    precision = FailureBoundaryPrecision.EXECUTION_LEVEL
    if finding.precision is not None:
        if finding.precision is DiagnosticPrecision.EXTERNAL_BOUNDARY:
            precision = FailureBoundaryPrecision.EXTERNAL_BOUNDARY
        else:
            precision = FailureBoundaryPrecision(finding.precision.value)
    return DiagnosticFailureBoundary(
        execution_id=execution_id,
        precision=precision,
        certainty=FailureBoundaryCertainty.PROVEN,
        evidence_refs=evidence_refs,
    )


class ExecutionFailureTopologyAnalyzer:
    """
    Localizes proven execution failure boundaries and lineage-derived impact.

    Never infers cause from parent-child lineage edges.
    """

    def analyze(
        self,
        reconstruction: ExecutionReconstruction,
        findings: tuple[DiagnosticFinding, ...],
    ) -> FailureBoundaryAnalysis:
        from intergrax.runtime.diagnostics.diagnostic_assessment import (
            DiagnosticFindingKind,
        )

        failure_findings = tuple(
            finding
            for finding in findings
            if finding.kind
            in {
                DiagnosticFindingKind.EXECUTION_FAILED,
                DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED,
            }
        )
        if not failure_findings:
            return FailureBoundaryAnalysis.unavailable()

        proven_boundaries = tuple(_boundary_from_finding(f) for f in failure_findings)
        failed_ids = frozenset(b.execution_id for b in proven_boundaries)

        attempts_by_id = {attempt.attempt_id: attempt for attempt in reconstruction.attempts}
        parent_maps: list[dict[ExecutionId, ExecutionId | None]] = []
        all_lineage_ids: set[ExecutionId] = set()
        lineage_usable = True
        lineage_degraded = False

        attempt_ids = {finding.attempt_id for finding in failure_findings}
        for attempt_id in attempt_ids:
            if attempt_id is None:
                lineage_usable = False
                continue
            attempt = attempts_by_id.get(attempt_id)
            if attempt is None or attempt.lineage is None:
                lineage_usable = False
                continue
            lineage = attempt.lineage
            if lineage.read_status is not ExecutionLineageReadStatus.AVAILABLE:
                lineage_usable = False
                continue
            if lineage.degraded:
                lineage_degraded = True
            parent_map, lineage_ids = _lineage_execution_index(lineage)
            parent_maps.append(parent_map)
            all_lineage_ids.update(lineage_ids)

        if not lineage_usable or not parent_maps:
            topology = DiagnosticFailureTopology(
                failed_boundaries=proven_boundaries,
                affected_executions=(),
                healthy_executions=(),
                unknown_scope=tuple(sorted(failed_ids, key=str)),
                completeness=DiagnosticFailureTopologyCompleteness.UNAVAILABLE,
            )
            return FailureBoundaryAnalysis(
                boundaries=proven_boundaries,
                topology=topology,
            )

        merged_parent: dict[ExecutionId, ExecutionId | None] = {}
        for parent_map in parent_maps:
            merged_parent.update(parent_map)

        deepest_failed = _deepest_failed_execution_ids(failed_ids, merged_parent)
        localized_boundaries = tuple(
            boundary
            for boundary in proven_boundaries
            if boundary.execution_id in deepest_failed
        )
        if not localized_boundaries:
            localized_boundaries = proven_boundaries

        affected: list[ExecutionId] = []
        seen_affected: set[ExecutionId] = set()
        boundary_ids = {boundary.execution_id for boundary in localized_boundaries}
        for boundary_id in boundary_ids:
            for ancestor in _ancestors(boundary_id, merged_parent):
                if ancestor in seen_affected or ancestor in boundary_ids:
                    continue
                seen_affected.add(ancestor)
                affected.append(ancestor)

        healthy: list[ExecutionId] = []
        for execution_id in sorted(all_lineage_ids, key=str):
            if execution_id in failed_ids:
                continue
            if execution_id in seen_affected:
                continue
            healthy.append(execution_id)

        completeness = DiagnosticFailureTopologyCompleteness.COMPLETE
        if lineage_degraded:
            completeness = DiagnosticFailureTopologyCompleteness.DEGRADED

        topology = DiagnosticFailureTopology(
            failed_boundaries=localized_boundaries,
            affected_executions=tuple(affected),
            healthy_executions=tuple(healthy),
            unknown_scope=(),
            completeness=completeness,
        )
        return FailureBoundaryAnalysis(
            boundaries=localized_boundaries,
            topology=topology,
        )


__all__ = ["ExecutionFailureTopologyAnalyzer"]
