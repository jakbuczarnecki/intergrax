# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator-safe lineage projection from execution reconstruction (DG-001 R1)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticAttemptLineageView,
    DiagnosticExecutionLineageView,
    DiagnosticExecutionNodeView,
    DiagnosticExecutionSegmentView,
)
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageReadStatus,
    ReconstructedAttemptLineage,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstruction,
)


def project_execution_lineage_view(
    reconstruction: ExecutionReconstruction,
) -> DiagnosticExecutionLineageView | None:
    if not any(attempt.lineage is not None for attempt in reconstruction.attempts):
        return None
    return DiagnosticExecutionLineageView(
        attempts=tuple(
            _project_attempt_lineage(attempt.lineage)
            for attempt in reconstruction.attempts
            if attempt.lineage is not None
        ),
    )


def _project_attempt_lineage(
    lineage: ReconstructedAttemptLineage,
) -> DiagnosticAttemptLineageView:
    segments = ()
    if lineage.read_status is ExecutionLineageReadStatus.AVAILABLE:
        segments = tuple(
            DiagnosticExecutionSegmentView(
                root_execution_id=segment.root_execution_id,
                predecessor_root_execution_id=segment.predecessor_root_execution_id,
                lifecycle=segment.lifecycle,
                executions=tuple(
                    DiagnosticExecutionNodeView(
                        execution_id=admission.execution_id,
                        parent_execution_id=admission.parent_execution_id,
                        admission_position=admission.admission_position,
                        graph_node_id=admission.graph_node_id,
                    )
                    for admission in segment.admissions
                ),
            )
            for segment in lineage.segments
        )
    return DiagnosticAttemptLineageView(
        attempt_id=lineage.attempt_id,
        read_status=lineage.read_status,
        completeness=lineage.completeness,
        degraded=lineage.degraded,
        closure_kind=lineage.closure_kind,
        segments=segments,
    )


__all__ = ["project_execution_lineage_view"]
