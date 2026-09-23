# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition-only wiring for worker qualified capability resume (UCA-6C-R6-R5.8-R2-H1-R1)."""

from __future__ import annotations

from intergrax.autonomous_work.execution_authority_admission import WorkerExecutionAdmissionPort
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_ports import (
    QualifiedCapabilityBindingPort,
    WorkerQualifiedCapabilityAsyncExecutionPort,
    WorkerQualifiedCapabilityExecutionPort,
)


def build_worker_qualified_capability_resume_coordinator(
    *,
    binding: QualifiedCapabilityBindingPort,
    execution: WorkerQualifiedCapabilityExecutionPort,
    async_execution: WorkerQualifiedCapabilityAsyncExecutionPort | None = None,
    authority_admission: WorkerExecutionAdmissionPort | None = None,
) -> WorkerQualifiedCapabilityResumeCoordinator:
    """Construct resume coordinator with explicit port dependencies only."""
    return WorkerQualifiedCapabilityResumeCoordinator(
        binding=binding,
        execution=execution,
        authority_admission=authority_admission,
        async_execution=async_execution,
    )


__all__ = ["build_worker_qualified_capability_resume_coordinator"]
