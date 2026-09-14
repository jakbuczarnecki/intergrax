# © Artur Czarnecki. All rights reserved.

"""Typed models for EE-B4-B reference shutdown certification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from intergrax.contracts.execution_reliability import ExecutionRuntimeShutdownPhase


class ReferenceRootAdmissionDecision(StrEnum):
    ADMITTED = "admitted"
    REJECTED_SHUTDOWN = "rejected_shutdown"
    REJECTED_TERMINATED = "rejected_terminated"
    REJECTED_CAPACITY = "rejected_capacity"


class ReferenceShutdownFailureKind(StrEnum):
    WORKER = "worker"
    MANDATORY_EVIDENCE = "mandatory_evidence"
    FINAL_STATE = "final_state"
    DRAIN_TIMEOUT = "drain_timeout"
    OBSERVABILITY_EXPORT = "observability_export"


@dataclass(frozen=True, slots=True)
class ReferenceShutdownTerminalOutcome:
    """Deterministic shutdown terminal report for certification assertions."""

    clean_success: bool
    primary_failure_kind: ReferenceShutdownFailureKind | None
    secondary_failure_kinds: tuple[ReferenceShutdownFailureKind, ...]
    phase_trail: tuple[ExecutionRuntimeShutdownPhase, ...]
    held_root_permits: int
    managed_worker_count: int
    managed_task_count: int
    double_release_attempts: int
    new_roots_after_stop: int


@dataclass(slots=True)
class _ShutdownState:
    phase: ExecutionRuntimeShutdownPhase | None = None
    terminated: bool = False
    outcome: ReferenceShutdownTerminalOutcome | None = None
    stop_boundary_reached: bool = False
    new_roots_after_stop: int = 0
    double_release_attempts: int = 0
    worker_fault_during_drain: bool = False
    phase_trail: list[ExecutionRuntimeShutdownPhase] = field(default_factory=list)
