# © Artur Czarnecki. All rights reserved.

"""Serializable models for OBS-DG005 worker IPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
)


from testing_support.obs_distributed_topology.provider_contract import (
    EvidenceProviderDescriptor,
)


@dataclass(frozen=True, slots=True)
class PlannedRuntimeEvent:
    event_id: EventId
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    event_type: str
    timestamp_iso: str


@dataclass(frozen=True, slots=True)
class Dg005Scenario:
    qualification_sha: str
    provider: EvidenceProviderDescriptor
    primary_tenant: str
    foreign_tenant: str
    primary_task_id: TaskId
    primary_run_id: RunId
    isolated_run_id: RunId
    foreign_run_id: RunId
    primary_attempt_id: AttemptId
    primary_execution_id: ExecutionId
    primary_events: tuple[PlannedRuntimeEvent, ...]
    isolated_run_events: tuple[PlannedRuntimeEvent, ...]
    foreign_tenant_events: tuple[PlannedRuntimeEvent, ...]
    diagnostics_task_id: TaskId
    diagnostics_run_id: RunId
    diagnostics_attempt_id: AttemptId
    idempotent_event: PlannedRuntimeEvent
    reconstruction_initial_limit: int
    as_of_position_index: int


@dataclass(frozen=True, slots=True)
class PositionedEventSummary:
    event_id: str
    position: int
    tenant_id: str
    task_id: str
    run_id: str
    attempt_id: str
    execution_id: str


@dataclass(frozen=True, slots=True)
class WriterWorkerResult:
    role: Literal["writer"]
    qualification_sha: str
    import_root: str
    intergrax_file: str
    history_len_after_writes: int
    primary_summaries: tuple[PositionedEventSummary, ...]
    writer_provider_object_id: int


@dataclass(frozen=True, slots=True)
class ReaderWorkerResult:
    role: Literal["reader"]
    qualification_sha: str
    import_root: str
    intergrax_file: str
    reader_provider_object_id: int
    runtime_history_completeness: str
    event_ids_in_order: tuple[str, ...]
    positions_in_order: tuple[int, ...]
    as_of_event_ids: tuple[str, ...]
    isolated_run_event_ids: tuple[str, ...]
    foreign_tenant_visible_count: int
    task_grouped_run_ids: tuple[str, ...]
    idempotent_run_count: int


@dataclass(frozen=True, slots=True)
class DiagnosticsWorkerResult:
    role: Literal["diagnostics"]
    qualification_sha: str
    import_root: str
    intergrax_file: str
    execution_analyses: int
    grouping_candidates: int


@dataclass(frozen=True, slots=True)
class IdempotentRetryWorkerResult:
    role: Literal["idempotent_retry"]
    qualification_sha: str
    import_root: str
    intergrax_file: str
    listed_count: int
