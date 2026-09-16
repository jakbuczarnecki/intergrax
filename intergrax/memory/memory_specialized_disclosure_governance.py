# © Artur Czarnecki. All rights reserved.

"""Shared governance helpers for specialized memory disclosure paths (MEM-ENT-10C)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TypeVar

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceRecordSnapshot,
    MemoryGovernanceTarget,
    MemorySecurityContext,
)
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.memory_specialized_mutation_governance import memory_control_scope_from_entity_scope

__all__ = [
    "evaluate_memory_disclosure",
    "filter_memory_disclosure_candidates",
    "memory_security_context_for_recall",
]

_T = TypeVar("_T")


def memory_security_context_for_recall(
    identity: RequestIdentity,
    scope: EntityMemoryScope,
    *,
    reference_time: datetime | None = None,
) -> MemorySecurityContext:
    control_scope = memory_control_scope_from_entity_scope(scope)
    return MemorySecurityContext(
        identity=identity,
        scope=control_scope,
        operation=MemoryGovernanceOperation.RECALL,
        reference_time=reference_time,
    )


def evaluate_memory_disclosure(
    governance: MemorySecurityGovernanceService,
    context: MemorySecurityContext,
    existing_record: MemoryGovernanceRecordSnapshot,
    *,
    source_records: tuple[MemoryGovernanceRecordSnapshot, ...] = (),
) -> bool:
    request = MemoryGovernanceEvaluationRequest(
        context=context,
        target=MemoryGovernanceTarget(
            memory_id=existing_record.memory_id,
            revision=existing_record.revision,
            kind=existing_record.kind,
            scope=context.scope,
        ),
        existing_record=existing_record,
        source_records=source_records,
    )
    return governance.evaluate(request).permits_disclosure()


def filter_memory_disclosure_candidates(
    governance: MemorySecurityGovernanceService,
    context: MemorySecurityContext,
    candidates: tuple[_T, ...],
    *,
    to_snapshot: Callable[[_T], MemoryGovernanceRecordSnapshot],
    source_records_for: Callable[[_T], tuple[MemoryGovernanceRecordSnapshot, ...]]
    | None = None,
) -> tuple[_T, ...]:
    allowed: list[_T] = []
    for candidate in candidates:
        snapshot = to_snapshot(candidate)
        sources = source_records_for(candidate) if source_records_for is not None else ()
        if evaluate_memory_disclosure(
            governance,
            context,
            snapshot,
            source_records=sources,
        ):
            allowed.append(candidate)
    return tuple(allowed)
