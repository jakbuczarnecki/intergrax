# © Artur Czarnecki. All rights reserved.

"""Shared composition for durable run budget + deadline authority (HARNESS-02-R1A)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.execution.budget.persistence import (
    RunBudgetPersistence,
    wire_run_budget_persistence,
)
from intergrax.runtime.execution.deadline_authority import (
    ExecutionDeadlineAuthorityResolver,
    wire_execution_deadline_persistence,
)


@dataclass(frozen=True, slots=True)
class DurableExecutionRuntimeDependencies:
    """Immutable bundle wired into canonical durable :class:`ExecutionRuntime`."""

    run_budget_persistence: RunBudgetPersistence
    deadline_authority_resolver: ExecutionDeadlineAuthorityResolver


def wire_durable_execution_runtime_dependencies(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> DurableExecutionRuntimeDependencies:
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_durable_execution_runtime_dependencies accepts kv_store or "
            "document_store, not both",
        )
    if kv_store is None and document_store is None:
        raise ValueError(
            "wire_durable_execution_runtime_dependencies requires kv_store or document_store",
        )
    run_budget_persistence = wire_run_budget_persistence(
        kv_store=kv_store,
        document_store=document_store,
    )
    deadline_persistence = wire_execution_deadline_persistence(
        kv_store=kv_store,
        document_store=document_store,
    )
    resolver = ExecutionDeadlineAuthorityResolver(deadline_persistence)
    return DurableExecutionRuntimeDependencies(
        run_budget_persistence=run_budget_persistence,
        deadline_authority_resolver=resolver,
    )


__all__ = [
    "DurableExecutionRuntimeDependencies",
    "wire_durable_execution_runtime_dependencies",
]
