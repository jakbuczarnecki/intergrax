# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineagePersistence,
    build_execution_lineage_run_scope,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.lineage.codecs import encode_execution_lineage_attempt_scope
from intergrax.runtime.execution.lineage.persistence import execution_lineage_partition_key


def register_v1_attempt(
    persistence: ExecutionLineagePersistence,
    scope: ExecutionLineageAttemptScope,
) -> ExecutionLineageAttemptState:
    run_scope = build_execution_lineage_run_scope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    persistence.register_attempt_for_run(run_scope, scope.attempt_id)
    return persistence.open_attempt(scope, discovery_contract_version=1)


def seed_legacy_attempt_state(
    document_store: InMemoryDocumentStore,
    scope: ExecutionLineageAttemptScope,
) -> ExecutionLineageAttemptState:
    state = ExecutionLineageAttemptState(
        scope=scope,
        generation=1,
        next_admission_position=1,
        active_segment_root_execution_id=None,
        degraded=False,
        sealed=False,
        closure_kind=None,
        discovery_contract_version=None,
    )
    payload = {
        "schema_version": 1,
        "scope": encode_execution_lineage_attempt_scope(scope),
        "generation": state.generation,
        "next_admission_position": state.next_admission_position,
        "active_segment_root_execution_id": None,
        "degraded": False,
        "sealed": False,
        "closure_kind": None,
    }
    document_store.put_if_absent(
        DocumentRecord(
            partition_key=execution_lineage_partition_key(scope),
            row_key="meta:attempt",
            data=payload,
        ),
    )
    return state
