# © Artur Czarnecki. All rights reserved.

"""Default in-process procedural memory store (MEM-ENT-8)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.memory.contracts.procedural_memory import (
    ProceduralMemoryScope,
    ProcedureMemoryViolation,
    ProcedureQuery,
    ProcedureRecord,
    ProcedureStatus,
    ProcedureSupersessionRequest,
    procedure_id_for_source_memory,
)


def _source_revision_stale(stored_revision: int | None, incoming_revision: int | None) -> bool:
    if stored_revision is None or incoming_revision is None:
        return False
    return incoming_revision < stored_revision


def _scope_user_key(scope: ProceduralMemoryScope) -> str:
    return (scope.user_id or "").strip()


def _scope_workspace_key(scope: ProceduralMemoryScope) -> str:
    if scope.workspace_id is None:
        return ""
    stripped = scope.workspace_id.strip()
    if not stripped:
        raise ProcedureMemoryViolation(
            "workspace_id when set must be non-empty for procedural scope"
        )
    return stripped


def _storage_key(scope: ProceduralMemoryScope, procedure_id: str) -> tuple[str, str, str, str]:
    tenant = (scope.tenant_id or "").strip()
    if not tenant:
        raise ProcedureMemoryViolation("tenant_id must be non-empty")
    return (tenant, _scope_user_key(scope), _scope_workspace_key(scope), procedure_id.strip())


def _procedure_projection_equal(existing: ProcedureRecord, incoming: ProcedureRecord) -> bool:
    return existing == incoming


class InMemoryProceduralMemoryStore:
    """Vendor-neutral in-memory ``ProcedureMemoryStore``."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str, str], ProcedureRecord] = {}

    def upsert_procedure(
        self,
        scope: ProceduralMemoryScope,
        record: ProcedureRecord,
    ) -> ProcedureRecord:
        key = _storage_key(scope, record.procedure_id)
        existing = self._records.get(key)
        if existing is not None:
            if _source_revision_stale(
                existing.source_memory_revision,
                record.source_memory_revision,
            ):
                return existing
            if record.source_memory_id is not None and _procedure_projection_equal(existing, record):
                return existing
            if record.source_memory_id is None:
                merged = replace(
                    record,
                    revision=max(existing.revision, record.revision),
                )
            else:
                merged = record
        else:
            merged = record
        self._records[key] = merged
        return merged

    def get_procedure(
        self,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None:
        return self._records.get(_storage_key(scope, procedure_id))

    def query_procedure_candidates(
        self,
        scope: ProceduralMemoryScope,
        query: ProcedureQuery,
    ) -> tuple[ProcedureRecord, ...]:
        tenant = (scope.tenant_id or "").strip()
        user_key = _scope_user_key(scope)
        workspace_key = _scope_workspace_key(scope)
        type_filter = (query.procedure_type or "").strip()
        matched: list[ProcedureRecord] = []
        for key, record in self._records.items():
            key_tenant, key_user, key_workspace, _pid = key
            if key_tenant != tenant:
                continue
            if key_user != user_key:
                continue
            if key_workspace != workspace_key:
                continue
            if type_filter and record.procedure_type.value != type_filter:
                continue
            if not query.include_history:
                if record.status is not ProcedureStatus.ACTIVE:
                    continue
            elif record.status not in query.statuses:
                continue
            matched.append(record)
        return tuple(matched)

    def deprecate_procedure(
        self,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None:
        key = _storage_key(scope, procedure_id)
        existing = self._records.get(key)
        if existing is None:
            return None
        updated = replace(existing, status=ProcedureStatus.DEPRECATED)
        self._records[key] = updated
        return updated

    def apply_supersession(
        self,
        scope: ProceduralMemoryScope,
        request: ProcedureSupersessionRequest,
    ) -> tuple[ProcedureRecord, ProcedureRecord]:
        old_key = _storage_key(scope, request.superseded_procedure_id)
        existing = self._records.get(old_key)
        if existing is None:
            raise ProcedureMemoryViolation("superseded procedure not found")
        superseding = request.superseding_record
        if superseding.status is not ProcedureStatus.ACTIVE:
            raise ProcedureMemoryViolation("superseding record must be ACTIVE")
        new_id = superseding.procedure_id.strip()
        superseded = replace(
            existing,
            status=ProcedureStatus.SUPERSEDED,
            superseded_by_procedure_id=new_id,
        )
        self._records[old_key] = superseded
        upserted = self.upsert_procedure(scope, superseding)
        return superseded, upserted

    def delete_by_source_memory(self, scope: ProceduralMemoryScope, source_memory_id: str) -> int:
        memory_id = (source_memory_id or "").strip()
        if not memory_id:
            return 0
        procedure_id = procedure_id_for_source_memory(scope, memory_id)
        key = _storage_key(scope, procedure_id)
        if key in self._records:
            del self._records[key]
            return 1
        return 0
