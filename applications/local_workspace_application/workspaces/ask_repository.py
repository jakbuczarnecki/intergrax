# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed persistence for Ask Workspace runs (MVP-2 + Hybrid Ask V2)."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import ValidationError

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from local_workspace_application.workspaces.ask_models import WorkspaceAskRun
from local_workspace_application.workspaces.hybrid_ask_models import WorkspaceAskRunV2

_ENTITY = "ask_run"


class AskRunSchemaVersion(IntEnum):
    V1 = 1
    V2 = 2


class WorkspaceAskRepositoryError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def _partition(tenant_id: str) -> str:
    return f"lkw.ask_run:{tenant_id}:{_ENTITY}"


def detect_ask_run_schema_version(data: dict[str, Any]) -> AskRunSchemaVersion:
    version = data.get("run_schema_version")
    if version is None:
        return AskRunSchemaVersion.V1
    if version == 2:
        return AskRunSchemaVersion.V2
    raise WorkspaceAskRepositoryError("ask_run_schema_version_unknown")


def _parse_ask_run(data: dict[str, Any]) -> WorkspaceAskRun | WorkspaceAskRunV2:
    schema_version = detect_ask_run_schema_version(data)
    try:
        if schema_version is AskRunSchemaVersion.V1:
            return WorkspaceAskRun.model_validate(data)
        return WorkspaceAskRunV2.model_validate(data)
    except ValidationError as exc:
        raise WorkspaceAskRepositoryError("ask_run_malformed") from exc


class WorkspaceAskRepository:
    """Tier-3 Ask-run repository over the provider-neutral DocumentStore contract."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    def put_run(self, run: WorkspaceAskRun) -> WorkspaceAskRun:
        self._store.put(
            DocumentRecord(
                partition_key=_partition(run.tenant_id),
                row_key=run.run_id,
                data=run.model_dump(mode="json"),
            )
        )
        return run

    def put_run_v2(self, run: WorkspaceAskRunV2) -> WorkspaceAskRunV2:
        payload = run.model_dump(mode="json")
        self._store.put(
            DocumentRecord(
                partition_key=_partition(run.tenant_id),
                row_key=run.run_id,
                data=payload,
            )
        )
        return run

    def get_run(self, *, tenant_id: str, run_id: str) -> WorkspaceAskRun | None:
        record = self._store.get(_partition(tenant_id), run_id)
        if record is None:
            return None
        parsed = _parse_ask_run(dict(record.data))
        if isinstance(parsed, WorkspaceAskRunV2):
            raise WorkspaceAskRepositoryError("ask_run_schema_version_mismatch")
        return parsed

    def get_run_v2(self, *, tenant_id: str, run_id: str) -> WorkspaceAskRunV2 | None:
        record = self._store.get(_partition(tenant_id), run_id)
        if record is None:
            return None
        parsed = _parse_ask_run(dict(record.data))
        if isinstance(parsed, WorkspaceAskRun):
            raise WorkspaceAskRepositoryError("ask_run_schema_version_mismatch")
        return parsed

    def get_run_any(
        self,
        *,
        tenant_id: str,
        run_id: str,
    ) -> tuple[WorkspaceAskRun | WorkspaceAskRunV2, AskRunSchemaVersion] | None:
        record = self._store.get(_partition(tenant_id), run_id)
        if record is None:
            return None
        data = dict(record.data)
        schema_version = detect_ask_run_schema_version(data)
        return _parse_ask_run(data), schema_version

    def get_stored_run_schema_version(
        self,
        *,
        tenant_id: str,
        run_id: str,
    ) -> AskRunSchemaVersion | None:
        record = self._store.get(_partition(tenant_id), run_id)
        if record is None:
            return None
        return detect_ask_run_schema_version(dict(record.data))

    def list_runs(self, *, tenant_id: str, limit: int = 2000) -> list[WorkspaceAskRun]:
        result = self._store.query(_partition(tenant_id), limit=limit)
        runs: list[WorkspaceAskRun] = []
        for doc in result.documents:
            parsed = _parse_ask_run(dict(doc.data))
            if isinstance(parsed, WorkspaceAskRunV2):
                raise WorkspaceAskRepositoryError("ask_run_schema_version_mismatch")
            runs.append(parsed)
        return runs

    def list_runs_v2(self, *, tenant_id: str, limit: int = 2000) -> list[WorkspaceAskRunV2]:
        result = self._store.query(_partition(tenant_id), limit=limit)
        runs: list[WorkspaceAskRunV2] = []
        for doc in result.documents:
            parsed = _parse_ask_run(dict(doc.data))
            if isinstance(parsed, WorkspaceAskRun):
                raise WorkspaceAskRepositoryError("ask_run_schema_version_mismatch")
            runs.append(parsed)
        return runs

    def delete_run(self, *, tenant_id: str, run_id: str) -> None:
        self._store.delete(_partition(tenant_id), run_id)

    def delete_runs_for_workspace(self, *, tenant_id: str, workspace_id: str) -> int:
        """Policy A: remove workspace-owned Ask runs (immutable retention not required)."""
        deleted = 0
        target = (workspace_id or "").strip()
        result = self._store.query(_partition(tenant_id), limit=2000)
        for doc in result.documents:
            parsed = _parse_ask_run(dict(doc.data))
            if (parsed.workspace_id or "").strip() != target:
                continue
            self.delete_run(tenant_id=tenant_id, run_id=parsed.run_id)
            deleted += 1
        return deleted
