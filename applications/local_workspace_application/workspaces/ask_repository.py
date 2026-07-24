# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed persistence for Ask Workspace runs (MVP-2)."""

from __future__ import annotations

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from local_workspace_application.workspaces.ask_models import WorkspaceAskRun

_ENTITY = "ask_run"


def _partition(tenant_id: str) -> str:
    return f"lkw.ask_run:{tenant_id}:{_ENTITY}"


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

    def get_run(self, *, tenant_id: str, run_id: str) -> WorkspaceAskRun | None:
        record = self._store.get(_partition(tenant_id), run_id)
        if record is None:
            return None
        return WorkspaceAskRun.model_validate(dict(record.data))

    def list_runs(self, *, tenant_id: str, limit: int = 2000) -> list[WorkspaceAskRun]:
        result = self._store.query(_partition(tenant_id), limit=limit)
        return [
            WorkspaceAskRun.model_validate(dict(doc.data)) for doc in result.documents
        ]

    def delete_run(self, *, tenant_id: str, run_id: str) -> None:
        self._store.delete(_partition(tenant_id), run_id)

    def delete_runs_for_workspace(self, *, tenant_id: str, workspace_id: str) -> int:
        """Policy A: remove workspace-owned Ask runs (immutable retention not required)."""
        deleted = 0
        target = (workspace_id or "").strip()
        for run in self.list_runs(tenant_id=tenant_id):
            if (run.workspace_id or "").strip() != target:
                continue
            self.delete_run(tenant_id=tenant_id, run_id=run.run_id)
            deleted += 1
        return deleted
