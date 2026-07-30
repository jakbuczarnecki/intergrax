# © Artur Czarnecki. All rights reserved.

"""Managed-file fixture indexing for model runtime portability proof."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from local_workspace_application.model_runtime_proof.contracts import FIXTURE_TEXT
from local_workspace_application.model_runtime_proof.runtime import ProofRuntimeSession
from local_workspace_application.workspaces.models import (
    WorkspaceOperationStatus,
    WorkspaceSourceStatus,
)


_PREFIX = "/v1/local_workspace"
_FIXTURE_NAME = "model-runtime-proof.txt"


@dataclass(frozen=True, slots=True)
class IndexedFixture:
    tenant_id: str
    workspace_id: str
    input_id: str | None
    source_id: str
    operation_id: str
    document_id: str | None
    content_hash: str | None
    chunk_count: int | None


def index_managed_file_fixture(
    session: ProofRuntimeSession,
    *,
    tenant_id: str,
    workspace_name: str = "Model Runtime Proof",
) -> IndexedFixture:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            index_managed_file_fixture_async(
                session,
                tenant_id=tenant_id,
                workspace_name=workspace_name,
            )
        )
    raise RuntimeError("index_managed_file_fixture_must_be_called_without_running_loop")


async def index_managed_file_fixture_async(
    session: ProofRuntimeSession,
    *,
    tenant_id: str,
    workspace_name: str = "Model Runtime Proof",
) -> IndexedFixture:
    client = session.client
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers={"X-Tenant-Id": tenant_id},
        json={"name": workspace_name},
    )
    if created.status_code != 201:
        raise RuntimeError(
            f"workspace_create_failed:{created.status_code}:{created.text}"
        )
    workspace_id = str(created.json()["workspace_id"])

    files = {
        "files": (
            _FIXTURE_NAME,
            io.BytesIO(FIXTURE_TEXT.encode("utf-8")),
            "text/plain",
        )
    }
    accepted = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers={
            "X-Tenant-Id": tenant_id,
            "Idempotency-Key": "model-runtime-proof-fixture",
        },
        files=files,
    )
    if accepted.status_code != 202:
        raise RuntimeError(
            f"managed_file_intake_failed:{accepted.status_code}:{accepted.text}"
        )
    body = accepted.json()
    items = body.get("items") or []
    if not items:
        raise RuntimeError(f"managed_file_intake_empty_items:{body}")
    first = items[0]
    source_id = str(first["source_id"])
    operation_id = str(first["operation_id"])

    ingestion = session.app.state.lkw_knowledge_ingestion_service
    await ingestion.run_operation(tenant_id=tenant_id, operation_id=operation_id)

    repo = session.repository
    op = repo.get_operation(tenant_id=tenant_id, operation_id=operation_id)
    if op is None or op.status is not WorkspaceOperationStatus.COMPLETED:
        raise RuntimeError(
            f"indexing_operation_incomplete:{getattr(op, 'status', None)}"
        )

    source = repo.get_source(
        tenant_id=tenant_id, workspace_id=workspace_id, source_id=source_id
    )
    if source is None or source.status is not WorkspaceSourceStatus.READY:
        raise RuntimeError("source_not_ready")

    refs = repo.list_document_refs(tenant_id=tenant_id, workspace_id=workspace_id)
    if not refs:
        raise RuntimeError("document_ref_missing")
    ref = refs[0]

    input_id = str(first.get("input_id") or "") or None
    if not input_id:
        inputs = repo.list_knowledge_inputs(
            tenant_id=tenant_id, workspace_id=workspace_id
        )
        if inputs:
            input_id = inputs[0].input_id

    return IndexedFixture(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        input_id=input_id,
        source_id=source_id,
        operation_id=operation_id,
        document_id=ref.document_id,
        content_hash=ref.content_hash,
        chunk_count=getattr(op, "documents_indexed", None),
    )


def write_fixture_file(data_home: Path) -> Path:
    docs = data_home / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / _FIXTURE_NAME
    path.write_text(FIXTURE_TEXT, encoding="utf-8")
    return path
