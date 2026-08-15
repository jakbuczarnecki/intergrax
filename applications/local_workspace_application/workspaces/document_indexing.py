# © Artur Czarnecki. All rights reserved.

"""Shared single-file document indexing orchestration for managed workspaces."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from local_workspace_application.workspaces.connected_source_recovery_ownership_index import (
    ConnectedSourceRecoveryOwnershipIndexError,
    index_entry_for_index_receipt,
)
from local_workspace_application.workspaces.idempotency import logical_document_id
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipModeV1,
    KnowledgeMaterializationOwnershipV1,
    KnowledgeMaterializationVisibilityAuthorityTypeV1,
)
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.runtime.task.task import Task, TaskContext


class TaskExecutorPort(Protocol):
    async def execute(self, task: Task) -> Any: ...


class IndexedVectorVerifierPort(Protocol):
    def has_indexed_vectors(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        document_id: str,
    ) -> bool: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def extract_ingest_summary(result: Any) -> dict[str, Any]:
    """Normalize indexer / task-executor result shapes into an ingest summary."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        summary = metadata.get("ingest_summary")
        if isinstance(summary, dict):
            return summary
        for key in ("domain_summary", "result"):
            nested = metadata.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("ingest_summary"), dict):
                return nested["ingest_summary"]
        evidence = metadata.get("lkw_evidence.v1")
        if isinstance(evidence, dict):
            diagnostics = evidence.get("diagnostics")
            if isinstance(diagnostics, dict):
                index_diag = diagnostics.get("lkw.index_summary.v1")
                if isinstance(index_diag, dict):
                    ingested_count = int(index_diag.get("ingested_count") or 0)
                    return {
                        "used": ingested_count > 0,
                        "reason": "ingest_complete" if ingested_count > 0 else "ingest_failed",
                    }

    execution = getattr(result, "execution_result", None)
    if execution is not None:
        structured = getattr(execution, "structured_data", None)
        if isinstance(structured, dict):
            summary = structured.get("ingest_summary")
            if isinstance(summary, dict):
                return summary

    answer = str(getattr(result, "answer", "") or "")
    if "index failed" in answer:
        return {"used": False, "reason": "index_failed"}
    if "ingested=" in answer:
        try:
            token = next(part for part in answer.split(",") if "ingested=" in part)
            count = int(token.split("ingested=", 1)[1].strip())
            return {
                "used": count > 0,
                "reason": "ingest_complete" if count > 0 else "no_paths_ingested",
            }
        except (StopIteration, ValueError):
            pass
    if "ingest_complete" in answer:
        return {"used": True, "reason": "ingest_complete"}
    return {"used": False, "reason": "ingest_summary_missing"}


class WorkspaceDocumentIndexingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indexed: bool
    unchanged: bool
    document_id: str = ""
    documents_indexed: int = Field(default=0, ge=0)
    num_chunks: int = Field(default=0, ge=0)
    reason: str = ""


class WorkspaceDocumentIndexingError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        code = (error_code or "").strip()
        if not code:
            raise ValueError("error_code_required")
        self.error_code = code
        super().__init__(code)


class _WorkspaceDocumentIndexReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    source_id: str
    operation_id: str
    logical_source_path: str
    safe_file_name: str
    content_hash: str
    document_id: str
    status: Literal["in_progress", "completed"]
    num_chunks: int = Field(default=0, ge=0)
    created_at: datetime
    completed_at: datetime | None = None
    materialization_scope: str | None = None
    materialization_ownership: KnowledgeMaterializationOwnershipV1 | None = None

    @model_validator(mode="after")
    def _validate_ownership_record(self) -> _WorkspaceDocumentIndexReceipt:
        ownership = self.materialization_ownership
        if ownership is None:
            return self
        if ownership.ownership_mode is not KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE:
            raise ValueError("index_receipt_legacy_ownership_invalid")
        if (
            ownership.tenant_id != self.tenant_id
            or ownership.workspace_id != self.workspace_id
            or ownership.source_id != self.source_id
            or self.materialization_scope != ownership.identity_scope
        ):
            raise ValueError("index_receipt_ownership_identity_mismatch")
        return self


def _index_receipt_partition(tenant_id: str) -> str:
    return f"lkw.managed_workspace:{tenant_id}:document_index_receipt"


def _index_receipt_row_key(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    logical_source_path: str,
    content_hash: str,
    materialization_scope: str | None = None,
) -> str:
    canonical = json.dumps(
        {
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "source_id": source_id,
            "logical_source_path": logical_source_path,
            "content_hash": content_hash,
            "materialization_scope": materialization_scope or "",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class WorkspaceDocumentIndexingService:
    """Owns document-ref lookup, unchanged detection and local.workspace.index invocation."""

    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        task_executor: TaskExecutorPort,
        *,
        indexed_vector_verifier: IndexedVectorVerifierPort | None = None,
    ) -> None:
        self._repository = repository
        self._task_executor = task_executor
        self._indexed_vector_verifier = indexed_vector_verifier

    def _indexed_vectors_missing(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        document_id: str,
    ) -> bool:
        if self._indexed_vector_verifier is None:
            return False
        return not self._indexed_vector_verifier.has_indexed_vectors(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            document_id=document_id,
        )

    def _get_index_receipt(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        logical_source_path: str,
        content_hash: str,
        materialization_scope: str | None,
        materialization_ownership: KnowledgeMaterializationOwnershipV1 | None = None,
    ) -> tuple[DocumentRecord, _WorkspaceDocumentIndexReceipt] | None:
        row_key = _index_receipt_row_key(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            logical_source_path=logical_source_path,
            content_hash=content_hash,
            materialization_scope=materialization_scope,
        )
        record = self._repository.document_store.get(
            _index_receipt_partition(tenant_id),
            row_key,
        )
        if record is None:
            return None
        try:
            receipt = _WorkspaceDocumentIndexReceipt.model_validate(
                dict(record.data),
                strict=False,
            )
        except ValueError:
            raise WorkspaceDocumentIndexingError("index_receipt_corrupt") from None
        if (
            receipt.tenant_id != tenant_id
            or receipt.workspace_id != workspace_id
            or receipt.source_id != source_id
            or receipt.logical_source_path != logical_source_path
            or receipt.content_hash != content_hash
            or receipt.materialization_scope != materialization_scope
        ):
            raise WorkspaceDocumentIndexingError("index_receipt_identity_conflict")
        if materialization_scope is not None:
            if receipt.materialization_ownership is None:
                raise WorkspaceDocumentIndexingError(
                    "index_receipt_migration_required"
                )
            if receipt.materialization_ownership != materialization_ownership:
                raise WorkspaceDocumentIndexingError(
                    "index_receipt_ownership_identity_conflict"
                )
        elif receipt.materialization_ownership is not None:
            raise WorkspaceDocumentIndexingError("index_receipt_identity_conflict")
        return record, receipt

    def _put_index_receipt_if_absent(
        self,
        receipt: _WorkspaceDocumentIndexReceipt,
    ) -> bool:
        document_store = self._repository.document_store
        if not isinstance(document_store, ConditionalDocumentStore):
            raise WorkspaceDocumentIndexingError("index_receipt_store_unavailable")
        partition_key = _index_receipt_partition(receipt.tenant_id)
        row_key = _index_receipt_row_key(
            tenant_id=receipt.tenant_id,
            workspace_id=receipt.workspace_id,
            source_id=receipt.source_id,
            logical_source_path=receipt.logical_source_path,
            content_hash=receipt.content_hash,
            materialization_scope=receipt.materialization_scope,
        )
        record = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=receipt.model_dump(mode="json"),
        )
        created = document_store.put_if_absent(record)
        if (
            created
            and receipt.materialization_ownership is not None
            and receipt.materialization_ownership.ownership_mode
            is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
        ):
            try:
                self._repository.put_connected_source_recovery_ownership_index_entry(
                    index_entry_for_index_receipt(
                        receipt,
                        canonical_partition_key=partition_key,
                        canonical_row_key=row_key,
                    )
                )
            except ConnectedSourceRecoveryOwnershipIndexError as exc:
                raise WorkspaceDocumentIndexingError(str(exc)) from exc
        return created

    def _complete_index_receipt(
        self,
        *,
        record: DocumentRecord,
        receipt: _WorkspaceDocumentIndexReceipt,
        num_chunks: int,
    ) -> _WorkspaceDocumentIndexReceipt:
        completed = receipt.model_copy(
            update={
                "status": "completed",
                "num_chunks": num_chunks,
                "completed_at": _utc_now(),
            }
        )
        document_store = self._repository.document_store
        if not isinstance(document_store, ConditionalDocumentStore):
            raise WorkspaceDocumentIndexingError("index_receipt_store_unavailable")
        replacement = record.model_copy(update={"data": completed.model_dump(mode="json")})
        if not document_store.replace_if_match(expected=record, replacement=replacement):
            reloaded = self._get_index_receipt(
                tenant_id=receipt.tenant_id,
                workspace_id=receipt.workspace_id,
                source_id=receipt.source_id,
                logical_source_path=receipt.logical_source_path,
                content_hash=receipt.content_hash,
                materialization_scope=receipt.materialization_scope,
                materialization_ownership=receipt.materialization_ownership,
            )
            if reloaded is None or reloaded[1].status != "completed":
                raise WorkspaceDocumentIndexingError("index_receipt_conflict")
            return reloaded[1]
        if (
            completed.materialization_ownership is not None
            and completed.materialization_ownership.ownership_mode
            is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
        ):
            try:
                self._repository.put_connected_source_recovery_ownership_index_entry(
                    index_entry_for_index_receipt(
                        completed,
                        canonical_partition_key=record.partition_key,
                        canonical_row_key=record.row_key,
                    )
                )
            except ConnectedSourceRecoveryOwnershipIndexError as exc:
                raise WorkspaceDocumentIndexingError(str(exc)) from exc
        return completed

    async def _execute_workspace_index_task(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        operation_id: str,
        physical_path: Path,
        logical_source_path: str,
        safe_file_name: str,
        content_hash: str,
        document_id: str,
        materialization_scope: str | None,
        materialization_ownership: KnowledgeMaterializationOwnershipV1,
        receipt: _WorkspaceDocumentIndexReceipt,
    ) -> WorkspaceDocumentIndexingResult:
        task = Task(
            task_id=document_id,
            tenant_id=tenant_id,
            user_id="lkw.managed_workspace",
            message=f"Index managed workspace source file {safe_file_name}",
            context=TaskContext(capability="local.workspace.index"),
            metadata={
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "source_id": source_id,
                "collection_id": workspace_id,
                "document_id": document_id,
                "source_paths": [str(physical_path)],
                "chunking_strategy_id": "recursive",
                "logical_source_path": logical_source_path,
                "display_file_name": safe_file_name,
                "content_hash": content_hash,
                "operation_id": operation_id,
                "requested_by": "lkw.managed_workspace.index",
                **(
                    {"materialization_scope": materialization_scope}
                    if materialization_scope is not None
                    else {}
                ),
            },
        )
        result = await self._task_executor.execute(task)
        ingest_summary = extract_ingest_summary(result)
        if not ingest_summary.get("used"):
            reason = str(ingest_summary.get("reason") or "ingest_failed")
            raise WorkspaceDocumentIndexingError(reason)

        completed_record = self._get_index_receipt(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            logical_source_path=logical_source_path,
            content_hash=content_hash,
            materialization_scope=materialization_scope,
            materialization_ownership=materialization_ownership,
        )
        if completed_record is None:
            raise WorkspaceDocumentIndexingError("index_receipt_missing")
        completed_receipt = self._complete_index_receipt(
            record=completed_record[0],
            receipt=receipt,
            num_chunks=int(ingest_summary.get("num_chunks") or 0),
        )
        self._repository.put_document_ref(
            WorkspaceDocumentReference(
                document_id=document_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                source_path=logical_source_path,
                file_name=safe_file_name,
                content_hash=content_hash,
                indexed_at=completed_receipt.completed_at or completed_receipt.created_at,
                materialization_ownership=materialization_ownership,
                visibility_authority_type=(
                    KnowledgeMaterializationVisibilityAuthorityTypeV1.DELIVERY_MANIFEST
                    if materialization_ownership.ownership_mode
                    is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                    else KnowledgeMaterializationVisibilityAuthorityTypeV1.LEGACY_IMMEDIATE
                ),
                visibility_authority_ref=(
                    materialization_ownership.delivery_id
                    if materialization_ownership.ownership_mode
                    is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                    else None
                ),
            )
        )
        return WorkspaceDocumentIndexingResult(
            indexed=True,
            unchanged=False,
            document_id=document_id,
            documents_indexed=1,
            num_chunks=completed_receipt.num_chunks,
            reason=str(ingest_summary.get("reason") or "ingest_complete"),
        )

    async def index_one(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        operation_id: str,
        physical_path: Path,
        logical_source_path: str,
        safe_file_name: str,
        content_hash: str,
    ) -> WorkspaceDocumentIndexingResult:
        return await self._index_one(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            operation_id=operation_id,
            physical_path=physical_path,
            logical_source_path=logical_source_path,
            safe_file_name=safe_file_name,
            content_hash=content_hash,
            materialization_ownership=KnowledgeMaterializationOwnershipV1.legacy(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
            ),
        )

    async def index_connected_source_one(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        operation_id: str,
        physical_path: Path,
        logical_source_path: str,
        safe_file_name: str,
        content_hash: str,
        materialization_ownership: KnowledgeMaterializationOwnershipV1,
        document_id: str | None = None,
    ) -> WorkspaceDocumentIndexingResult:
        if (
            materialization_ownership.ownership_mode
            is not KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
            or materialization_ownership.tenant_id != tenant_id
            or materialization_ownership.workspace_id != workspace_id
            or materialization_ownership.source_id != source_id
        ):
            raise WorkspaceDocumentIndexingError(
                "connected_materialization_ownership_required"
            )
        return await self._index_one(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            operation_id=operation_id,
            physical_path=physical_path,
            logical_source_path=logical_source_path,
            safe_file_name=safe_file_name,
            content_hash=content_hash,
            materialization_ownership=materialization_ownership,
            document_id=document_id,
        )

    async def _index_one(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        operation_id: str,
        physical_path: Path,
        logical_source_path: str,
        safe_file_name: str,
        content_hash: str,
        materialization_ownership: KnowledgeMaterializationOwnershipV1,
        document_id: str | None = None,
    ) -> WorkspaceDocumentIndexingResult:
        materialization_scope = (
            materialization_ownership.identity_scope
            if materialization_ownership.ownership_mode
            is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
            else None
        )
        document_id = document_id or logical_document_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            normalized_source_path=logical_source_path,
            content_hash=content_hash,
            materialization_scope=materialization_scope,
        )
        receipt_record = self._get_index_receipt(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            logical_source_path=logical_source_path,
            content_hash=content_hash,
            materialization_scope=materialization_scope,
            materialization_ownership=materialization_ownership,
        )
        if receipt_record is not None:
            _, receipt = receipt_record
            if receipt.document_id != document_id:
                raise WorkspaceDocumentIndexingError("index_receipt_identity_conflict")
            if receipt.status == "in_progress":
                return await self._execute_workspace_index_task(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    operation_id=operation_id,
                    physical_path=physical_path,
                    logical_source_path=logical_source_path,
                    safe_file_name=safe_file_name,
                    content_hash=content_hash,
                    document_id=document_id,
                    materialization_scope=materialization_scope,
                    materialization_ownership=materialization_ownership,
                    receipt=receipt,
                )
            vectors_missing = self._indexed_vectors_missing(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                document_id=receipt.document_id,
            )
            if vectors_missing:
                return await self._execute_workspace_index_task(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    operation_id=operation_id,
                    physical_path=physical_path,
                    logical_source_path=logical_source_path,
                    safe_file_name=safe_file_name,
                    content_hash=content_hash,
                    document_id=document_id,
                    materialization_scope=materialization_scope,
                    materialization_ownership=materialization_ownership,
                    receipt=receipt,
                )
            self._repository.put_document_ref(
                WorkspaceDocumentReference(
                    document_id=receipt.document_id,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    source_path=logical_source_path,
                    file_name=receipt.safe_file_name,
                    content_hash=content_hash,
                    indexed_at=receipt.completed_at or receipt.created_at,
                    materialization_ownership=materialization_ownership,
                    visibility_authority_type=(
                        KnowledgeMaterializationVisibilityAuthorityTypeV1.DELIVERY_MANIFEST
                        if materialization_ownership.ownership_mode
                        is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                        else KnowledgeMaterializationVisibilityAuthorityTypeV1.LEGACY_IMMEDIATE
                    ),
                    visibility_authority_ref=(
                        materialization_ownership.delivery_id
                        if materialization_ownership.ownership_mode
                        is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                        else None
                    ),
                )
            )
            if receipt.operation_id == operation_id:
                return WorkspaceDocumentIndexingResult(
                    indexed=True,
                    unchanged=False,
                    document_id=receipt.document_id,
                    documents_indexed=1,
                    num_chunks=receipt.num_chunks,
                    reason="index_replayed",
                )
            return WorkspaceDocumentIndexingResult(
                indexed=False,
                unchanged=True,
                document_id=receipt.document_id,
                documents_indexed=0,
                num_chunks=0,
                reason="unchanged",
            )

        existing = self._repository.get_document_ref_by_path(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            source_path=logical_source_path,
        )
        if existing is not None and existing.content_hash == content_hash:
            vectors_missing = self._indexed_vectors_missing(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                document_id=existing.document_id,
            )
            if not vectors_missing:
                if (
                    materialization_ownership.ownership_mode
                    is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                ):
                    self._repository.put_document_ref(
                        existing.model_copy(
                            update={
                                "materialization_ownership": materialization_ownership,
                                "visibility_authority_type": (
                                    KnowledgeMaterializationVisibilityAuthorityTypeV1.DELIVERY_MANIFEST
                                ),
                                "visibility_authority_ref": materialization_ownership.delivery_id,
                            }
                        )
                    )
                return WorkspaceDocumentIndexingResult(
                    indexed=False,
                    unchanged=True,
                    document_id=existing.document_id,
                    documents_indexed=0,
                    num_chunks=0,
                    reason="unchanged",
                )

        receipt = _WorkspaceDocumentIndexReceipt(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            operation_id=operation_id,
            logical_source_path=logical_source_path,
            safe_file_name=safe_file_name,
            content_hash=content_hash,
            document_id=document_id,
            status="in_progress",
            created_at=_utc_now(),
            materialization_scope=materialization_scope,
            materialization_ownership=(
                materialization_ownership
                if materialization_ownership.ownership_mode
                is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                else None
            ),
        )
        if not self._put_index_receipt_if_absent(receipt):
            reloaded = self._get_index_receipt(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                logical_source_path=logical_source_path,
                content_hash=content_hash,
                materialization_scope=materialization_scope,
                materialization_ownership=materialization_ownership,
            )
            if reloaded is None:
                raise WorkspaceDocumentIndexingError("index_receipt_conflict")
            _, existing_receipt = reloaded
            if existing_receipt.status == "in_progress":
                return await self._execute_workspace_index_task(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    operation_id=operation_id,
                    physical_path=physical_path,
                    logical_source_path=logical_source_path,
                    safe_file_name=safe_file_name,
                    content_hash=content_hash,
                    document_id=document_id,
                    materialization_scope=materialization_scope,
                    materialization_ownership=materialization_ownership,
                    receipt=existing_receipt,
                )
            if existing_receipt.status != "completed":
                raise WorkspaceDocumentIndexingError("index_recovery_required")
            return await self._index_one(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                operation_id=operation_id,
                physical_path=physical_path,
                logical_source_path=logical_source_path,
                safe_file_name=safe_file_name,
                content_hash=content_hash,
                materialization_ownership=materialization_ownership,
                document_id=document_id,
            )

        return await self._execute_workspace_index_task(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            operation_id=operation_id,
            physical_path=physical_path,
            logical_source_path=logical_source_path,
            safe_file_name=safe_file_name,
            content_hash=content_hash,
            document_id=document_id,
            materialization_scope=materialization_scope,
            materialization_ownership=materialization_ownership,
            receipt=receipt,
        )
