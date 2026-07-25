# © Artur Czarnecki. All rights reserved.

"""Managed-object materialization and Knowledge Ingestion processor."""

from __future__ import annotations

import hashlib
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from intergrax.integrations.contracts.object_storage import ObjectStorage
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingError,
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionProcessorError,
    KnowledgeIngestionResult,
)
from local_workspace_application.workspaces.models import (
    KnowledgeInput,
    KnowledgeInputKind,
    ManagedFileObject,
    ManagedFileObjectStatus,
    WorkspaceOperation,
    WorkspaceSource,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


def _staging_dir_name(*, input_id: str, operation_id: str) -> str:
    digest = hashlib.sha256(f"{input_id}\0{operation_id}".encode("utf-8")).hexdigest()
    return digest[:32]


class ManagedObjectMaterializer:
    def __init__(self, object_storage: ObjectStorage, staging_root: Path) -> None:
        self._object_storage = object_storage
        self._staging_root = staging_root

    @contextmanager
    def materialize(self, *, managed_file: ManagedFileObject) -> Iterator[Path]:
        try:
            stored = self._object_storage.get(managed_file.storage_key)
        except Exception:
            raise KnowledgeIngestionProcessorError("managed_object_read_failed") from None
        if stored is None:
            raise KnowledgeIngestionProcessorError("managed_object_missing")
        body = stored.body
        if len(body) != managed_file.size_bytes:
            raise KnowledgeIngestionProcessorError("managed_object_size_mismatch")
        digest = f"sha256:{hashlib.sha256(body).hexdigest()}"
        if digest != managed_file.content_hash:
            raise KnowledgeIngestionProcessorError("managed_object_hash_mismatch")

        op_dir = (
            self._staging_root
            / _staging_dir_name(
                input_id=managed_file.input_id,
                operation_id=managed_file.operation_id,
            )
        ).resolve()
        try:
            op_dir.relative_to(self._staging_root.resolve())
        except ValueError as exc:
            raise KnowledgeIngestionProcessorError(
                "managed_object_materialization_failed"
            ) from exc

        target = (op_dir / managed_file.safe_file_name).resolve()
        try:
            target.relative_to(op_dir)
        except ValueError as exc:
            raise KnowledgeIngestionProcessorError(
                "managed_object_materialization_failed"
            ) from exc

        try:
            op_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            yield target
        except KnowledgeIngestionProcessorError:
            raise
        except Exception as exc:  # noqa: BLE001 - map to stable code
            raise KnowledgeIngestionProcessorError(
                "managed_object_materialization_failed"
            ) from exc
        finally:
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
            try:
                if op_dir.exists() and not any(op_dir.iterdir()):
                    op_dir.rmdir()
            except OSError:
                try:
                    shutil.rmtree(op_dir, ignore_errors=True)
                except OSError:
                    pass


class ManagedFileKnowledgeIngestionProcessor:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        materializer: ManagedObjectMaterializer,
        indexing_service: WorkspaceDocumentIndexingService,
    ) -> None:
        self._repository = repository
        self._materializer = materializer
        self._indexing_service = indexing_service

    async def process(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
        operation: WorkspaceOperation,
    ) -> KnowledgeIngestionResult:
        if knowledge_input.input_kind is not KnowledgeInputKind.MANAGED_FILE:
            raise KnowledgeIngestionProcessorError("managed_file_kind_required")
        if source.source_type is not WorkspaceSourceType.MANAGED_UPLOAD:
            raise KnowledgeIngestionProcessorError("managed_file_source_conflict")

        managed = self._repository.get_managed_file(
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            input_id=knowledge_input.input_id,
        )
        if managed is None:
            raise KnowledgeIngestionProcessorError("managed_file_record_missing")

        if (
            managed.tenant_id != knowledge_input.tenant_id
            or managed.workspace_id != knowledge_input.workspace_id
            or managed.operation_id != operation.operation_id
            or (managed.source_id is not None and managed.source_id != source.source_id)
        ):
            raise KnowledgeIngestionProcessorError("managed_file_state_conflict")

        try:
            with self._materializer.materialize(managed_file=managed) as physical_path:
                logical_source_path = f"managed/{source.source_id}/{managed.safe_file_name}"
                try:
                    index_result = await self._indexing_service.index_one(
                        tenant_id=knowledge_input.tenant_id,
                        workspace_id=knowledge_input.workspace_id,
                        source_id=source.source_id,
                        operation_id=operation.operation_id,
                        physical_path=physical_path,
                        logical_source_path=logical_source_path,
                        safe_file_name=managed.safe_file_name,
                        content_hash=managed.content_hash,
                    )
                except WorkspaceDocumentIndexingError as exc:
                    raise KnowledgeIngestionProcessorError(
                        "managed_file_indexing_failed"
                    ) from exc
        except KnowledgeIngestionProcessorError as exc:
            if exc.error_code == "managed_object_missing":
                self._repository.put_managed_file(
                    managed.model_copy(
                        update={
                            "status": ManagedFileObjectStatus.MISSING,
                            "error_code": "managed_object_missing",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )
            elif exc.error_code in {
                "managed_object_read_failed",
                "managed_object_size_mismatch",
                "managed_object_hash_mismatch",
            }:
                self._repository.put_managed_file(
                    managed.model_copy(
                        update={
                            "status": ManagedFileObjectStatus.ERROR,
                            "error_code": exc.error_code,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )
            raise

        if index_result.unchanged:
            return KnowledgeIngestionResult(
                files_processed=1,
                files_failed=0,
                documents_indexed=0,
                documents_unchanged=1,
            )
        return KnowledgeIngestionResult(
            files_processed=1,
            files_failed=0,
            documents_indexed=index_result.documents_indexed,
            documents_unchanged=0,
        )
