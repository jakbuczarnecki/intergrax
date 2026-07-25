# © Artur Czarnecki. All rights reserved.

"""Managed-file Knowledge Intake acceptance and multi-file batch facade."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, Sequence

from intergrax.integrations.contracts.object_storage import ObjectStorage
from local_workspace_application.workspaces.knowledge_intake import (
    KnowledgeInputResolutionError,
    KnowledgeIntakeService,
    KnowledgeInputSourceResolver,
    deterministic_knowledge_input_id,
    deterministic_knowledge_operation_id,
)
from local_workspace_application.workspaces.models import (
    IntakeBatch,
    IntakeBatchItem,
    IntakeBatchItemStatus,
    IntakeBatchStatus,
    KnowledgeInput,
    KnowledgeInputKind,
    ManagedFileObject,
    ManagedFileObjectStatus,
    WorkspaceOperation,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_PUBLIC_MANAGED_FILE_ITEM_ERROR_CODES = frozenset(
    {
        "managed_file_name_required",
        "managed_file_name_too_long",
        "managed_file_name_unsafe",
        "managed_file_extension_required",
        "managed_file_content_type_invalid",
        "managed_file_content_type_too_long",
        "managed_file_body_required",
        "managed_file_empty",
        "managed_file_too_large",
        "managed_file_upload_read_failed",
        "managed_file_storage_read_failed",
        "managed_file_storage_write_failed",
        "knowledge_intake_accept_failed",
        "managed_file_accept_failed",
    }
)


@dataclass(frozen=True)
class ManagedFileUpload:
    file_name: str
    content_type: str
    body: bytes


@dataclass(frozen=True)
class ManagedFileBatchCandidate:
    raw_file_name: str
    raw_content_type: str
    body: bytes | None
    size_bytes: int
    body_hash: str
    request_fingerprint: str
    preflight_error_code: str | None = None


@dataclass(frozen=True)
class ManagedFileAcceptance:
    managed_file: ManagedFileObject
    knowledge_input: KnowledgeInput
    source: WorkspaceSource
    operation: WorkspaceOperation


class ManagedFileValidationError(ValueError):
    def __init__(self, error_code: str) -> None:
        code = (error_code or "").strip()
        if not code:
            raise ValueError("error_code_required")
        self.error_code = code
        super().__init__(code)


class ManagedFileIdempotencyConflict(RuntimeError):
    def __init__(self, message: str = "managed_file_idempotency_conflict") -> None:
        super().__init__(message)


class IntakeBatchIdempotencyConflict(RuntimeError):
    def __init__(self, message: str = "intake_batch_idempotency_conflict") -> None:
        super().__init__(message)


class ManagedFileCleanupPort(Protocol):
    def delete_workspace_files(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _content_hash(body: bytes) -> str:
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


def managed_file_request_fingerprint(
    *,
    raw_file_name: str,
    raw_content_type: str,
    size_bytes: int,
    body_hash: str,
) -> str:
    digest = hashlib.sha256(
        "\0".join(
            (
                raw_file_name,
                raw_content_type,
                str(size_bytes),
                body_hash,
            )
        ).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def normalize_managed_file_item_error_code(error_code: str | None) -> str:
    code = (error_code or "").strip()
    if code in _PUBLIC_MANAGED_FILE_ITEM_ERROR_CODES:
        return code
    return "managed_file_accept_failed"


def _validate_safe_file_name(file_name: str) -> str:
    raw = (file_name or "").strip()
    if not raw:
        raise ManagedFileValidationError("managed_file_name_required")
    if len(raw) > 255:
        raise ManagedFileValidationError("managed_file_name_too_long")
    if "/" in raw or "\\" in raw or "\x00" in raw or _CONTROL_RE.search(raw):
        raise ManagedFileValidationError("managed_file_name_unsafe")
    if raw in {".", ".."} or raw.endswith("."):
        raise ManagedFileValidationError("managed_file_name_unsafe")
    if "." not in raw:
        raise ManagedFileValidationError("managed_file_extension_required")
    suffix = raw.rsplit(".", 1)[1].strip()
    if not suffix:
        raise ManagedFileValidationError("managed_file_extension_required")
    return raw


def _validate_content_type(content_type: str) -> str:
    raw = (content_type or "").strip()
    if not raw:
        return "application/octet-stream"
    if _CONTROL_RE.search(raw):
        raise ManagedFileValidationError("managed_file_content_type_invalid")
    if len(raw) > 255:
        raise ManagedFileValidationError("managed_file_content_type_too_long")
    return raw


def _validate_body(body: bytes, *, max_bytes: int) -> None:
    if not isinstance(body, bytes):
        raise ManagedFileValidationError("managed_file_body_required")
    if len(body) < 1:
        raise ManagedFileValidationError("managed_file_empty")
    if len(body) > max_bytes:
        raise ManagedFileValidationError("managed_file_too_large")


def _object_id(*, tenant_id: str, workspace_id: str, input_id: str) -> str:
    return f"mfo:{_stable_digest(tenant_id, workspace_id, input_id)}"


def _storage_key(*, tenant_id: str, workspace_id: str, input_id: str) -> str:
    digest = _stable_digest(tenant_id, workspace_id, input_id)
    return f"lkw/managed/{digest}/original"


def _batch_id(*, tenant_id: str, workspace_id: str, idempotency_key: str) -> str:
    return f"batch:{_stable_digest(tenant_id, workspace_id, idempotency_key)}"


def _item_idempotency_key(*, batch_idempotency_key: str, position: int) -> str:
    return f"{batch_idempotency_key}:item:{position}"


def _item_id(*, batch_id: str, position: int) -> str:
    return f"item:{_stable_digest(batch_id, str(position))}"


def _rejected_item_file_name(position: int) -> str:
    return f"rejected-item-{position}.bin"


def _safe_batch_file_name(raw_file_name: str, position: int) -> str:
    try:
        return _validate_safe_file_name(raw_file_name)
    except ManagedFileValidationError:
        return _rejected_item_file_name(position)


def _candidate_from_upload(
    upload: ManagedFileUpload,
    *,
    max_bytes: int,
) -> ManagedFileBatchCandidate:
    raw_file_name = upload.file_name if isinstance(upload.file_name, str) else ""
    raw_content_type = upload.content_type if isinstance(upload.content_type, str) else ""
    if not isinstance(upload.body, bytes):
        body_hash = _content_hash(b"")
        return ManagedFileBatchCandidate(
            raw_file_name=raw_file_name,
            raw_content_type=raw_content_type,
            body=None,
            size_bytes=0,
            body_hash=body_hash,
            request_fingerprint=managed_file_request_fingerprint(
                raw_file_name=raw_file_name,
                raw_content_type=raw_content_type,
                size_bytes=0,
                body_hash=body_hash,
            ),
            preflight_error_code="managed_file_body_required",
        )
    body = upload.body
    size_bytes = len(body)
    body_hash = _content_hash(body)
    request_fingerprint = managed_file_request_fingerprint(
        raw_file_name=raw_file_name,
        raw_content_type=raw_content_type,
        size_bytes=size_bytes,
        body_hash=body_hash,
    )
    if size_bytes > max_bytes:
        return ManagedFileBatchCandidate(
            raw_file_name=raw_file_name,
            raw_content_type=raw_content_type,
            body=None,
            size_bytes=size_bytes,
            body_hash=body_hash,
            request_fingerprint=request_fingerprint,
            preflight_error_code="managed_file_too_large",
        )
    return ManagedFileBatchCandidate(
        raw_file_name=raw_file_name,
        raw_content_type=raw_content_type,
        body=body,
        size_bytes=size_bytes,
        body_hash=body_hash,
        request_fingerprint=request_fingerprint,
        preflight_error_code=None,
    )


class ManagedFileSourceResolver(KnowledgeInputSourceResolver):
    def __init__(self, repository: ManagedWorkspaceRepository) -> None:
        self._repository = repository

    def resolve(
        self,
        *,
        knowledge_input: KnowledgeInput,
        suggested_source_id: str,
    ) -> WorkspaceSource:
        if knowledge_input.input_kind is not KnowledgeInputKind.MANAGED_FILE:
            raise KnowledgeInputResolutionError("managed_file_kind_required")

        expected_source_id = suggested_source_id
        if (
            knowledge_input.source_id is not None
            and knowledge_input.source_id != expected_source_id
        ):
            raise KnowledgeInputResolutionError("managed_file_source_conflict")

        managed = self._repository.get_managed_file(
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            input_id=knowledge_input.input_id,
        )
        if managed is None:
            raise KnowledgeInputResolutionError("managed_file_record_missing")
        if managed.operation_id != knowledge_input.operation_id:
            raise KnowledgeInputResolutionError("managed_file_state_conflict")

        existing = self._repository.get_source(
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            source_id=expected_source_id,
        )
        if existing is not None:
            self._validate_expected_source(
                existing,
                knowledge_input=knowledge_input,
                expected_source_id=expected_source_id,
            )
            self._bind_or_validate_managed_source(
                managed,
                expected_source_id=expected_source_id,
            )
            return existing

        self._bind_or_validate_managed_source(
            managed,
            expected_source_id=expected_source_id,
        )
        now = _utc_now()
        return WorkspaceSource(
            source_id=expected_source_id,
            workspace_id=knowledge_input.workspace_id,
            tenant_id=knowledge_input.tenant_id,
            source_type=WorkspaceSourceType.MANAGED_UPLOAD,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=now,
        )

    def _bind_or_validate_managed_source(
        self,
        managed: ManagedFileObject,
        *,
        expected_source_id: str,
    ) -> ManagedFileObject:
        if managed.source_id is None:
            return self._repository.put_managed_file(
                managed.model_copy(
                    update={
                        "source_id": expected_source_id,
                        "updated_at": _utc_now(),
                    }
                )
            )
        if managed.source_id == expected_source_id:
            return managed
        raise KnowledgeInputResolutionError("managed_file_source_conflict")

    def _validate_expected_source(
        self,
        source: WorkspaceSource,
        *,
        knowledge_input: KnowledgeInput,
        expected_source_id: str,
    ) -> None:
        if (
            source.source_id != expected_source_id
            or source.tenant_id != knowledge_input.tenant_id
            or source.workspace_id != knowledge_input.workspace_id
            or source.source_type is not WorkspaceSourceType.MANAGED_UPLOAD
        ):
            raise KnowledgeInputResolutionError("managed_file_source_conflict")


class ManagedFileObjectCleanup:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        object_storage: ObjectStorage,
    ) -> None:
        self._repository = repository
        self._object_storage = object_storage

    def delete_workspace_files(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        managed_files = self._repository.list_managed_files(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        deleted = 0
        for managed in managed_files:
            try:
                self._object_storage.delete(managed.storage_key)
            except Exception as exc:  # noqa: BLE001 - fail closed for workspace delete
                raise RuntimeError("workspace_managed_file_cleanup_failed") from exc
            self._repository.delete_managed_file(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_id=managed.input_id,
            )
            deleted += 1
        self._repository.delete_intake_batches_for_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        return deleted


class ManagedFileIntakeService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        object_storage: ObjectStorage,
        knowledge_intake: KnowledgeIntakeService,
        *,
        max_bytes: int,
        max_batch_files: int,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("managed_file_max_bytes")
        if max_batch_files < 1:
            raise ValueError("managed_file_max_batch_files")
        self._repository = repository
        self._object_storage = object_storage
        self._knowledge_intake = knowledge_intake
        self._max_bytes = max_bytes
        self._max_batch_files = max_batch_files

    def accept_one(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        upload: ManagedFileUpload,
    ) -> ManagedFileAcceptance:
        tenant_id = tenant_id.strip()
        workspace_id = workspace_id.strip()
        idempotency_key = idempotency_key.strip()
        if not tenant_id or not workspace_id or not idempotency_key:
            raise ManagedFileValidationError("tenant_workspace_idempotency_required")

        workspace = self._repository.get_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise LookupError("workspace_not_found")

        if not isinstance(upload.body, bytes):
            raise ManagedFileValidationError("managed_file_body_required")
        safe_name = _validate_safe_file_name(upload.file_name)
        content_type = _validate_content_type(upload.content_type)
        body = upload.body
        _validate_body(body, max_bytes=self._max_bytes)
        content_hash = _content_hash(body)
        input_id = deterministic_knowledge_input_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key,
        )
        object_id = _object_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=input_id,
        )
        storage_key = _storage_key(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=input_id,
        )
        operation_id = deterministic_knowledge_operation_id(input_id=input_id)

        existing = self._repository.get_managed_file(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=input_id,
        )
        if existing is not None:
            self._assert_idempotent_match(
                existing,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_id=input_id,
                safe_file_name=safe_name,
                content_type=content_type,
                size_bytes=len(body),
                content_hash=content_hash,
                storage_key=storage_key,
            )
            try:
                stored = self._object_storage.get(existing.storage_key)
            except Exception:
                self._repository.put_managed_file(
                    existing.model_copy(
                        update={
                            "status": ManagedFileObjectStatus.ERROR,
                            "error_code": "managed_file_storage_read_failed",
                            "updated_at": _utc_now(),
                        }
                    )
                )
                raise ManagedFileValidationError(
                    "managed_file_storage_read_failed"
                ) from None
            if stored is None:
                self._put_bytes(storage_key=storage_key, body=body, content_type=content_type)
                existing = self._repository.put_managed_file(
                    existing.model_copy(
                        update={
                            "status": ManagedFileObjectStatus.STORED,
                            "error_code": None,
                            "updated_at": _utc_now(),
                        }
                    )
                )
            return self._resume_intake(
                managed_file=existing,
                idempotency_key=idempotency_key,
                safe_file_name=safe_name,
            )

        try:
            self._put_bytes(storage_key=storage_key, body=body, content_type=content_type)
        except ManagedFileValidationError:
            raise
        except Exception:
            raise ManagedFileValidationError("managed_file_storage_write_failed") from None

        now = _utc_now()
        managed = ManagedFileObject(
            object_id=object_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=input_id,
            operation_id=operation_id,
            source_id=None,
            storage_key=storage_key,
            safe_file_name=safe_name,
            content_type=content_type,
            size_bytes=len(body),
            content_hash=content_hash,
            status=ManagedFileObjectStatus.STORED,
            created_at=now,
            updated_at=now,
            error_code=None,
        )
        self._repository.put_managed_file(managed)
        return self._resume_intake(
            managed_file=managed,
            idempotency_key=idempotency_key,
            safe_file_name=safe_name,
        )

    def accept_many(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        uploads: Sequence[ManagedFileUpload],
    ) -> IntakeBatch:
        candidates = [
            _candidate_from_upload(upload, max_bytes=self._max_bytes) for upload in uploads
        ]
        return self.accept_prepared_many(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key,
            candidates=candidates,
        )

    def accept_prepared_many(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        candidates: Sequence[ManagedFileBatchCandidate],
    ) -> IntakeBatch:
        tenant_id = tenant_id.strip()
        workspace_id = workspace_id.strip()
        idempotency_key = idempotency_key.strip()
        if not idempotency_key:
            raise ManagedFileValidationError("managed_file_batch_idempotency_required")
        if not candidates:
            raise ManagedFileValidationError("managed_file_batch_empty")
        if len(candidates) > self._max_batch_files:
            raise ManagedFileValidationError("managed_file_batch_too_large")

        workspace = self._repository.get_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise LookupError("workspace_not_found")

        batch_id = _batch_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key,
        )
        prepared = list(candidates)
        existing = self._repository.get_intake_batch(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            batch_id=batch_id,
        )
        now = _utc_now()
        if existing is not None:
            self._assert_batch_idempotent(
                existing,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                idempotency_key=idempotency_key,
                prepared=prepared,
            )
            batch = existing
        else:
            items = [
                IntakeBatchItem(
                    position=position,
                    item_id=_item_id(batch_id=batch_id, position=position),
                    item_idempotency_key=_item_idempotency_key(
                        batch_idempotency_key=idempotency_key,
                        position=position,
                    ),
                    safe_file_name=_safe_batch_file_name(
                        candidate.raw_file_name,
                        position,
                    ),
                    status=IntakeBatchItemStatus.PENDING,
                    request_fingerprint=candidate.request_fingerprint,
                    content_hash=candidate.body_hash,
                )
                for position, candidate in enumerate(prepared)
            ]
            batch = IntakeBatch(
                batch_id=batch_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                idempotency_key=idempotency_key,
                status=IntakeBatchStatus.ACCEPTING,
                items=items,
                created_at=now,
                updated_at=now,
            )
            self._repository.put_intake_batch(batch)

        updated_items: list[IntakeBatchItem] = []
        for position, candidate in enumerate(prepared):
            item = batch.items[position]
            if item.status is IntakeBatchItemStatus.ACCEPTED:
                updated_items.append(item)
                continue
            item_key = item.item_idempotency_key
            preflight = normalize_managed_file_item_error_code(
                candidate.preflight_error_code
            ) if candidate.preflight_error_code else None
            if preflight is not None:
                updated_items.append(
                    item.model_copy(
                        update={
                            "status": IntakeBatchItemStatus.FAILED,
                            "error_code": preflight,
                            "input_id": None,
                            "source_id": None,
                            "operation_id": None,
                            "safe_file_name": _safe_batch_file_name(
                                candidate.raw_file_name,
                                position,
                            ),
                            "request_fingerprint": candidate.request_fingerprint,
                            "content_hash": candidate.body_hash,
                        }
                    )
                )
            else:
                try:
                    if candidate.body is None or not isinstance(candidate.body, bytes):
                        raise ManagedFileValidationError("managed_file_body_required")
                    acceptance = self.accept_one(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        idempotency_key=item_key,
                        upload=ManagedFileUpload(
                            file_name=candidate.raw_file_name,
                            content_type=candidate.raw_content_type,
                            body=candidate.body,
                        ),
                    )
                    updated_items.append(
                        item.model_copy(
                            update={
                                "status": IntakeBatchItemStatus.ACCEPTED,
                                "input_id": acceptance.knowledge_input.input_id,
                                "source_id": acceptance.source.source_id,
                                "operation_id": acceptance.operation.operation_id,
                                "error_code": None,
                                "content_hash": acceptance.managed_file.content_hash,
                                "safe_file_name": acceptance.managed_file.safe_file_name,
                                "request_fingerprint": candidate.request_fingerprint,
                            }
                        )
                    )
                except ManagedFileIdempotencyConflict:
                    raise IntakeBatchIdempotencyConflict(
                        "intake_batch_idempotency_conflict"
                    ) from None
                except IntakeBatchIdempotencyConflict:
                    raise
                except ManagedFileValidationError as exc:
                    error_code = normalize_managed_file_item_error_code(exc.error_code)
                    updated_items.append(
                        item.model_copy(
                            update={
                                "status": IntakeBatchItemStatus.FAILED,
                                "error_code": error_code,
                                "input_id": None,
                                "source_id": None,
                                "operation_id": None,
                                "safe_file_name": _safe_batch_file_name(
                                    candidate.raw_file_name,
                                    position,
                                ),
                                "request_fingerprint": candidate.request_fingerprint,
                                "content_hash": candidate.body_hash,
                            }
                        )
                    )
                except Exception:
                    updated_items.append(
                        item.model_copy(
                            update={
                                "status": IntakeBatchItemStatus.FAILED,
                                "error_code": "managed_file_accept_failed",
                                "input_id": None,
                                "source_id": None,
                                "operation_id": None,
                                "safe_file_name": _safe_batch_file_name(
                                    candidate.raw_file_name,
                                    position,
                                ),
                                "request_fingerprint": candidate.request_fingerprint,
                                "content_hash": candidate.body_hash,
                            }
                        )
                    )
            batch = batch.model_copy(
                update={
                    "items": [
                        updated_items[i] if i < len(updated_items) else batch.items[i]
                        for i in range(len(batch.items))
                    ],
                    "updated_at": _utc_now(),
                }
            )
            self._repository.put_intake_batch(batch)

        accepted = sum(1 for item in updated_items if item.status is IntakeBatchItemStatus.ACCEPTED)
        failed = sum(1 for item in updated_items if item.status is IntakeBatchItemStatus.FAILED)
        if accepted and failed:
            status = IntakeBatchStatus.PARTIAL
        elif accepted:
            status = IntakeBatchStatus.ACCEPTED
        else:
            status = IntakeBatchStatus.FAILED
        finalized = batch.model_copy(
            update={
                "items": updated_items,
                "status": status,
                "updated_at": _utc_now(),
            }
        )
        return self._repository.put_intake_batch(finalized)

    def _put_bytes(self, *, storage_key: str, body: bytes, content_type: str) -> None:
        self._object_storage.put(
            storage_key,
            body,
            content_type=content_type,
            metadata=None,
        )

    def _resume_intake(
        self,
        *,
        managed_file: ManagedFileObject,
        idempotency_key: str,
        safe_file_name: str,
    ) -> ManagedFileAcceptance:
        try:
            acceptance = self._knowledge_intake.accept(
                tenant_id=managed_file.tenant_id,
                workspace_id=managed_file.workspace_id,
                input_kind=KnowledgeInputKind.MANAGED_FILE,
                idempotency_key=idempotency_key,
                submission_metadata={"label": safe_file_name},
            )
        except Exception:
            self._repository.put_managed_file(
                managed_file.model_copy(
                    update={
                        "status": ManagedFileObjectStatus.ERROR,
                        "error_code": "knowledge_intake_accept_failed",
                        "updated_at": _utc_now(),
                    }
                )
            )
            raise

        updated = self._repository.put_managed_file(
            managed_file.model_copy(
                update={
                    "source_id": acceptance.source.source_id,
                    "operation_id": acceptance.operation.operation_id,
                    "status": ManagedFileObjectStatus.ACCEPTED,
                    "error_code": None,
                    "updated_at": _utc_now(),
                }
            )
        )
        return ManagedFileAcceptance(
            managed_file=updated,
            knowledge_input=acceptance.knowledge_input,
            source=acceptance.source,
            operation=acceptance.operation,
        )

    def _assert_idempotent_match(
        self,
        existing: ManagedFileObject,
        *,
        tenant_id: str,
        workspace_id: str,
        input_id: str,
        safe_file_name: str,
        content_type: str,
        size_bytes: int,
        content_hash: str,
        storage_key: str,
    ) -> None:
        if (
            existing.tenant_id != tenant_id
            or existing.workspace_id != workspace_id
            or existing.input_id != input_id
            or existing.safe_file_name != safe_file_name
            or existing.content_type != content_type
            or existing.size_bytes != size_bytes
            or existing.content_hash != content_hash
            or existing.storage_key != storage_key
        ):
            raise ManagedFileIdempotencyConflict("managed_file_idempotency_conflict")

    def _assert_batch_idempotent(
        self,
        existing: IntakeBatch,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        prepared: list[ManagedFileBatchCandidate],
    ) -> None:
        if (
            existing.tenant_id != tenant_id
            or existing.workspace_id != workspace_id
            or existing.idempotency_key != idempotency_key
            or len(existing.items) != len(prepared)
        ):
            raise IntakeBatchIdempotencyConflict("intake_batch_idempotency_conflict")
        for position, candidate in enumerate(prepared):
            item = existing.items[position]
            expected_key = _item_idempotency_key(
                batch_idempotency_key=idempotency_key,
                position=position,
            )
            expected_item_id = _item_id(batch_id=existing.batch_id, position=position)
            if (
                item.position != position
                or item.item_id != expected_item_id
                or item.item_idempotency_key != expected_key
            ):
                raise IntakeBatchIdempotencyConflict("intake_batch_idempotency_conflict")
            if item.request_fingerprint != candidate.request_fingerprint:
                raise IntakeBatchIdempotencyConflict("intake_batch_idempotency_conflict")
