# © Artur Czarnecki. All rights reserved.

"""WEB_URL Knowledge Intake acceptance, resolution, capture and indexing."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from intergrax.websearch.capture.contracts import (
    CapturedWebContent,
    WebContentCapture,
    WebContentCaptureError,
    WebContentCaptureErrorCode,
    WebContentCaptureRequest,
)
from intergrax.websearch.capture.url_policy import WebUrlAccessPolicy
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingError,
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionProcessorError,
    KnowledgeIngestionResult,
)
from local_workspace_application.workspaces.knowledge_intake import (
    KnowledgeInputIdempotencyConflict,
    KnowledgeInputResolutionError,
    KnowledgeIntakeDispatchError,
    KnowledgeIntakeService,
    deterministic_knowledge_input_id,
)
from local_workspace_application.workspaces.models import (
    KnowledgeInput,
    KnowledgeInputKind,
    WebUrlSourceLocator,
    WorkspaceOperation,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_METADATA_KEY = "source_fingerprint"
_STAGING_FILE_NAME = "web-content.txt"

_BAD_REQUEST_CAPTURE_CODES = frozenset(
    {
        WebContentCaptureErrorCode.WEB_URL_INVALID,
        WebContentCaptureErrorCode.WEB_URL_SCHEME_NOT_ALLOWED,
        WebContentCaptureErrorCode.WEB_URL_CREDENTIALS_NOT_ALLOWED,
        WebContentCaptureErrorCode.WEB_URL_PORT_NOT_ALLOWED,
        WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED,
        WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED,
    }
)

_SERVICE_UNAVAILABLE_CAPTURE_CODES = frozenset(
    {
        WebContentCaptureErrorCode.WEB_URL_RESOLUTION_FAILED,
        WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
    }
)


class WebUrlValidationError(ValueError):
    def __init__(self, error_code: str) -> None:
        code = (error_code or "").strip()
        if not code:
            raise ValueError("error_code_required")
        self.error_code = code
        super().__init__(code)


class WebUrlIdempotencyConflict(RuntimeError):
    def __init__(self, message: str = "web_url_idempotency_conflict") -> None:
        super().__init__(message)


class WebUrlAlreadyRegistered(RuntimeError):
    def __init__(self, message: str = "web_url_already_registered") -> None:
        super().__init__(message)


class WebUrlStateConflict(RuntimeError):
    def __init__(self, message: str = "web_url_state_conflict") -> None:
        super().__init__(message)


@dataclass(frozen=True)
class WebUrlAcceptance:
    input_id: str
    workspace_id: str
    source_id: str
    operation_id: str
    status: str
    safe_display_url: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _staging_dir_name(*, input_id: str, operation_id: str) -> str:
    digest = hashlib.sha256(f"{input_id}\0{operation_id}".encode("utf-8")).hexdigest()
    return digest[:32]


def map_web_url_capture_error(exc: WebContentCaptureError) -> str:
    return exc.code.value


def http_status_for_web_url_error(error_code: str) -> int:
    if error_code in {
        "idempotency_key_required",
        "tenant_workspace_idempotency_required",
        "web_url_invalid",
        "web_url_scheme_not_allowed",
        "web_url_credentials_not_allowed",
        "web_url_port_not_allowed",
        "web_url_host_not_allowed",
        "web_url_non_global_address_blocked",
    }:
        return 400
    if error_code in {
        "web_url_idempotency_conflict",
        "web_url_already_registered",
        "web_url_state_conflict",
    }:
        return 409
    if error_code == "not_found":
        return 404
    return 503


class WebUrlTextMaterializer:
    def __init__(self, staging_root: Path) -> None:
        self._staging_root = staging_root.resolve()

    @contextmanager
    def materialize(
        self,
        *,
        input_id: str,
        operation_id: str,
        normalized_text: str,
    ) -> Iterator[Path]:
        op_dir = (self._staging_root / _staging_dir_name(input_id=input_id, operation_id=operation_id)).resolve()
        try:
            op_dir.relative_to(self._staging_root)
        except ValueError as exc:
            raise KnowledgeIngestionProcessorError(
                "web_url_materialization_failed"
            ) from exc

        target = (op_dir / _STAGING_FILE_NAME).resolve()
        try:
            target.relative_to(op_dir)
        except ValueError as exc:
            raise KnowledgeIngestionProcessorError(
                "web_url_materialization_failed"
            ) from exc

        try:
            op_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(normalized_text, encoding="utf-8")
            yield target
        except KnowledgeIngestionProcessorError:
            raise
        except Exception as exc:  # noqa: BLE001 - map to stable code
            raise KnowledgeIngestionProcessorError(
                "web_url_materialization_failed"
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


class WebUrlIntakeService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        knowledge_intake: KnowledgeIntakeService,
        url_policy: WebUrlAccessPolicy,
        *,
        preflight_timeout_seconds: float = 10.0,
    ) -> None:
        if preflight_timeout_seconds <= 0:
            raise ValueError("web_url_preflight_timeout_invalid")
        self._repository = repository
        self._knowledge_intake = knowledge_intake
        self._url_policy = url_policy
        self._preflight_timeout_seconds = preflight_timeout_seconds

    async def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        raw_url: str,
        idempotency_key: str,
    ) -> WebUrlAcceptance:
        tenant_id = tenant_id.strip()
        workspace_id = workspace_id.strip()
        idempotency_key = idempotency_key.strip()
        if not tenant_id or not workspace_id or not idempotency_key:
            raise WebUrlValidationError("tenant_workspace_idempotency_required")

        workspace = self._repository.get_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise LookupError("workspace_not_found")

        try:
            canonical = self._url_policy.canonicalize(raw_url)
        except WebContentCaptureError as exc:
            raise WebUrlValidationError(map_web_url_capture_error(exc)) from exc

        try:
            await asyncio.wait_for(
                self._url_policy.approve_target(canonical),
                timeout=self._preflight_timeout_seconds,
            )
        except TimeoutError as exc:
            raise WebUrlValidationError("web_url_timeout") from exc
        except WebContentCaptureError as exc:
            code = map_web_url_capture_error(exc)
            if code in {item.value for item in _BAD_REQUEST_CAPTURE_CODES}:
                raise WebUrlValidationError(code) from exc
            if code in {item.value for item in _SERVICE_UNAVAILABLE_CAPTURE_CODES}:
                raise WebUrlValidationError(code) from exc
            raise WebUrlValidationError(code) from exc

        input_id = deterministic_knowledge_input_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key,
        )
        metadata = {_METADATA_KEY: canonical.fingerprint}

        existing_input = self._repository.get_knowledge_input(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=input_id,
        )
        if existing_input is not None:
            if existing_input.input_kind is not KnowledgeInputKind.WEB_URL:
                raise WebUrlIdempotencyConflict()
            stored_fingerprint = existing_input.submission_metadata.get(_METADATA_KEY, "")
            if stored_fingerprint != canonical.fingerprint:
                raise WebUrlIdempotencyConflict()
            return await self._resume_intake(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                idempotency_key=idempotency_key,
                safe_display_url=canonical.safe_display_url,
            )

        existing_locator = self._repository.get_web_url_locator(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            requested_url_fingerprint=canonical.fingerprint,
        )
        if existing_locator is not None and existing_locator.input_id != input_id:
            raise WebUrlAlreadyRegistered()

        now = _utc_now()
        locator = WebUrlSourceLocator(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=input_id,
            canonical_private_url=canonical.canonical_private_url,
            requested_url_fingerprint=canonical.fingerprint,
            safe_display_url=canonical.safe_display_url,
            created_at=now,
            updated_at=now,
        )
        if existing_locator is None:
            self._repository.put_web_url_locator(locator)
        else:
            self._assert_locator_match(existing_locator, locator)

        try:
            acceptance = self._knowledge_intake.accept(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_kind=KnowledgeInputKind.WEB_URL,
                idempotency_key=idempotency_key,
                submission_metadata=metadata,
            )
        except KnowledgeInputIdempotencyConflict as exc:
            raise WebUrlIdempotencyConflict() from exc
        except KnowledgeInputResolutionError as exc:
            code = str(exc).strip() or "web_url_state_conflict"
            raise WebUrlStateConflict(code) from exc
        except KnowledgeIntakeDispatchError as exc:
            raise KnowledgeIntakeDispatchError("web_url_dispatch_failed") from exc

        return WebUrlAcceptance(
            input_id=acceptance.knowledge_input.input_id,
            workspace_id=workspace_id,
            source_id=acceptance.source.source_id,
            operation_id=acceptance.operation.operation_id,
            status=self._public_status(acceptance.operation),
            safe_display_url=canonical.safe_display_url,
        )

    async def _resume_intake(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        safe_display_url: str,
    ) -> WebUrlAcceptance:
        stored_fingerprint = self._repository.get_knowledge_input(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=deterministic_knowledge_input_id(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                idempotency_key=idempotency_key,
            ),
        )
        if stored_fingerprint is None:
            raise WebUrlStateConflict()
        fingerprint = stored_fingerprint.submission_metadata.get(_METADATA_KEY, "")
        try:
            acceptance = self._knowledge_intake.accept(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_kind=KnowledgeInputKind.WEB_URL,
                idempotency_key=idempotency_key,
                submission_metadata={_METADATA_KEY: fingerprint},
            )
        except KnowledgeInputIdempotencyConflict as exc:
            raise WebUrlIdempotencyConflict() from exc
        except KnowledgeInputResolutionError as exc:
            code = str(exc).strip() or "web_url_state_conflict"
            raise WebUrlStateConflict(code) from exc
        except KnowledgeIntakeDispatchError as exc:
            raise KnowledgeIntakeDispatchError("web_url_dispatch_failed") from exc
        return WebUrlAcceptance(
            input_id=acceptance.knowledge_input.input_id,
            workspace_id=workspace_id,
            source_id=acceptance.source.source_id,
            operation_id=acceptance.operation.operation_id,
            status=self._public_status(acceptance.operation),
            safe_display_url=safe_display_url,
        )

    @staticmethod
    def _assert_locator_match(
        existing: WebUrlSourceLocator,
        expected: WebUrlSourceLocator,
    ) -> None:
        if (
            existing.tenant_id != expected.tenant_id
            or existing.workspace_id != expected.workspace_id
            or existing.input_id != expected.input_id
            or existing.requested_url_fingerprint != expected.requested_url_fingerprint
            or existing.canonical_private_url != expected.canonical_private_url
            or existing.safe_display_url != expected.safe_display_url
        ):
            raise WebUrlStateConflict()

    @staticmethod
    def _public_status(operation: WorkspaceOperation) -> str:
        value = operation.status.value
        if value == "running":
            return "processing"
        if value in {"accepted", "queued", "processing", "completed", "failed"}:
            return value
        return "accepted"


class WebUrlSourceResolver:
    def __init__(self, repository: ManagedWorkspaceRepository) -> None:
        self._repository = repository

    def resolve(
        self,
        *,
        knowledge_input: KnowledgeInput,
        suggested_source_id: str,
    ) -> WorkspaceSource:
        if knowledge_input.input_kind is not KnowledgeInputKind.WEB_URL:
            raise KnowledgeInputResolutionError("web_url_kind_required")

        metadata = knowledge_input.submission_metadata
        if set(metadata.keys()) != {_METADATA_KEY}:
            raise KnowledgeInputResolutionError("web_url_metadata_invalid")
        fingerprint = metadata.get(_METADATA_KEY, "")
        if _FINGERPRINT_RE.fullmatch(fingerprint) is None:
            raise KnowledgeInputResolutionError("web_url_fingerprint_invalid")

        locator = self._repository.get_web_url_locator(
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            requested_url_fingerprint=fingerprint,
        )
        if locator is None:
            raise KnowledgeInputResolutionError("web_url_locator_missing")
        if (
            locator.tenant_id != knowledge_input.tenant_id
            or locator.workspace_id != knowledge_input.workspace_id
            or locator.input_id != knowledge_input.input_id
            or locator.requested_url_fingerprint != fingerprint
            or not locator.canonical_private_url
        ):
            raise KnowledgeInputResolutionError("web_url_locator_mismatch")

        expected_source_id = suggested_source_id
        if knowledge_input.source_id is not None and knowledge_input.source_id != expected_source_id:
            raise KnowledgeInputResolutionError("web_url_source_conflict")

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
            return existing

        now = _utc_now()
        return WorkspaceSource(
            source_id=expected_source_id,
            workspace_id=knowledge_input.workspace_id,
            tenant_id=knowledge_input.tenant_id,
            source_type=WorkspaceSourceType.WEB_RESOURCE,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=now,
        )

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
            or source.source_type is not WorkspaceSourceType.WEB_RESOURCE
            or source.path != ""
            or source.recursive
        ):
            raise KnowledgeInputResolutionError("web_url_source_conflict")


class WebUrlKnowledgeIngestionProcessor:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        web_content_capture: WebContentCapture,
        indexing_service: WorkspaceDocumentIndexingService,
        materializer: WebUrlTextMaterializer,
    ) -> None:
        self._repository = repository
        self._web_content_capture = web_content_capture
        self._indexing_service = indexing_service
        self._materializer = materializer

    async def process(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
        operation: WorkspaceOperation,
    ) -> KnowledgeIngestionResult:
        locator = self._validate_state(
            knowledge_input=knowledge_input,
            source=source,
            operation=operation,
        )

        try:
            captured = await self._web_content_capture.capture(
                WebContentCaptureRequest(url=locator.canonical_private_url)
            )
        except WebContentCaptureError as exc:
            raise KnowledgeIngestionProcessorError(map_web_url_capture_error(exc)) from exc

        now = _utc_now()
        self._repository.put_web_url_locator(
            locator.model_copy(
                update={
                    "final_url_fingerprint": captured.final_url_fingerprint,
                    "final_safe_display_url": captured.safe_display_url,
                    "final_host_changed": captured.final_host_changed,
                    "content_hash": captured.content_hash,
                    "last_captured_at": captured.fetched_at,
                    "updated_at": now,
                }
            )
        )

        try:
            with self._materializer.materialize(
                input_id=knowledge_input.input_id,
                operation_id=operation.operation_id,
                normalized_text=captured.text,
            ) as physical_path:
                logical_source_path = f"web/{source.source_id}/content.txt"
                try:
                    index_result = await self._indexing_service.index_one(
                        tenant_id=knowledge_input.tenant_id,
                        workspace_id=knowledge_input.workspace_id,
                        source_id=source.source_id,
                        operation_id=operation.operation_id,
                        physical_path=physical_path,
                        logical_source_path=logical_source_path,
                        safe_file_name=captured.safe_display_url,
                        content_hash=captured.content_hash,
                    )
                except WorkspaceDocumentIndexingError as exc:
                    raise KnowledgeIngestionProcessorError(
                        "web_url_indexing_failed"
                    ) from exc
        except KnowledgeIngestionProcessorError:
            raise
        except Exception as exc:  # noqa: BLE001 - map to stable code
            raise KnowledgeIngestionProcessorError(
                "web_url_materialization_failed"
            ) from exc

        if index_result.unchanged:
            return KnowledgeIngestionResult(
                files_discovered=1,
                files_processed=1,
                files_failed=0,
                documents_indexed=0,
                documents_unchanged=1,
            )
        return KnowledgeIngestionResult(
            files_discovered=1,
            files_processed=1,
            files_failed=0,
            documents_indexed=index_result.documents_indexed,
            documents_unchanged=0,
        )

    def _validate_state(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
        operation: WorkspaceOperation,
    ) -> WebUrlSourceLocator:
        if knowledge_input.input_kind is not KnowledgeInputKind.WEB_URL:
            raise KnowledgeIngestionProcessorError("web_url_kind_required")
        if source.source_type is not WorkspaceSourceType.WEB_RESOURCE:
            raise KnowledgeIngestionProcessorError("web_url_source_conflict")
        if (
            knowledge_input.tenant_id != source.tenant_id
            or knowledge_input.workspace_id != source.workspace_id
            or knowledge_input.tenant_id != operation.tenant_id
            or knowledge_input.workspace_id != operation.workspace_id
            or operation.source_id != source.source_id
            or operation.input_id != knowledge_input.input_id
            or source.path != ""
            or source.recursive
        ):
            raise KnowledgeIngestionProcessorError("web_url_state_conflict")

        metadata = knowledge_input.submission_metadata
        fingerprint = metadata.get(_METADATA_KEY, "")
        if set(metadata.keys()) != {_METADATA_KEY} or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
            raise KnowledgeIngestionProcessorError("web_url_metadata_invalid")

        locator = self._repository.get_web_url_locator(
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            requested_url_fingerprint=fingerprint,
        )
        if locator is None:
            raise KnowledgeIngestionProcessorError("web_url_locator_missing")
        if (
            locator.input_id != knowledge_input.input_id
            or locator.requested_url_fingerprint != fingerprint
            or not locator.canonical_private_url
        ):
            raise KnowledgeIngestionProcessorError("web_url_locator_mismatch")
        return locator
