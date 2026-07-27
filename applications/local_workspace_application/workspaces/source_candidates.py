# © Artur Czarnecki. All rights reserved.

"""Preconfigured local-folder Source Candidates — registry, intake, resolver, processor."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, NoReturn

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
from local_workspace_application.workspaces.local_folder_indexing import LocalFolderIndexingService
from local_workspace_application.workspaces.models import (
    KnowledgeInput,
    KnowledgeInputKind,
    KnowledgeInputStatus,
    WorkspaceOperation,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.path_policy import (
    SourcePathPolicyError,
    validate_local_folder_source_path,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "lkw.source_candidates.v1"
_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_URL_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_ABS_RE = re.compile(r"[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"\\\\")
# Absolute POSIX path at start or after non-alphanumeric separator (fail-closed).
# Slash between alphanumerics (e.g. Version 1/2, AC/DC) is not treated as a path.
_UNIX_ABS_RE = re.compile(r"(?:^|(?<![A-Za-z0-9]))/(?:[\w.-]+(?:/[\w.-]+)+|[\w.-]+)")
_ALLOWED_METADATA_KEYS = frozenset({"candidate_id", "candidate_fingerprint"})
_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourceCandidateRegistryError(RuntimeError):
    def __init__(self, error_code: str = "source_candidate_configuration_invalid") -> None:
        code = (error_code or "").strip() or "source_candidate_configuration_invalid"
        self.error_code = code
        super().__init__(code)


class SourceCandidateUnavailable(RuntimeError):
    def __init__(self, message: str = "source_candidate_unavailable") -> None:
        super().__init__(message)


class SourceCandidateAlreadyRegistered(RuntimeError):
    def __init__(self, message: str = "source_candidate_already_registered") -> None:
        super().__init__(message)


class SourceCandidateIdempotencyConflict(RuntimeError):
    def __init__(self, message: str = "source_candidate_idempotency_conflict") -> None:
        super().__init__(message)


@dataclass(frozen=True)
class ConfiguredSourceCandidate:
    candidate_id: str
    tenant_id: str
    label: str
    description: str
    source_type: str
    path: str
    recursive: bool
    enabled: bool

    def fingerprint(self) -> str:
        return _candidate_fingerprint(
            tenant_id=self.tenant_id,
            candidate_id=self.candidate_id,
            source_type=self.source_type,
            path=self.path,
            recursive=self.recursive,
            enabled=self.enabled,
        )


@dataclass(frozen=True)
class SourceCandidateConfiguration:
    schema_version: str
    candidates: tuple[ConfiguredSourceCandidate, ...]


@dataclass(frozen=True)
class SourceCandidateSummary:
    candidate_id: str
    label: str
    description: str
    source_type: str
    available: bool


@dataclass(frozen=True)
class SourceCandidateAcceptance:
    candidate_id: str
    label: str
    workspace_id: str
    source_id: str
    operation_id: str
    status: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _collapse_whitespace(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip())


def _reject_public_text(value: str, *, field_name: str, max_length: int) -> str:
    cleaned = _collapse_whitespace(value)
    if not cleaned:
        raise ValueError(f"{field_name}_required")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name}_too_long")
    if _CONTROL_RE.search(cleaned):
        raise ValueError(f"{field_name}_unsafe")
    if _URL_SCHEME_RE.search(cleaned):
        raise ValueError(f"{field_name}_unsafe")
    if _WINDOWS_ABS_RE.search(cleaned):
        raise ValueError(f"{field_name}_unsafe")
    if _UNC_RE.search(cleaned):
        raise ValueError(f"{field_name}_unsafe")
    if _UNIX_ABS_RE.search(cleaned):
        raise ValueError(f"{field_name}_unsafe")
    return cleaned


def _candidate_fingerprint(
    *,
    tenant_id: str,
    candidate_id: str,
    source_type: str,
    path: str,
    recursive: bool,
    enabled: bool,
) -> str:
    payload = {
        "version": 1,
        "tenant_id": tenant_id,
        "candidate_id": candidate_id,
        "source_type": source_type,
        "path": path,
        "recursive": recursive,
        "enabled": enabled,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _parse_candidate(raw: Mapping[str, Any]) -> ConfiguredSourceCandidate:
    if not isinstance(raw, dict):
        raise ValueError("candidate_must_be_object")
    allowed = {
        "candidate_id",
        "tenant_id",
        "label",
        "description",
        "source_type",
        "path",
        "recursive",
        "enabled",
    }
    extra = set(raw.keys()) - allowed
    if extra:
        raise ValueError("candidate_extra_fields")

    candidate_id = raw.get("candidate_id")
    if not isinstance(candidate_id, str):
        raise ValueError("candidate_id_required")
    candidate_id = candidate_id.strip()
    if not candidate_id or len(candidate_id) > 128:
        raise ValueError("candidate_id_invalid")
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise ValueError("candidate_id_invalid")

    tenant_id = raw.get("tenant_id")
    if not isinstance(tenant_id, str):
        raise ValueError("tenant_id_required")
    tenant_id = tenant_id.strip()
    if not tenant_id or len(tenant_id) > 128:
        raise ValueError("tenant_id_invalid")

    label_raw = raw.get("label")
    if not isinstance(label_raw, str):
        raise ValueError("label_required")
    label = _reject_public_text(label_raw, field_name="label", max_length=120)

    description_raw = raw.get("description", "")
    if description_raw is None:
        description = ""
    elif not isinstance(description_raw, str):
        raise ValueError("description_invalid")
    else:
        description = _collapse_whitespace(description_raw)
        if description:
            description = _reject_public_text(
                description,
                field_name="description",
                max_length=500,
            )

    source_type = raw.get("source_type")
    if not isinstance(source_type, str) or source_type.strip() != "local_folder":
        raise ValueError("source_type_unsupported")

    path = raw.get("path")
    if not isinstance(path, str):
        raise ValueError("path_required")
    path = path.strip()
    if not path or len(path) > 4096:
        raise ValueError("path_invalid")

    recursive = raw.get("recursive")
    if not isinstance(recursive, bool):
        raise ValueError("recursive_must_be_bool")

    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("enabled_must_be_bool")

    return ConfiguredSourceCandidate(
        candidate_id=candidate_id,
        tenant_id=tenant_id,
        label=label,
        description=description,
        source_type="local_folder",
        path=path,
        recursive=recursive,
        enabled=enabled,
    )


def _parse_configuration(raw: Any) -> SourceCandidateConfiguration:
    if not isinstance(raw, dict):
        raise ValueError("root_must_be_object")
    allowed = {"schema_version", "candidates"}
    if set(raw.keys()) - allowed:
        raise ValueError("root_extra_fields")
    schema_version = raw.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version_invalid")
    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValueError("candidates_must_be_list")

    parsed: list[ConfiguredSourceCandidate] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates_raw:
        candidate = _parse_candidate(item)
        key = (candidate.tenant_id, candidate.candidate_id)
        if key in seen:
            raise ValueError("candidate_identity_duplicate")
        seen.add(key)
        parsed.append(candidate)
    return SourceCandidateConfiguration(
        schema_version=_SCHEMA_VERSION,
        candidates=tuple(parsed),
    )


class SourceCandidateRegistry:
    """Read-only, instance-local, tenant-scoped Source Candidate registry."""

    def __init__(
        self,
        *,
        available: bool,
        by_identity: Mapping[tuple[str, str], ConfiguredSourceCandidate] | None = None,
    ) -> None:
        self._available = available
        mapping = dict(by_identity or {})
        self._by_identity: dict[tuple[str, str], ConfiguredSourceCandidate] = mapping

    @classmethod
    def empty(cls) -> SourceCandidateRegistry:
        return cls(available=True, by_identity={})

    @classmethod
    def unavailable(cls) -> SourceCandidateRegistry:
        return cls(available=False, by_identity={})

    @classmethod
    def load(cls, path: str | Path) -> SourceCandidateRegistry:
        config_path = Path(path)
        if not config_path.exists():
            return cls.empty()
        try:
            text = config_path.read_text(encoding="utf-8")
            raw = json.loads(text)
            configuration = _parse_configuration(raw)
        except Exception as exc:  # noqa: BLE001 - degrade candidate capability only
            logger.warning(
                "source_candidate_configuration_invalid exception_class=%s",
                type(exc).__name__,
            )
            return cls.unavailable()

        by_identity = {
            (item.tenant_id, item.candidate_id): item for item in configuration.candidates
        }
        return cls(available=True, by_identity=by_identity)

    @property
    def is_available(self) -> bool:
        return self._available

    def _require_available(self) -> None:
        if not self._available:
            raise SourceCandidateRegistryError("source_candidate_configuration_invalid")

    def list_for_tenant(self, tenant_id: str) -> tuple[ConfiguredSourceCandidate, ...]:
        self._require_available()
        tenant = (tenant_id or "").strip()
        if not tenant:
            return ()
        items = [
            candidate
            for (item_tenant, _), candidate in self._by_identity.items()
            if item_tenant == tenant and candidate.enabled
        ]
        items.sort(key=lambda item: (item.label.casefold(), item.candidate_id))
        return tuple(items)

    def get(self, tenant_id: str, candidate_id: str) -> ConfiguredSourceCandidate | None:
        self._require_available()
        tenant = (tenant_id or "").strip()
        candidate = (candidate_id or "").strip()
        if not tenant or not candidate:
            return None
        return self._by_identity.get((tenant, candidate))


class SourceCandidateIntakeService:
    """Product facade for listing and accepting preconfigured Source Candidates."""

    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        registry: SourceCandidateRegistry,
        knowledge_intake: KnowledgeIntakeService,
        *,
        allowlist_roots: frozenset[str] | None = None,
        shadow_roots: tuple[Path, ...] = (),
    ) -> None:
        self._repository = repository
        self._registry = registry
        self._knowledge_intake = knowledge_intake
        self._allowlist_roots = allowlist_roots
        self._shadow_roots = shadow_roots

    def list_candidates(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> tuple[SourceCandidateSummary, ...]:
        tenant_id = tenant_id.strip()
        workspace_id = workspace_id.strip()
        if not tenant_id or not workspace_id:
            raise ValueError("tenant_workspace_required")
        workspace = self._repository.get_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise LookupError("workspace_not_found")

        summaries: list[SourceCandidateSummary] = []
        for candidate in self._registry.list_for_tenant(tenant_id):
            summaries.append(
                SourceCandidateSummary(
                    candidate_id=candidate.candidate_id,
                    label=candidate.label,
                    description=candidate.description,
                    source_type=candidate.source_type,
                    available=self._is_available(candidate),
                )
            )
        return tuple(summaries)

    def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        candidate_id: str,
        idempotency_key: str,
    ) -> SourceCandidateAcceptance:
        tenant_id = tenant_id.strip()
        workspace_id = workspace_id.strip()
        candidate_id = candidate_id.strip()
        idempotency_key = idempotency_key.strip()
        if not tenant_id or not workspace_id or not candidate_id or not idempotency_key:
            raise ValueError("tenant_workspace_candidate_idempotency_required")

        workspace = self._repository.get_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise LookupError("workspace_not_found")

        input_id = deterministic_knowledge_input_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key,
        )
        existing = self._repository.get_knowledge_input(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=input_id,
        )
        if existing is not None:
            if existing.input_kind is not KnowledgeInputKind.SOURCE_CANDIDATE:
                raise SourceCandidateIdempotencyConflict()
            stored_candidate_id = existing.submission_metadata.get("candidate_id", "")
            if stored_candidate_id != candidate_id:
                raise SourceCandidateIdempotencyConflict()
            fingerprint = existing.submission_metadata.get("candidate_fingerprint", "")
            if not fingerprint or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
                raise SourceCandidateIdempotencyConflict()
            label = self._label_for_response(
                tenant_id=tenant_id,
                candidate_id=candidate_id,
            )
            try:
                acceptance = self._knowledge_intake.accept(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    input_kind=KnowledgeInputKind.SOURCE_CANDIDATE,
                    idempotency_key=idempotency_key,
                    submission_metadata={
                        "candidate_id": candidate_id,
                        "candidate_fingerprint": fingerprint,
                    },
                )
            except KnowledgeInputIdempotencyConflict as exc:
                raise SourceCandidateIdempotencyConflict() from exc
            except KnowledgeInputResolutionError as exc:
                self._raise_resolution_error(exc)
            except KnowledgeIntakeDispatchError as exc:
                raise KnowledgeIntakeDispatchError("source_candidate_dispatch_failed") from exc
            return SourceCandidateAcceptance(
                candidate_id=candidate_id,
                label=label,
                workspace_id=workspace_id,
                source_id=acceptance.source.source_id,
                operation_id=acceptance.operation.operation_id,
                status=self._public_status(acceptance.operation),
            )

        candidate = self._registry.get(tenant_id, candidate_id)
        if candidate is None or not candidate.enabled:
            raise LookupError("candidate_not_found")
        if not self._is_available(candidate):
            raise SourceCandidateUnavailable()

        fingerprint = candidate.fingerprint()
        self._reject_if_already_registered(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            candidate_id=candidate_id,
        )

        try:
            acceptance = self._knowledge_intake.accept(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_kind=KnowledgeInputKind.SOURCE_CANDIDATE,
                idempotency_key=idempotency_key,
                submission_metadata={
                    "candidate_id": candidate.candidate_id,
                    "candidate_fingerprint": fingerprint,
                },
            )
        except KnowledgeInputIdempotencyConflict as exc:
            raise SourceCandidateIdempotencyConflict() from exc
        except KnowledgeInputResolutionError as exc:
            self._raise_resolution_error(exc)
        except KnowledgeIntakeDispatchError as exc:
            raise KnowledgeIntakeDispatchError("source_candidate_dispatch_failed") from exc

        return SourceCandidateAcceptance(
            candidate_id=candidate.candidate_id,
            label=candidate.label,
            workspace_id=workspace_id,
            source_id=acceptance.source.source_id,
            operation_id=acceptance.operation.operation_id,
            status=self._public_status(acceptance.operation),
        )

    @staticmethod
    def _raise_resolution_error(exc: KnowledgeInputResolutionError) -> NoReturn:
        code = str(exc).strip()
        if code == "source_candidate_unavailable":
            raise SourceCandidateUnavailable() from exc
        if code in {"source_candidate_not_found", "source_candidate_configuration_changed"}:
            raise LookupError(code) from exc
        if code == "source_candidate_source_conflict":
            raise SourceCandidateAlreadyRegistered() from exc
        raise LookupError("candidate_not_found") from exc

    def _label_for_response(self, *, tenant_id: str, candidate_id: str) -> str:
        try:
            candidate = self._registry.get(tenant_id, candidate_id)
        except SourceCandidateRegistryError:
            return candidate_id
        if candidate is None:
            return candidate_id
        return candidate.label

    def _reject_if_already_registered(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        candidate_id: str,
    ) -> None:
        for item in self._repository.list_knowledge_inputs(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            if item.input_kind is not KnowledgeInputKind.SOURCE_CANDIDATE:
                continue
            if item.status is not KnowledgeInputStatus.RESOLVED:
                continue
            if item.source_id is None:
                continue
            meta = item.submission_metadata
            if meta.get("candidate_id") != candidate_id:
                continue
            source = self._repository.get_source(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=item.source_id,
            )
            if source is None:
                continue
            raise SourceCandidateAlreadyRegistered()

    def _is_available(self, candidate: ConfiguredSourceCandidate) -> bool:
        try:
            validate_local_folder_source_path(
                candidate.path,
                allowlist_roots=self._allowlist_roots,
                shadow_roots=self._shadow_roots,
            )
        except SourcePathPolicyError:
            return False
        except Exception:  # noqa: BLE001 - treat as unavailable without leaking reason
            return False
        return True

    @staticmethod
    def _public_status(operation: WorkspaceOperation) -> str:
        value = operation.status.value
        if value == "running":
            return "processing"
        if value in {"accepted", "queued", "processing", "completed", "failed"}:
            return value
        return "accepted"


class SourceCandidateSourceResolver:
    """Resolves SOURCE_CANDIDATE Knowledge Inputs to durable LOCAL_FOLDER Sources."""

    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        registry: SourceCandidateRegistry,
        *,
        allowlist_roots: frozenset[str] | None = None,
        shadow_roots: tuple[Path, ...] = (),
    ) -> None:
        self._repository = repository
        self._registry = registry
        self._allowlist_roots = allowlist_roots
        self._shadow_roots = shadow_roots

    def resolve(
        self,
        *,
        knowledge_input: KnowledgeInput,
        suggested_source_id: str,
    ) -> WorkspaceSource:
        if knowledge_input.source_id:
            existing = self._repository.get_source(
                tenant_id=knowledge_input.tenant_id,
                workspace_id=knowledge_input.workspace_id,
                source_id=knowledge_input.source_id,
            )
            if existing is not None:
                self._validate_durable_source(
                    existing,
                    knowledge_input=knowledge_input,
                )
                return existing

        if knowledge_input.input_kind is not KnowledgeInputKind.SOURCE_CANDIDATE:
            raise KnowledgeInputResolutionError("source_candidate_kind_required")

        meta = knowledge_input.submission_metadata
        if set(meta.keys()) != _ALLOWED_METADATA_KEYS:
            raise KnowledgeInputResolutionError("source_candidate_metadata_invalid")
        candidate_id = meta.get("candidate_id", "").strip()
        fingerprint = meta.get("candidate_fingerprint", "").strip()
        if not candidate_id or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
            raise KnowledgeInputResolutionError("source_candidate_metadata_invalid")

        try:
            candidate = self._registry.get(knowledge_input.tenant_id, candidate_id)
        except SourceCandidateRegistryError as exc:
            raise KnowledgeInputResolutionError("source_candidate_not_found") from exc
        if candidate is None or not candidate.enabled:
            raise KnowledgeInputResolutionError("source_candidate_not_found")
        if candidate.fingerprint() != fingerprint:
            raise KnowledgeInputResolutionError("source_candidate_configuration_changed")

        try:
            resolved_path = validate_local_folder_source_path(
                candidate.path,
                allowlist_roots=self._allowlist_roots,
                shadow_roots=self._shadow_roots,
            )
        except SourcePathPolicyError as exc:
            raise KnowledgeInputResolutionError("source_candidate_unavailable") from exc

        existing = self._repository.get_source(
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            source_id=suggested_source_id,
        )
        if existing is not None:
            self._validate_durable_source(existing, knowledge_input=knowledge_input)
            return existing

        return WorkspaceSource(
            source_id=suggested_source_id,
            workspace_id=knowledge_input.workspace_id,
            tenant_id=knowledge_input.tenant_id,
            source_type=WorkspaceSourceType.LOCAL_FOLDER,
            path=str(resolved_path),
            recursive=candidate.recursive,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=_utc_now(),
            last_sync_at=None,
        )

    def _validate_durable_source(
        self,
        source: WorkspaceSource,
        *,
        knowledge_input: KnowledgeInput,
    ) -> None:
        if (
            source.tenant_id != knowledge_input.tenant_id
            or source.workspace_id != knowledge_input.workspace_id
            or source.source_type is not WorkspaceSourceType.LOCAL_FOLDER
            or not str(source.path).strip()
            or not isinstance(source.recursive, bool)
        ):
            raise KnowledgeInputResolutionError("source_candidate_source_conflict")


class SourceCandidateKnowledgeIngestionProcessor:
    """First-processing processor for SOURCE_CANDIDATE → LOCAL_FOLDER Sources."""

    def __init__(self, folder_indexing: LocalFolderIndexingService) -> None:
        self._folder_indexing = folder_indexing

    async def process(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
        operation: WorkspaceOperation,
    ) -> KnowledgeIngestionResult:
        if knowledge_input.input_kind is not KnowledgeInputKind.SOURCE_CANDIDATE:
            raise KnowledgeIngestionProcessorError("source_candidate_kind_required")
        if source.source_type is not WorkspaceSourceType.LOCAL_FOLDER:
            raise KnowledgeIngestionProcessorError("source_candidate_source_conflict")
        if (
            knowledge_input.tenant_id != source.tenant_id
            or knowledge_input.tenant_id != operation.tenant_id
            or knowledge_input.workspace_id != source.workspace_id
            or knowledge_input.workspace_id != operation.workspace_id
            or operation.source_id != source.source_id
        ):
            raise KnowledgeIngestionProcessorError("source_candidate_source_conflict")

        try:
            result = await self._folder_indexing.index_source(
                tenant_id=knowledge_input.tenant_id,
                workspace_id=knowledge_input.workspace_id,
                source=source,
                operation_id=operation.operation_id,
            )
        except KnowledgeIngestionProcessorError:
            raise
        except Exception as exc:  # noqa: BLE001 - stable product code only
            raise KnowledgeIngestionProcessorError(
                "source_candidate_indexing_failed"
            ) from exc

        if (
            result.files_discovered > 0
            and result.documents_indexed == 0
            and result.documents_unchanged == 0
        ):
            raise KnowledgeIngestionProcessorError(
                "source_candidate_sync_produced_no_documents"
            )

        return KnowledgeIngestionResult(
            files_discovered=result.files_discovered,
            files_processed=result.files_processed,
            files_failed=result.files_failed,
            documents_indexed=result.documents_indexed,
            documents_unchanged=result.documents_unchanged,
        )
