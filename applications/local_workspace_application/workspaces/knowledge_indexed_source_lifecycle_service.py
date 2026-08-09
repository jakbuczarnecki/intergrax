# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Indexed Source lifecycle orchestration."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
    indexed_source_binding_id,
)
from local_workspace_application.workspaces.connected_source_source_projection import (
    ConnectedSourceOriginValidationError,
    validate_connected_source_durable_origin,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    CreateIndexedSourceMutationIntent,
    DisableIndexedSourceMutationIntent,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    normalize_create_indexed_source_request_hash,
    normalize_disable_indexed_source_request_hash,
    semantic_identity_hash_for_create_indexed_source,
    semantic_identity_hash_for_disable_indexed_source,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
    WorkspaceKnowledgeMutationExecutionResult,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_configuration_validation import (
    validate_configuration_idempotency_hash,
)
from local_workspace_application.workspaces.knowledge_materialization_purge import (
    KnowledgeMaterializationPurgeError,
    KnowledgeMaterializationPurgeRequestV1,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncPublicationFenceConflict,
    KnowledgeSyncPublicationFencePort,
    KnowledgeSyncPublicationFenceV1,
    KnowledgeSyncPublicationInProgress,
)

_RESULT_TYPE = "indexed_source_binding"


class IndexedSourceLifecycleStateV1(StrEnum):
    READY = "ready"
    SYNCING = "syncing"
    ACTIVE = "active"
    DISABLED = "disabled"
    DETACHING = "detaching"
    DETACH_BLOCKED = "detach_blocked"
    DETACHED = "detached"
    ERROR = "error"


class IndexedSourceSyncStateV1(StrEnum):
    NEVER_SYNCED = "never_synced"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IndexedSourceLifecycleViewV1(BaseModel):
    """Credential-free projection of the existing lifecycle authorities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    source_id: str
    indexed_source_binding_id: str
    knowledge_source_binding_ref: str
    lifecycle_state: IndexedSourceLifecycleStateV1
    lifecycle_revision: int = Field(ge=0)
    enabled: bool
    detached: bool
    sync_state: IndexedSourceSyncStateV1
    last_delivery_id: str | None = None
    last_successful_sync_at: datetime | None = None
    last_error_code: str | None = None
    purge_state: str | None = None
    updated_at: datetime


class IndexedSourceLifecycleResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    view: IndexedSourceLifecycleViewV1
    operation_id: str | None = None
    mutation_id: str | None = None


@dataclass(frozen=True, slots=True)
class IndexedSourceLifecycleCommand:
    tenant_id: str
    workspace_id: str
    indexed_source_binding_id: str
    expected_revision: int
    idempotency_key_hash: str = "0" * 64


@dataclass(frozen=True, slots=True)
class IndexedSourceSyncCommand:
    tenant_id: str
    workspace_id: str
    indexed_source_binding_id: str
    expected_revision: int
    idempotency_key_hash: str = "0" * 64


@dataclass(frozen=True, slots=True)
class IndexedSourceRetryCommand:
    tenant_id: str
    workspace_id: str
    indexed_source_binding_id: str
    operation_id: str
    expected_revision: int
    idempotency_key_hash: str = "0" * 64


class IndexedSourceSyncRequestPort(Protocol):
    def create_sync_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        allow_concurrent: bool = False,
    ) -> WorkspaceOperation: ...

    def get_operation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> WorkspaceOperation | None: ...


class IndexedSourcePurgePort(Protocol):
    def start_or_resume(
        self,
        request: KnowledgeMaterializationPurgeRequestV1,
    ) -> object: ...

    def get_state(
        self,
        request: KnowledgeMaterializationPurgeRequestV1,
    ) -> object | None: ...


@dataclass(frozen=True, slots=True)
class _PurgeReadFailure:
    status: str = "failed"
    last_error_code: str = "BLOCKED_CORRUPT_STATE"
    updated_at: datetime = dataclass_field(default_factory=lambda: datetime.now(UTC))


class TenantKnowledgeSourceBindingPort(Protocol):
    def get_binding(
        self, *, tenant_id: str, binding_id: str
    ) -> KnowledgeSourceBinding | None: ...

class WorkspaceIndexedSourceLifecycleError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code

@dataclass(frozen=True, slots=True)
class ActivateWorkspaceIndexedSourceCommand:
    tenant_id: str
    workspace_id: str
    knowledge_source_binding_ref: str
    expected_revision: int
    idempotency_key_hash: str
    sync_mode: IndexedSourceSyncModeV1
    audience_eligibility: IndexedSourceAudienceEligibilityV1
    cached_safe_display_label: str | None = None

@dataclass(frozen=True, slots=True)
class DisableWorkspaceIndexedSourceCommand:
    tenant_id: str
    workspace_id: str
    indexed_source_binding_id: str
    expected_revision: int
    idempotency_key_hash: str

@dataclass(frozen=True, slots=True)
class WorkspaceIndexedSourceLifecycleResult:
    binding: WorkspaceIndexedSourceBinding
    configuration_revision: int
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1
    created_new_source: bool
    mutation: WorkspaceKnowledgeMutationRecord

def _incomplete() -> WorkspaceIndexedSourceLifecycleError:
    return WorkspaceIndexedSourceLifecycleError("indexed_source_projection_incomplete")

def _highest_binding(versions: list[WorkspaceIndexedSourceBinding], *, binding_id: str, revision: int):
    matches = [v for v in versions if v.indexed_source_binding_id == binding_id and v.effective_revision <= revision]
    if not matches:
        return None
    top = max(matches, key=lambda item: item.effective_revision)
    if sum(1 for item in matches if item.effective_revision == top.effective_revision) > 1:
        raise _incomplete()
    return top

def _resolve_historical_binding(
    repository: ManagedWorkspaceRepository,
    *,
    result: WorkspaceKnowledgeMutationExecutionResult,
    tenant_id: str,
    workspace_id: str,
    binding_id: str,
    source_id: str,
    request_hash: str,
    semantic_hash: str,
    expected_status: WorkspaceIndexedSourceBindingStatusV1,
) -> WorkspaceIndexedSourceBinding:
    mutation = result.mutation
    if (
        mutation.normalized_request_hash != request_hash
        or mutation.semantic_identity_hash != semantic_hash
        or mutation.result_entity_type != _RESULT_TYPE
        or mutation.result_entity_id != binding_id
        or mutation.committed_revision != result.configuration_revision
    ):
        raise _incomplete()
    binding = _highest_binding(
        repository.list_knowledge_indexed_source_versions(tenant_id=tenant_id, workspace_id=workspace_id),
        binding_id=binding_id,
        revision=result.configuration_revision,
    )
    if binding is None or binding.tenant_id != tenant_id or binding.workspace_id != workspace_id:
        raise _incomplete()
    if binding.indexed_source_binding_id != binding_id or binding.source_id != source_id:
        raise _incomplete()
    if not binding.knowledge_source_binding_ref.strip():
        raise _incomplete()
    if binding.semantic_identity_hash != semantic_hash:
        raise _incomplete()
    expected_binding_id = indexed_source_binding_id(
        tenant_id, workspace_id, binding.knowledge_source_binding_ref,
    )
    expected_source_id = connected_source_id(
        tenant_id, workspace_id, binding.knowledge_source_binding_ref,
    )
    if binding.indexed_source_binding_id != expected_binding_id:
        raise _incomplete()
    if binding.source_id != expected_source_id:
        raise _incomplete()
    if binding.status is not expected_status:
        raise _incomplete()
    if mutation.outcome is WorkspaceKnowledgeMutationOutcomeV1.APPLIED:
        if (
            mutation.target_revision != mutation.committed_revision
            or binding.effective_revision != mutation.target_revision
            or binding.mutation_id != mutation.mutation_id
        ):
            raise _incomplete()
    elif mutation.outcome is WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT:
        if mutation.committed_revision is None:
            raise _incomplete()
        if mutation.target_revision is not None or binding.effective_revision > mutation.committed_revision:
            raise _incomplete()
    else:
        raise _incomplete()
    if expected_status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE:
        try:
            validate_connected_source_durable_origin(
                repository=repository,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                binding=binding,
                committed_configuration_revision=result.configuration_revision,
            )
        except ConnectedSourceOriginValidationError:
            raise _incomplete() from None
    return binding

def _validate_tenant_binding(port: TenantKnowledgeSourceBindingPort, *, tenant_id: str, binding_id: str):
    try:
        binding = port.get_binding(tenant_id=tenant_id, binding_id=binding_id)
    except Exception as exc:
        raise WorkspaceIndexedSourceLifecycleError("knowledge_source_binding_unavailable") from exc
    if binding is None:
        raise WorkspaceIndexedSourceLifecycleError("knowledge_source_binding_not_found")
    for field in ("tenant_id", "binding_id", "connection_ref", "safe_display_name"):
        value = getattr(binding, field, None)
        if not isinstance(value, str) or not value.strip():
            raise WorkspaceIndexedSourceLifecycleError("knowledge_source_binding_invalid")
    if binding.tenant_id != tenant_id or binding.binding_id != binding_id:
        raise WorkspaceIndexedSourceLifecycleError("knowledge_source_binding_invalid")
    if binding.status is not KnowledgeSourceBindingStatus.ACTIVE:
        raise WorkspaceIndexedSourceLifecycleError("knowledge_source_binding_unavailable")
    return binding

def _activation_hashes(command: ActivateWorkspaceIndexedSourceCommand):
    tenant_id, workspace_id = command.tenant_id.strip(), command.workspace_id.strip()
    binding_ref = command.knowledge_source_binding_ref.strip()
    request_hash = normalize_create_indexed_source_request_hash(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        knowledge_source_binding_ref=binding_ref,
        sync_mode=command.sync_mode,
        audience_eligibility=command.audience_eligibility,
    )
    semantic_hash = semantic_identity_hash_for_create_indexed_source(
        tenant_id=tenant_id, workspace_id=workspace_id, knowledge_source_binding_ref=binding_ref
    )
    return tenant_id, workspace_id, binding_ref, request_hash, semantic_hash

def _lifecycle_result(result, *, binding, created_new_source: bool) -> WorkspaceIndexedSourceLifecycleResult:
    return WorkspaceIndexedSourceLifecycleResult(
        binding=binding,
        configuration_revision=result.configuration_revision,
        disposition=result.disposition,
        created_new_source=created_new_source,
        mutation=result.mutation,
    )

class WorkspaceIndexedSourceLifecycleService:
    def __init__(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
        tenant_binding_port: TenantKnowledgeSourceBindingPort,
        publication_fence_port: KnowledgeSyncPublicationFencePort | None = None,
        sync_request_port: IndexedSourceSyncRequestPort | None = None,
        purge_service: IndexedSourcePurgePort | None = None,
    ) -> None:
        self._repository = repository
        self._configuration_service = configuration_service
        self._mutation_engine = mutation_engine
        self._tenant_binding_port = tenant_binding_port
        self._publication_fence_port = publication_fence_port or (
            DocumentStoreKnowledgeSyncPublicationFenceRepository(repository.document_store)
        )
        self._sync_request_port = sync_request_port or ManagedWorkspaceService(repository)
        self._purge_service = purge_service

    def attach(
        self,
        command: ActivateWorkspaceIndexedSourceCommand,
    ) -> IndexedSourceLifecycleResultV1:
        """Attach/replay one semantic Indexed Source binding."""
        result = self.activate_indexed_source(command)
        view = self.get(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            indexed_source_binding_id=result.binding.indexed_source_binding_id,
        )
        return IndexedSourceLifecycleResultV1(
            view=view,
            mutation_id=result.mutation.mutation_id,
        )

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        indexed_source_binding_id: str | None = None,
        source_id: str | None = None,
        knowledge_source_binding_ref: str | None = None,
    ) -> IndexedSourceLifecycleViewV1:
        configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise WorkspaceIndexedSourceLifecycleError("workspace_not_found")
        binding = next(
            (
                candidate
                for candidate in configuration.indexed_sources
                if (
                    indexed_source_binding_id is None
                    or candidate.indexed_source_binding_id == indexed_source_binding_id
                )
                and (source_id is None or candidate.source_id == source_id)
                and (
                    knowledge_source_binding_ref is None
                    or candidate.knowledge_source_binding_ref
                    == knowledge_source_binding_ref
                )
            ),
            None,
        )
        if binding is None:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_not_found")
        fence = self._publication_fence_port.read_fence(
            tenant_id=tenant_id,
            binding_id=binding.knowledge_source_binding_ref,
        )
        if fence is None:
            raise WorkspaceIndexedSourceLifecycleError("publication_fence_missing")

        operations = []
        operation_page = self._repository.list_source_sync_operations_page(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=binding.source_id,
            limit=1,
        )
        for record in operation_page.documents:
            data = dict(record.data)
            if any(
                data.get(field) != expected
                for field, expected in (
                    ("tenant_id", tenant_id),
                    ("workspace_id", workspace_id),
                    ("source_id", binding.source_id),
                    ("operation_type", "source_sync"),
                )
            ):
                continue
            operation_id = data.get("operation_id")
            if not isinstance(operation_id, str) or not operation_id.strip():
                continue
            operation = self._repository.get_operation(
                tenant_id=tenant_id,
                operation_id=operation_id,
            )
            if (
                operation is not None
                and operation.tenant_id == tenant_id
                and operation.workspace_id == workspace_id
                and operation.source_id == binding.source_id
                and operation.operation_type.value == "source_sync"
            ):
                operations.append(operation)
        latest_operation = max(
            operations,
            key=lambda operation: (
                operation.created_at
                or operation.started_at
                or operation.completed_at
                or datetime.min.replace(tzinfo=UTC),
                operation.operation_id,
            ),
            default=None,
        )
        active_operation = (
            latest_operation
            if latest_operation is not None
            and latest_operation.status
            in {
                WorkspaceOperationStatus.ACCEPTED,
                WorkspaceOperationStatus.QUEUED,
                WorkspaceOperationStatus.RUNNING,
                WorkspaceOperationStatus.PROCESSING,
            }
            else None
        )
        head = self._publication_fence_port.read_publication_head(
            tenant_id=tenant_id,
            binding_id=binding.knowledge_source_binding_ref,
        )
        if active_operation is not None:
            sync_state = (
                IndexedSourceSyncStateV1.RUNNING
                if active_operation.status
                in {WorkspaceOperationStatus.RUNNING, WorkspaceOperationStatus.PROCESSING}
                else IndexedSourceSyncStateV1.QUEUED
            )
        elif latest_operation is not None and latest_operation.status is WorkspaceOperationStatus.FAILED:
            sync_state = IndexedSourceSyncStateV1.FAILED
        elif head is not None:
            sync_state = IndexedSourceSyncStateV1.SUCCEEDED
        else:
            sync_state = IndexedSourceSyncStateV1.NEVER_SYNCED

        purge_state = self._read_purge_state(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding=binding,
            lifecycle_revision=fence.lifecycle_revision,
        )
        purge_status = None if purge_state is None else getattr(purge_state, "status", None)
        purge_status_value = (
            None
            if purge_status is None
            else getattr(purge_status, "value", purge_status)
        )
        purge_error = (
            None
            if purge_state is None
            else getattr(purge_state, "last_error_code", None)
        )
        last_error = (
            self._application_purge_error(purge_error)
            if purge_error
            else (
                None
                if latest_operation is None or latest_operation.status is not WorkspaceOperationStatus.FAILED
                else latest_operation.error_code or latest_operation.error
            )
        )
        if purge_status_value == "failed" and purge_error:
            lifecycle_state = IndexedSourceLifecycleStateV1.DETACH_BLOCKED
        elif fence.detached:
            if purge_status_value == "completed":
                lifecycle_state = IndexedSourceLifecycleStateV1.DETACHED
            elif purge_error or purge_status_value == "failed":
                lifecycle_state = IndexedSourceLifecycleStateV1.DETACH_BLOCKED
            else:
                lifecycle_state = IndexedSourceLifecycleStateV1.DETACHING
        elif not fence.enabled:
            lifecycle_state = IndexedSourceLifecycleStateV1.DISABLED
        elif active_operation is not None:
            lifecycle_state = IndexedSourceLifecycleStateV1.SYNCING
        elif head is not None:
            lifecycle_state = IndexedSourceLifecycleStateV1.ACTIVE
        elif last_error is not None and sync_state is IndexedSourceSyncStateV1.FAILED:
            lifecycle_state = IndexedSourceLifecycleStateV1.ERROR
        else:
            lifecycle_state = IndexedSourceLifecycleStateV1.READY

        timestamps = [binding.updated_at]
        if latest_operation is not None:
            timestamps.extend(
                timestamp
                for timestamp in (
                    latest_operation.created_at,
                    latest_operation.started_at,
                    latest_operation.completed_at,
                )
                if timestamp is not None
            )
        if head is not None:
            timestamps.append(head.committed_at)
        if purge_state is not None:
            purge_updated = getattr(purge_state, "updated_at", None)
            if purge_updated is not None:
                timestamps.append(purge_updated)
        return IndexedSourceLifecycleViewV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=binding.source_id,
            indexed_source_binding_id=binding.indexed_source_binding_id,
            knowledge_source_binding_ref=binding.knowledge_source_binding_ref,
            lifecycle_state=lifecycle_state,
            lifecycle_revision=fence.lifecycle_revision,
            enabled=fence.enabled,
            detached=fence.detached,
            sync_state=sync_state,
            last_delivery_id=None if head is None else head.delivery_id,
            last_successful_sync_at=None if head is None else head.committed_at,
            last_error_code=last_error,
            purge_state=purge_status_value,
            updated_at=max(timestamps),
        )

    @staticmethod
    def _application_purge_error(error_code: str | None) -> str | None:
        return {
            "publication_in_progress": "detach_blocked_publication_in_progress",
            "BLOCKED_LEGACY_MIGRATION": "detach_blocked_legacy_migration",
            "BLOCKED_CORRUPT_STATE": "detach_blocked_corrupt_state",
        }.get(error_code or "", error_code)

    def request_sync(
        self,
        command: IndexedSourceSyncCommand,
    ) -> IndexedSourceLifecycleResultV1:
        view = self.get(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            indexed_source_binding_id=command.indexed_source_binding_id,
        )
        self._assert_current_revision(view, command.expected_revision)
        if view.detached:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_detached")
        if not view.enabled:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_disabled")
        try:
            operation = self._sync_request_port.create_sync_operation(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                source_id=view.source_id,
            )
        except Exception as exc:
            if str(exc) == "sync_already_in_progress":
                raise WorkspaceIndexedSourceLifecycleError(
                    "sync_in_progress"
                ) from exc
            raise
        return IndexedSourceLifecycleResultV1(
            view=self.get(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                indexed_source_binding_id=command.indexed_source_binding_id,
            ),
            operation_id=operation.operation_id,
        )

    def retry_sync(
        self,
        command: IndexedSourceRetryCommand,
    ) -> IndexedSourceLifecycleResultV1:
        view = self.get(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            indexed_source_binding_id=command.indexed_source_binding_id,
        )
        self._assert_current_revision(view, command.expected_revision)
        if view.detached:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_detached")
        if not view.enabled:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_disabled")
        previous = self._sync_request_port.get_operation(
            tenant_id=command.tenant_id,
            operation_id=command.operation_id,
        )
        if (
            previous is None
            or previous.workspace_id != command.workspace_id
            or previous.source_id != view.source_id
            or previous.status
            not in {WorkspaceOperationStatus.FAILED, WorkspaceOperationStatus.COMPLETED}
        ):
            raise WorkspaceIndexedSourceLifecycleError("sync_not_retryable")
        if previous.status is WorkspaceOperationStatus.COMPLETED:
            head = self._publication_fence_port.read_publication_head(
                tenant_id=command.tenant_id,
                binding_id=view.knowledge_source_binding_ref,
            )
            if head is not None:
                raise WorkspaceIndexedSourceLifecycleError("sync_not_retryable")
        result = self.request_sync(
            IndexedSourceSyncCommand(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                indexed_source_binding_id=command.indexed_source_binding_id,
                expected_revision=command.expected_revision,
                idempotency_key_hash=command.idempotency_key_hash,
            )
        )
        return result

    def disable(
        self,
        command: IndexedSourceLifecycleCommand,
    ) -> IndexedSourceLifecycleResultV1:
        view = self.get(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            indexed_source_binding_id=command.indexed_source_binding_id,
        )
        if view.detached:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_detached")
        if view.enabled or view.lifecycle_revision < command.expected_revision:
            self._assert_current_revision(view, command.expected_revision)
        self._disable_publication_fence(
            tenant_id=command.tenant_id,
            binding_ref=view.knowledge_source_binding_ref,
            expected_revision=view.lifecycle_revision,
        )
        configuration = self._configuration_service.get_configuration(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
        )
        if configuration is None:
            raise WorkspaceIndexedSourceLifecycleError("workspace_not_found")
        legacy = self.disable_indexed_source(
            DisableWorkspaceIndexedSourceCommand(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                indexed_source_binding_id=command.indexed_source_binding_id,
                expected_revision=configuration.configuration_revision,
                idempotency_key_hash=command.idempotency_key_hash,
            )
        )
        return IndexedSourceLifecycleResultV1(
            view=self.get(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                indexed_source_binding_id=command.indexed_source_binding_id,
            ),
            mutation_id=legacy.mutation.mutation_id,
        )

    def enable(
        self,
        command: IndexedSourceLifecycleCommand,
    ) -> IndexedSourceLifecycleResultV1:
        view = self.get(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            indexed_source_binding_id=command.indexed_source_binding_id,
        )
        self._assert_current_revision(view, command.expected_revision)
        if view.detached:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_detached")
        if view.enabled:
            return IndexedSourceLifecycleResultV1(view=view)
        configuration = self._configuration_service.get_configuration(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
        )
        if configuration is None:
            raise WorkspaceIndexedSourceLifecycleError("workspace_not_found")
        binding_config = next(
            (
                candidate
                for candidate in configuration.indexed_sources
                if candidate.indexed_source_binding_id
                == command.indexed_source_binding_id
            ),
            None,
        )
        if binding_config is None:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_not_found")
        activated = self.activate_indexed_source(
            ActivateWorkspaceIndexedSourceCommand(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                knowledge_source_binding_ref=view.knowledge_source_binding_ref,
                expected_revision=configuration.configuration_revision,
                idempotency_key_hash=command.idempotency_key_hash,
                sync_mode=binding_config.sync_mode,
                audience_eligibility=binding_config.audience_eligibility,
                cached_safe_display_label=binding_config.cached_safe_display_label,
            )
        )
        return IndexedSourceLifecycleResultV1(
            view=self.get(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                indexed_source_binding_id=command.indexed_source_binding_id,
            ),
            mutation_id=activated.mutation.mutation_id,
        )

    def detach(
        self,
        command: IndexedSourceLifecycleCommand,
    ) -> IndexedSourceLifecycleResultV1:
        return self._detach(command)

    def resume_detach(
        self,
        command: IndexedSourceLifecycleCommand,
    ) -> IndexedSourceLifecycleResultV1:
        return self._detach(command)

    def _detach(
        self,
        command: IndexedSourceLifecycleCommand,
    ) -> IndexedSourceLifecycleResultV1:
        if self._purge_service is None:
            raise WorkspaceIndexedSourceLifecycleError("detach_purge_unavailable")
        view = self.get(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            indexed_source_binding_id=command.indexed_source_binding_id,
        )
        self._assert_current_revision(view, command.expected_revision)
        purge_request = KnowledgeMaterializationPurgeRequestV1(
            tenant_id=view.tenant_id,
            workspace_id=view.workspace_id,
            source_id=view.source_id,
            indexed_source_binding_id=view.indexed_source_binding_id,
            knowledge_source_binding_ref=view.knowledge_source_binding_ref,
            requested_lifecycle_revision=view.lifecycle_revision,
            operation_id=f"detach:{view.indexed_source_binding_id}",
        )
        try:
            self._purge_service.start_or_resume(purge_request)
        except KnowledgeSyncPublicationInProgress as exc:
            raise WorkspaceIndexedSourceLifecycleError(
                "DETACH_BLOCKED_PUBLICATION_IN_PROGRESS"
            ) from exc
        except KnowledgeMaterializationPurgeError:
            # Do not recreate or reset a corrupt durable purge.
            pass
        result_view = self.get(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            indexed_source_binding_id=command.indexed_source_binding_id,
        )
        return IndexedSourceLifecycleResultV1(view=result_view)

    @staticmethod
    def _assert_current_revision(
        view: IndexedSourceLifecycleViewV1,
        expected_revision: int,
    ) -> None:
        if view.lifecycle_revision != expected_revision:
            raise WorkspaceIndexedSourceLifecycleError("lifecycle_conflict")

    def _read_purge_state(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        binding: WorkspaceIndexedSourceBinding,
        lifecycle_revision: int,
    ) -> object | None:
        if self._purge_service is None:
            return None
        reader = getattr(self._purge_service, "get_state", None)
        if reader is None:
            return None
        request = KnowledgeMaterializationPurgeRequestV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=binding.source_id,
            indexed_source_binding_id=binding.indexed_source_binding_id,
            knowledge_source_binding_ref=binding.knowledge_source_binding_ref,
            requested_lifecycle_revision=lifecycle_revision,
            operation_id=f"detach:{binding.indexed_source_binding_id}",
        )
        try:
            return reader(request)
        except KnowledgeMaterializationPurgeError:
            return _PurgeReadFailure()

    def _disable_publication_fence(
        self,
        *,
        tenant_id: str,
        binding_ref: str,
        expected_revision: int,
    ) -> None:
        current = self._publication_fence_port.read_fence(
            tenant_id=tenant_id,
            binding_id=binding_ref,
        )
        if current is None:
            raise WorkspaceIndexedSourceLifecycleError("publication_fence_missing")
        self._assert_current_revision(
            IndexedSourceLifecycleViewV1(
                tenant_id=current.tenant_id,
                workspace_id="lifecycle",
                source_id="lifecycle",
                indexed_source_binding_id="lifecycle",
                knowledge_source_binding_ref=current.binding_id,
                lifecycle_state=IndexedSourceLifecycleStateV1.READY,
                lifecycle_revision=current.lifecycle_revision,
                enabled=current.enabled,
                detached=current.detached,
                sync_state=IndexedSourceSyncStateV1.NEVER_SYNCED,
                updated_at=datetime.now(UTC),
            ),
            expected_revision,
        )
        if current.detached:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_detached")
        if not current.enabled:
            return
        try:
            disable = getattr(self._publication_fence_port, "disable", None)
            if disable is not None:
                disable(
                    tenant_id=tenant_id,
                    binding_id=binding_ref,
                    lifecycle_revision=current.lifecycle_revision + 1,
                    lifecycle_token=secrets.token_urlsafe(32),
                    expected_revision=current.lifecycle_revision,
                )
                return
            replacement = current.model_copy(
                update={
                    "lifecycle_revision": current.lifecycle_revision + 1,
                    "lifecycle_token": secrets.token_urlsafe(32),
                    "enabled": False,
                }
            )
            self._publication_fence_port.write_fence(
                replacement,
                expected_revision=current.lifecycle_revision,
            )
        except KnowledgeSyncPublicationInProgress as exc:
            raise WorkspaceIndexedSourceLifecycleError(
                "publication_in_progress"
            ) from exc
        except KnowledgeSyncPublicationFenceConflict as exc:
            reloaded = self._publication_fence_port.read_fence(
                tenant_id=tenant_id,
                binding_id=binding_ref,
            )
            if reloaded is not None and not reloaded.enabled and not reloaded.detached:
                return
            raise WorkspaceIndexedSourceLifecycleError("lifecycle_conflict") from exc

    def _set_publication_fence(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        enabled: bool,
    ) -> None:
        current = self._publication_fence_port.read_fence(
            tenant_id=tenant_id,
            binding_id=binding_id,
        )
        if current is None:
            candidate = KnowledgeSyncPublicationFenceV1(
                tenant_id=tenant_id,
                binding_id=binding_id,
                lifecycle_revision=1,
                lifecycle_token=secrets.token_urlsafe(32),
                enabled=enabled,
                detached=False,
            )
            try:
                self._publication_fence_port.write_fence(
                    candidate,
                    expected_revision=None,
                )
            except Exception:
                if self._publication_fence_port.read_fence(
                    tenant_id=tenant_id,
                    binding_id=binding_id,
                ) is None:
                    raise
            return
        if current.enabled is enabled and not current.detached:
            return
        if current.detached:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_detached")
        replacement = current.model_copy(
            update={
                "lifecycle_revision": current.lifecycle_revision + 1,
                "lifecycle_token": secrets.token_urlsafe(32),
                "enabled": enabled,
                "detached": False,
            }
        )
        try:
            self._publication_fence_port.write_fence(
                replacement,
                expected_revision=current.lifecycle_revision,
            )
        except Exception:
            reloaded = self._publication_fence_port.read_fence(
                tenant_id=tenant_id,
                binding_id=binding_id,
            )
            if reloaded is None or reloaded.enabled is not enabled:
                raise

    def replay_activation_if_committed(
        self, command: ActivateWorkspaceIndexedSourceCommand
    ) -> WorkspaceIndexedSourceLifecycleResult | None:
        validate_configuration_idempotency_hash(command.idempotency_key_hash)
        tenant_id, workspace_id, binding_ref, request_hash, semantic_hash = _activation_hashes(command)
        binding_id = indexed_source_binding_id(tenant_id, workspace_id, binding_ref)
        source_id = connected_source_id(tenant_id, workspace_id, binding_ref)
        existing = self._repository.get_knowledge_configuration_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            idempotency_key_hash=command.idempotency_key_hash,
        )
        if existing is None:
            return None
        if existing.normalized_request_hash != request_hash:
            raise WorkspaceKnowledgeConfigurationMutationError("configuration_idempotency_conflict")
        if existing.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED:
            return None
        result = self._mutation_engine.execute(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            expected_revision=command.expected_revision,
            idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=request_hash,
            semantic_identity_hash=semantic_hash,
            intent=CreateIndexedSourceMutationIntent(
                knowledge_source_binding_ref=binding_ref,
                sync_mode=command.sync_mode,
                audience_eligibility=command.audience_eligibility,
                cached_safe_display_label=command.cached_safe_display_label,
            ),
        )
        if result.disposition is not WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY:
            return None
        binding = _resolve_historical_binding(
            self._repository,
            result=result,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            request_hash=request_hash,
            semantic_hash=semantic_hash,
            expected_status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
        )
        self._set_publication_fence(
            tenant_id=tenant_id,
            binding_id=binding_ref,
            enabled=True,
        )
        return _lifecycle_result(result, binding=binding, created_new_source=False)

    def activate_indexed_source(
        self, command: ActivateWorkspaceIndexedSourceCommand
    ) -> WorkspaceIndexedSourceLifecycleResult:
        replay = self.replay_activation_if_committed(command)
        if replay is not None:
            return replay
        validate_configuration_idempotency_hash(command.idempotency_key_hash)
        tenant_id, workspace_id, binding_ref, request_hash, semantic_hash = _activation_hashes(command)
        configuration = self._configuration_service.get_configuration(tenant_id=tenant_id, workspace_id=workspace_id)
        if configuration is None:
            raise WorkspaceIndexedSourceLifecycleError("workspace_not_found")
        tenant_binding = _validate_tenant_binding(self._tenant_binding_port, tenant_id=tenant_id, binding_id=binding_ref)
        attachments = [a for a in configuration.connection_attachments if a.connection_ref == tenant_binding.connection_ref.strip()]
        if len(attachments) != 1:
            raise WorkspaceIndexedSourceLifecycleError("connection_not_attached")
        if attachments[0].status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
            code = "connection_not_attached" if attachments[0].status is WorkspaceConnectionAttachmentStatusV1.DETACHED else "connection_unavailable"
            raise WorkspaceIndexedSourceLifecycleError(code)
        binding_id = indexed_source_binding_id(tenant_id, workspace_id, binding_ref)
        source_id = connected_source_id(tenant_id, workspace_id, binding_ref)
        had_source = self._repository.get_source(tenant_id=tenant_id, workspace_id=workspace_id, source_id=source_id) is not None
        result = self._mutation_engine.execute(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            expected_revision=command.expected_revision,
            idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=request_hash,
            semantic_identity_hash=semantic_hash,
            intent=CreateIndexedSourceMutationIntent(
                knowledge_source_binding_ref=binding_ref,
                sync_mode=command.sync_mode,
                audience_eligibility=command.audience_eligibility,
                cached_safe_display_label=tenant_binding.safe_display_name,
            ),
        )
        binding = _resolve_historical_binding(
            self._repository,
            result=result,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            request_hash=request_hash,
            semantic_hash=semantic_hash,
            expected_status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
        )
        self._set_publication_fence(
            tenant_id=tenant_id,
            binding_id=binding_ref,
            enabled=True,
        )
        created = result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED and not had_source
        return _lifecycle_result(result, binding=binding, created_new_source=created)

    def disable_indexed_source(
        self, command: DisableWorkspaceIndexedSourceCommand
    ) -> WorkspaceIndexedSourceLifecycleResult:
        validate_configuration_idempotency_hash(command.idempotency_key_hash)
        tenant_id, workspace_id = command.tenant_id.strip(), command.workspace_id.strip()
        binding_id = command.indexed_source_binding_id.strip()
        request_hash = normalize_disable_indexed_source_request_hash(
            tenant_id=tenant_id, workspace_id=workspace_id, indexed_source_binding_id=binding_id
        )
        existing = self._repository.get_knowledge_configuration_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE,
            idempotency_key_hash=command.idempotency_key_hash,
        )
        if existing is not None and existing.normalized_request_hash != request_hash:
            raise WorkspaceKnowledgeConfigurationMutationError("configuration_idempotency_conflict")
        if (
            existing is not None
            and existing.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
        ):
            result = self._mutation_engine.execute(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE,
                expected_revision=command.expected_revision,
                idempotency_key_hash=command.idempotency_key_hash,
                normalized_request_hash=request_hash,
                semantic_identity_hash=existing.semantic_identity_hash,
                intent=DisableIndexedSourceMutationIntent(
                    indexed_source_binding_id=binding_id,
                    knowledge_source_binding_ref="",
                ),
            )
            if result.disposition is not WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY:
                raise _incomplete()
            historical = _highest_binding(
                self._repository.list_knowledge_indexed_source_versions(
                    tenant_id=tenant_id, workspace_id=workspace_id,
                ),
                binding_id=binding_id,
                revision=result.configuration_revision,
            )
            if historical is None:
                raise _incomplete()
            binding_ref = historical.knowledge_source_binding_ref.strip()
            semantic_hash = semantic_identity_hash_for_disable_indexed_source(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                knowledge_source_binding_ref=binding_ref,
            )
            source_id = connected_source_id(tenant_id, workspace_id, binding_ref)
            binding = _resolve_historical_binding(
                self._repository, result=result, tenant_id=tenant_id, workspace_id=workspace_id,
                binding_id=binding_id, source_id=source_id, request_hash=request_hash,
                semantic_hash=semantic_hash,
                expected_status=WorkspaceIndexedSourceBindingStatusV1.DISABLED,
            )
            self._set_publication_fence(
                tenant_id=tenant_id,
                binding_id=binding_ref,
                enabled=False,
            )
            return _lifecycle_result(result, binding=binding, created_new_source=False)
        configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id, workspace_id=workspace_id,
        )
        if configuration is None:
            raise WorkspaceIndexedSourceLifecycleError("workspace_not_found")
        current = next(
            (b for b in configuration.indexed_sources if b.indexed_source_binding_id == binding_id),
            None,
        )
        if current is None:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_not_found")
        semantic_hash = semantic_identity_hash_for_disable_indexed_source(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            knowledge_source_binding_ref=current.knowledge_source_binding_ref,
        )
        source_id = connected_source_id(tenant_id, workspace_id, current.knowledge_source_binding_ref)
        result = self._mutation_engine.execute(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE,
            expected_revision=command.expected_revision,
            idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=request_hash,
            semantic_identity_hash=semantic_hash,
            intent=DisableIndexedSourceMutationIntent(
                indexed_source_binding_id=binding_id,
                knowledge_source_binding_ref=current.knowledge_source_binding_ref,
            ),
        )
        binding = _resolve_historical_binding(
            self._repository, result=result, tenant_id=tenant_id, workspace_id=workspace_id,
            binding_id=binding_id, source_id=source_id, request_hash=request_hash,
            semantic_hash=semantic_hash,
            expected_status=WorkspaceIndexedSourceBindingStatusV1.DISABLED,
        )
        self._set_publication_fence(
            tenant_id=tenant_id,
            binding_id=current.knowledge_source_binding_ref,
            enabled=False,
        )
        return _lifecycle_result(result, binding=binding, created_new_source=False)


IndexedSourceLifecycleService = WorkspaceIndexedSourceLifecycleService
IndexedSourceLifecycleResult = IndexedSourceLifecycleResultV1
