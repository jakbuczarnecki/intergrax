# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Indexed Source lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding, KnowledgeSourceBindingStatus
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
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_RESULT_TYPE = "indexed_source_binding"

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
    ) -> None:
        self._repository = repository
        self._configuration_service = configuration_service
        self._mutation_engine = mutation_engine
        self._tenant_binding_port = tenant_binding_port

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
        return _lifecycle_result(result, binding=binding, created_new_source=False)
