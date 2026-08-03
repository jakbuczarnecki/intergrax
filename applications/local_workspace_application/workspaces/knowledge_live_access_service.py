# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Live Access Binding lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
    RemoteResourceAvailabilityV1,
    RemoteResourceDescriptorV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
    TenantConnectionPort,
    TenantLiveCapabilityCatalogPort,
    is_bindable_read_only_capability,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    live_access_binding_id_from_semantic_hash,
    normalize_create_live_access_binding_request_hash,
    normalize_disable_live_access_binding_request_hash,
    normalize_live_access_capability_set,
    normalize_live_access_remote_resource_id,
    semantic_identity_hash_for_live_access_binding,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
    LiveAccessBindingStatusV1,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
    WorkspaceLiveAccessBinding,
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
from local_workspace_application.workspaces.knowledge_live_access_handlers import (
    CreateLiveAccessBindingMutationIntent,
    DisableLiveAccessBindingMutationIntent,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_RESULT_TYPE = "live_access_binding"


class LiveAccessRemoteResourceLookupPort(Protocol):
    async def get_remote_resource(
        self, *, tenant_id: str, connection_ref: str, remote_resource_id: str,
    ) -> RemoteResourceDescriptorV1 | None: ...


class WorkspaceLiveAccessBindingError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class CreateWorkspaceLiveAccessBindingCommand:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    remote_resource_id: str | None
    allowed_capability_ids: tuple[str, ...]
    expected_revision: int
    idempotency_key_hash: str
    audience_eligibility: KnowledgeAudienceEligibilityV1 = KnowledgeAudienceEligibilityV1.PERSONAL_ONLY


@dataclass(frozen=True, slots=True)
class DisableWorkspaceLiveAccessBindingCommand:
    tenant_id: str
    workspace_id: str
    live_access_binding_id: str
    expected_revision: int
    idempotency_key_hash: str


@dataclass(frozen=True, slots=True)
class WorkspaceLiveAccessBindingResult:
    binding: WorkspaceLiveAccessBinding
    configuration_revision: int
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1
    created_new_binding: bool
    mutation: WorkspaceKnowledgeMutationRecord


def _incomplete() -> WorkspaceLiveAccessBindingError:
    return WorkspaceLiveAccessBindingError("live_access_projection_incomplete")


def _highest_binding(versions: list[WorkspaceLiveAccessBinding], *, binding_id: str, revision: int):
    matches = [v for v in versions if v.live_access_binding_id == binding_id and v.effective_revision <= revision]
    if not matches:
        return None
    top = max(matches, key=lambda item: item.effective_revision)
    if sum(1 for item in matches if item.effective_revision == top.effective_revision) > 1:
        raise _incomplete()
    return top


def _create_hashes(command: CreateWorkspaceLiveAccessBindingCommand):
    tenant_id, workspace_id = command.tenant_id.strip(), command.workspace_id.strip()
    connection_ref = command.connection_ref.strip()
    resource_id = normalize_live_access_remote_resource_id(command.remote_resource_id)
    capabilities = normalize_live_access_capability_set(command.allowed_capability_ids)
    request_hash = normalize_create_live_access_binding_request_hash(
        tenant_id=tenant_id, workspace_id=workspace_id, connection_ref=connection_ref,
        remote_resource_id=resource_id, allowed_capability_ids=capabilities,
        audience_eligibility=command.audience_eligibility,
    )
    semantic_hash = semantic_identity_hash_for_live_access_binding(
        tenant_id=tenant_id, workspace_id=workspace_id, connection_ref=connection_ref,
        normalized_remote_resource_id=resource_id, normalized_capability_set=capabilities,
    )
    return tenant_id, workspace_id, connection_ref, request_hash, semantic_hash, (
        live_access_binding_id_from_semantic_hash(semantic_hash), resource_id, capabilities,
    )


def _intent_from_binding(binding: WorkspaceLiveAccessBinding) -> CreateLiveAccessBindingMutationIntent:
    return CreateLiveAccessBindingMutationIntent(
        connection_ref=binding.connection_ref, remote_resource_id=binding.remote_resource_id,
        allowed_capability_ids=binding.allowed_capability_ids,
        audience_eligibility=binding.audience_eligibility,
        derived_provider_id=binding.derived_provider_id,
        derived_integration_kind=binding.derived_integration_kind,
        derived_resource_type=binding.derived_resource_type,
        derived_safe_display_label=binding.derived_safe_display_label,
    )


def _validate_catalog(descriptors: object, connection: SafeTenantConnectionV1, selected: tuple[str, ...]):
    if not isinstance(descriptors, tuple):
        raise WorkspaceLiveAccessBindingError("capability_catalog_invalid")
    catalog: dict[str, LiveCapabilityDescriptorV1] = {}
    for raw in descriptors:
        if not isinstance(raw, BaseModel):
            raise WorkspaceLiveAccessBindingError("capability_catalog_invalid")
        descriptor = LiveCapabilityDescriptorV1.model_validate(raw.model_dump())
        if descriptor.provider_id != connection.provider_id or descriptor.integration_kind != connection.integration_kind:
            raise WorkspaceLiveAccessBindingError("capability_catalog_invalid")
        if descriptor.capability_id in catalog:
            raise WorkspaceLiveAccessBindingError("capability_catalog_invalid")
        catalog[descriptor.capability_id] = descriptor
    for capability_id in selected:
        descriptor = catalog.get(capability_id)
        if descriptor is None:
            raise WorkspaceLiveAccessBindingError("capability_not_found")
        if (
            descriptor.effect is not CapabilityEffectV1.READ or not descriptor.read_only
            or not descriptor.available or not is_bindable_read_only_capability(descriptor)
        ):
            raise WorkspaceLiveAccessBindingError("capability_not_read_only")
    return catalog


def _resolve_historical_binding(
    repository: ManagedWorkspaceRepository, *, result: WorkspaceKnowledgeMutationExecutionResult,
    tenant_id: str, workspace_id: str, binding_id: str, request_hash: str, semantic_hash: str,
    expected_status: LiveAccessBindingStatusV1,
) -> WorkspaceLiveAccessBinding:
    mutation = result.mutation
    if (
        mutation.normalized_request_hash != request_hash or mutation.semantic_identity_hash != semantic_hash
        or mutation.result_entity_type != _RESULT_TYPE or mutation.result_entity_id != binding_id
        or mutation.committed_revision != result.configuration_revision
    ):
        raise _incomplete()
    binding = _highest_binding(
        repository.list_knowledge_live_access_versions(tenant_id=tenant_id, workspace_id=workspace_id),
        binding_id=binding_id, revision=result.configuration_revision,
    )
    if binding is None or binding.tenant_id != tenant_id or binding.workspace_id != workspace_id:
        raise _incomplete()
    if binding.live_access_binding_id != binding_id or binding.semantic_identity_hash != semantic_hash:
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
    return binding


def _result(binding, result, *, created_new_binding: bool) -> WorkspaceLiveAccessBindingResult:
    return WorkspaceLiveAccessBindingResult(
        binding=binding, configuration_revision=result.configuration_revision,
        disposition=result.disposition, created_new_binding=created_new_binding, mutation=result.mutation,
    )


class WorkspaceLiveAccessBindingService:
    def __init__(
        self, *, repository: ManagedWorkspaceRepository,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
        tenant_connection_port: TenantConnectionPort,
        capability_catalog: TenantLiveCapabilityCatalogPort,
        remote_resource_lookup_port: LiveAccessRemoteResourceLookupPort | None,
    ) -> None:
        self._repository = repository
        self._configuration_service = configuration_service
        self._mutation_engine = mutation_engine
        self._tenant_connection_port = tenant_connection_port
        self._capability_catalog = capability_catalog
        self._remote_resource_lookup_port = remote_resource_lookup_port

    async def create_live_access_binding(
        self, command: CreateWorkspaceLiveAccessBindingCommand,
    ) -> WorkspaceLiveAccessBindingResult:
        validate_configuration_idempotency_hash(command.idempotency_key_hash)
        tenant_id, workspace_id, connection_ref, request_hash, semantic_hash, (
            binding_id, resource_id, capabilities,
        ) = _create_hashes(command)
        existing = self._repository.get_knowledge_configuration_mutation(
            tenant_id=tenant_id, workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING,
            idempotency_key_hash=command.idempotency_key_hash,
        )
        if existing is not None:
            if existing.normalized_request_hash != request_hash:
                raise WorkspaceKnowledgeConfigurationMutationError("configuration_idempotency_conflict")
            if existing.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
                historical = _highest_binding(
                    self._repository.list_knowledge_live_access_versions(
                        tenant_id=tenant_id, workspace_id=workspace_id,
                    ),
                    binding_id=binding_id, revision=existing.committed_revision or 0,
                )
                if historical is None:
                    raise _incomplete()
                replay = self._mutation_engine.execute(
                    tenant_id=tenant_id, workspace_id=workspace_id,
                    operation=WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING,
                    expected_revision=command.expected_revision,
                    idempotency_key_hash=command.idempotency_key_hash,
                    normalized_request_hash=request_hash, semantic_identity_hash=semantic_hash,
                    intent=_intent_from_binding(historical),
                )
                if replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY:
                    binding = _resolve_historical_binding(
                        self._repository, result=replay, tenant_id=tenant_id, workspace_id=workspace_id,
                        binding_id=binding_id, request_hash=request_hash, semantic_hash=semantic_hash,
                        expected_status=LiveAccessBindingStatusV1.ACTIVE,
                    )
                    return _result(binding, replay, created_new_binding=False)
        configuration = self._configuration_service.get_configuration(tenant_id=tenant_id, workspace_id=workspace_id)
        if configuration is None:
            raise WorkspaceLiveAccessBindingError("workspace_not_found")
        attachments = [a for a in configuration.connection_attachments if a.connection_ref == connection_ref]
        if len(attachments) != 1:
            raise WorkspaceLiveAccessBindingError("connection_not_attached")
        if attachments[0].status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
            raise WorkspaceLiveAccessBindingError(
                "connection_not_attached"
                if attachments[0].status is WorkspaceConnectionAttachmentStatusV1.DETACHED
                else "connection_unavailable"
            )
        connection = self._resolve_connection(tenant_id=tenant_id, connection_ref=connection_ref)
        try:
            descriptors = self._capability_catalog.list_capabilities(
                tenant_id=tenant_id, connection_ref=connection_ref, remote_resource_id=resource_id,
            )
        except Exception as exc:
            raise WorkspaceLiveAccessBindingError("capability_catalog_unavailable") from exc
        catalog = _validate_catalog(descriptors, connection, capabilities)
        if any(catalog[c].resource_scope_required for c in capabilities) and resource_id is None:
            raise WorkspaceLiveAccessBindingError("remote_resource_required")
        derived_type, derived_label = None, connection.safe_display_name
        if resource_id is not None:
            resource = await self._resolve_remote_resource(
                tenant_id=tenant_id, connection_ref=connection_ref, remote_resource_id=resource_id,
                connection=connection, selected_capabilities=capabilities, catalog=catalog,
            )
            derived_type, derived_label = resource.resource_type, resource.safe_display_label
        had_binding = any(
            v.live_access_binding_id == binding_id
            for v in self._repository.list_knowledge_live_access_versions(
                tenant_id=tenant_id, workspace_id=workspace_id,
            )
        )
        mutation_result = self._mutation_engine.execute(
            tenant_id=tenant_id, workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING,
            expected_revision=command.expected_revision, idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=request_hash, semantic_identity_hash=semantic_hash,
            intent=CreateLiveAccessBindingMutationIntent(
                connection_ref=connection_ref, remote_resource_id=resource_id,
                allowed_capability_ids=capabilities, audience_eligibility=command.audience_eligibility,
                derived_provider_id=connection.provider_id, derived_integration_kind=connection.integration_kind,
                derived_resource_type=derived_type, derived_safe_display_label=derived_label,
            ),
        )
        binding = _resolve_historical_binding(
            self._repository, result=mutation_result, tenant_id=tenant_id, workspace_id=workspace_id,
            binding_id=binding_id, request_hash=request_hash, semantic_hash=semantic_hash,
            expected_status=LiveAccessBindingStatusV1.ACTIVE,
        )
        created = (
            mutation_result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
            and not had_binding
        )
        return _result(binding, mutation_result, created_new_binding=created)

    def disable_live_access_binding(
        self, command: DisableWorkspaceLiveAccessBindingCommand,
    ) -> WorkspaceLiveAccessBindingResult:
        validate_configuration_idempotency_hash(command.idempotency_key_hash)
        tenant_id, workspace_id = command.tenant_id.strip(), command.workspace_id.strip()
        binding_id = command.live_access_binding_id.strip()
        request_hash = normalize_disable_live_access_binding_request_hash(
            tenant_id=tenant_id, workspace_id=workspace_id, live_access_binding_id=binding_id,
        )
        existing = self._repository.get_knowledge_configuration_mutation(
            tenant_id=tenant_id, workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING,
            idempotency_key_hash=command.idempotency_key_hash,
        )
        if existing is not None and existing.normalized_request_hash != request_hash:
            raise WorkspaceKnowledgeConfigurationMutationError("configuration_idempotency_conflict")
        semantic_hash: str | None = None
        if existing is not None and existing.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
            semantic_hash = existing.semantic_identity_hash
        else:
            configuration = self._configuration_service.get_configuration(tenant_id=tenant_id, workspace_id=workspace_id)
            if configuration is None:
                raise WorkspaceLiveAccessBindingError("workspace_not_found")
            current = next((b for b in configuration.live_access_bindings if b.live_access_binding_id == binding_id), None)
            if current is None:
                raise WorkspaceLiveAccessBindingError("live_access_binding_not_found")
            semantic_hash = current.semantic_identity_hash
        result = self._mutation_engine.execute(
            tenant_id=tenant_id, workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING,
            expected_revision=command.expected_revision, idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=request_hash, semantic_identity_hash=semantic_hash,
            intent=DisableLiveAccessBindingMutationIntent(live_access_binding_id=binding_id),
        )
        binding = _resolve_historical_binding(
            self._repository, result=result, tenant_id=tenant_id, workspace_id=workspace_id,
            binding_id=binding_id, request_hash=request_hash, semantic_hash=semantic_hash,
            expected_status=LiveAccessBindingStatusV1.DISABLED,
        )
        return _result(binding, result, created_new_binding=False)

    def _resolve_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1:
        try:
            connection = self._tenant_connection_port.get_connection(
                tenant_id=tenant_id, connection_ref=connection_ref,
            )
        except Exception as exc:
            raise WorkspaceLiveAccessBindingError("connection_unavailable") from exc
        if connection is None or connection.tenant_id != tenant_id:
            raise WorkspaceLiveAccessBindingError("connection_not_found")
        if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
            raise WorkspaceLiveAccessBindingError("connection_unavailable")
        return connection

    async def _resolve_remote_resource(
        self, *, tenant_id: str, connection_ref: str, remote_resource_id: str,
        connection: SafeTenantConnectionV1, selected_capabilities: tuple[str, ...],
        catalog: dict[str, LiveCapabilityDescriptorV1],
    ) -> RemoteResourceDescriptorV1:
        if self._remote_resource_lookup_port is None:
            raise WorkspaceLiveAccessBindingError("remote_resource_lookup_unavailable")
        try:
            resource = await self._remote_resource_lookup_port.get_remote_resource(
                tenant_id=tenant_id, connection_ref=connection_ref, remote_resource_id=remote_resource_id,
            )
        except Exception as exc:
            raise WorkspaceLiveAccessBindingError("remote_resource_lookup_unavailable") from exc
        if resource is None:
            raise WorkspaceLiveAccessBindingError("remote_resource_not_found")
        if resource.connection_ref.strip() != connection_ref.strip():
            raise WorkspaceLiveAccessBindingError("remote_resource_connection_mismatch")
        if resource.provider_id != connection.provider_id or resource.integration_kind != connection.integration_kind:
            raise WorkspaceLiveAccessBindingError("remote_resource_not_found")
        if resource.availability is not RemoteResourceAvailabilityV1.AVAILABLE:
            raise WorkspaceLiveAccessBindingError(
                "remote_resource_not_found"
                if resource.availability is RemoteResourceAvailabilityV1.NOT_FOUND
                else "remote_resource_unavailable"
            )
        for capability_id in selected_capabilities:
            if capability_id not in resource.supported_capability_ids:
                raise WorkspaceLiveAccessBindingError("remote_resource_capability_mismatch")
            descriptor = catalog[capability_id]
            if descriptor.supported_resource_types and resource.resource_type not in descriptor.supported_resource_types:
                raise WorkspaceLiveAccessBindingError("remote_resource_type_unsupported")
        return resource
