# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Live Access Binding lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    live_access_binding_id_from_semantic_hash,
    live_access_binding_stage_manifest_hash,
    normalize_create_live_access_binding_request_hash,
    normalize_detach_live_access_binding_request_hash,
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
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    AttachWorkspaceConnectionCommand,
    WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.knowledge_live_access_handlers import (
    CreateLiveAccessBindingMutationIntent,
    DetachLiveAccessBindingMutationIntent,
    DisableLiveAccessBindingMutationIntent,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from pydantic import BaseModel, ConfigDict, Field

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

_RESULT_TYPE = "live_access_binding"


class LiveAccessRemoteResourceLookupPort(Protocol):
    async def get_remote_resource(
        self, *, tenant_id: str, connection_ref: str, remote_resource_id: str,
    ) -> RemoteResourceDescriptorV1 | None: ...


class WorkspaceLiveAccessBindingError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class LiveAccessLifecycleStateV1(StrEnum):
    READY = "ready"
    ACTIVE = "active"
    DISABLED = "disabled"
    DETACHED = "detached"
    ERROR = "error"


class LiveAccessRuntimeBindingStateV1(StrEnum):
    USABLE = "usable"
    UNAVAILABLE = "unavailable"
    NOT_ATTACHED = "not_attached"
    NOT_FOUND = "not_found"
    DISABLED = "disabled"
    DETACHED = "detached"


class LiveAccessLifecycleViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    live_access_binding_id: str
    connection_ref: str
    remote_resource_id: str | None = None
    allowed_capability_ids: tuple[str, ...]
    lifecycle_state: LiveAccessLifecycleStateV1
    configuration_revision: int = Field(..., ge=0)
    enabled: bool
    detached: bool
    runtime_available: bool
    runtime_binding_state: LiveAccessRuntimeBindingStateV1
    last_error_code: str | None = None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AttachLiveAccessCommand:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    remote_resource_id: str | None
    allowed_capability_ids: tuple[str, ...]
    expected_revision: int
    idempotency_key_hash: str
    audience_eligibility: KnowledgeAudienceEligibilityV1 = (
        KnowledgeAudienceEligibilityV1.PERSONAL_ONLY
    )
    requested_safe_display_label: str | None = None


@dataclass(frozen=True, slots=True)
class GetLiveAccessCommand:
    tenant_id: str
    workspace_id: str
    live_access_binding_id: str


@dataclass(frozen=True, slots=True)
class EnableWorkspaceLiveAccessBindingCommand:
    tenant_id: str
    workspace_id: str
    live_access_binding_id: str
    expected_revision: int
    idempotency_key_hash: str


@dataclass(frozen=True, slots=True)
class DetachWorkspaceLiveAccessBindingCommand:
    tenant_id: str
    workspace_id: str
    live_access_binding_id: str
    expected_revision: int
    idempotency_key_hash: str


@dataclass(frozen=True, slots=True)
class LiveAccessLifecycleResultV1:
    binding: WorkspaceLiveAccessBinding
    view: LiveAccessLifecycleViewV1
    configuration_revision: int
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1


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
        try:
            descriptor = LiveCapabilityDescriptorV1.model_validate(raw.model_dump())
        except Exception:
            raise WorkspaceLiveAccessBindingError("capability_catalog_invalid") from None
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
        *,
        _validate_runtime: bool = True,
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
                    stage_manifest_hash=existing.stage_manifest_hash,
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
        historical = next(
            (
                item
                for item in configuration.live_access_bindings
                if item.live_access_binding_id == binding_id
            ),
            None,
        )
        if historical is not None and historical.status is LiveAccessBindingStatusV1.REVOKED:
            raise WorkspaceLiveAccessBindingError("live_access_detached")
        if historical is not None and (
            historical.semantic_identity_hash != semantic_hash
            or historical.connection_ref != connection_ref
            or historical.remote_resource_id != resource_id
            or historical.allowed_capability_ids != capabilities
            or historical.audience_eligibility is not command.audience_eligibility
        ):
            raise WorkspaceLiveAccessBindingError("live_access_binding_conflict")
        if not _validate_runtime and historical is None:
            raise WorkspaceLiveAccessBindingError("live_access_binding_not_found")
        attachments = [a for a in configuration.connection_attachments if a.connection_ref == connection_ref]
        if len(attachments) != 1:
            raise WorkspaceLiveAccessBindingError("connection_not_attached")
        if attachments[0].status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
            raise WorkspaceLiveAccessBindingError(
                "connection_not_attached"
                if attachments[0].status is WorkspaceConnectionAttachmentStatusV1.DETACHED
                else "connection_unavailable"
            )
        if not _validate_runtime and historical is not None:
            derived_provider_id = historical.derived_provider_id
            derived_integration_kind = historical.derived_integration_kind
            derived_type = historical.derived_resource_type
            derived_label = historical.derived_safe_display_label
        else:
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
            derived_provider_id = connection.provider_id
            derived_integration_kind = connection.integration_kind
        had_binding = any(
            v.live_access_binding_id == binding_id
            for v in self._repository.list_knowledge_live_access_versions(
                tenant_id=tenant_id, workspace_id=workspace_id,
            )
        )
        create_intent = CreateLiveAccessBindingMutationIntent(
            connection_ref=connection_ref, remote_resource_id=resource_id,
            allowed_capability_ids=capabilities, audience_eligibility=command.audience_eligibility,
            derived_provider_id=derived_provider_id, derived_integration_kind=derived_integration_kind,
            derived_resource_type=derived_type, derived_safe_display_label=derived_label,
        )
        manifest_hash = live_access_binding_stage_manifest_hash(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            live_access_binding_id=binding_id,
            connection_ref=connection_ref,
            remote_resource_id=resource_id,
            allowed_capability_ids=capabilities,
            audience_eligibility=command.audience_eligibility,
            derived_provider_id=derived_provider_id,
            derived_integration_kind=derived_integration_kind,
            derived_resource_type=derived_type,
            derived_safe_display_label=derived_label,
        )
        mutation_result = self._mutation_engine.execute(
            tenant_id=tenant_id, workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING,
            expected_revision=command.expected_revision, idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=request_hash, semantic_identity_hash=semantic_hash,
            stage_manifest_hash=manifest_hash,
            intent=create_intent,
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
            if current.status is LiveAccessBindingStatusV1.REVOKED:
                raise WorkspaceLiveAccessBindingError("live_access_detached")
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

    async def enable_live_access_binding(
        self,
        command: EnableWorkspaceLiveAccessBindingCommand,
    ) -> WorkspaceLiveAccessBindingResult:
        configuration = self._configuration_service.get_configuration(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
        )
        if configuration is None:
            raise WorkspaceLiveAccessBindingError("workspace_not_found")
        binding = next(
            (
                item
                for item in configuration.live_access_bindings
                if item.live_access_binding_id == command.live_access_binding_id
            ),
            None,
        )
        if binding is None:
            raise WorkspaceLiveAccessBindingError("live_access_binding_not_found")
        if binding.status is LiveAccessBindingStatusV1.REVOKED:
            raise WorkspaceLiveAccessBindingError("live_access_detached")
        return await self.create_live_access_binding(
            CreateWorkspaceLiveAccessBindingCommand(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                connection_ref=binding.connection_ref,
                remote_resource_id=binding.remote_resource_id,
                allowed_capability_ids=binding.allowed_capability_ids,
                expected_revision=command.expected_revision,
                idempotency_key_hash=command.idempotency_key_hash,
                audience_eligibility=binding.audience_eligibility,
            ),
            _validate_runtime=False,
        )

    def detach_live_access_binding(
        self,
        command: DetachWorkspaceLiveAccessBindingCommand,
    ) -> WorkspaceLiveAccessBindingResult:
        validate_configuration_idempotency_hash(command.idempotency_key_hash)
        tenant_id, workspace_id = command.tenant_id.strip(), command.workspace_id.strip()
        binding_id = command.live_access_binding_id.strip()
        request_hash = normalize_detach_live_access_binding_request_hash(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            live_access_binding_id=binding_id,
        )
        configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise WorkspaceLiveAccessBindingError("workspace_not_found")
        current = next(
            (
                item
                for item in configuration.live_access_bindings
                if item.live_access_binding_id == binding_id
            ),
            None,
        )
        if current is None:
            raise WorkspaceLiveAccessBindingError("live_access_binding_not_found")
        result = self._mutation_engine.execute(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.DETACH_LIVE_ACCESS_BINDING,
            expected_revision=command.expected_revision,
            idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=request_hash,
            semantic_identity_hash=current.semantic_identity_hash,
            intent=DetachLiveAccessBindingMutationIntent(
                live_access_binding_id=binding_id,
            ),
        )
        binding = _resolve_historical_binding(
            self._repository,
            result=result,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            request_hash=request_hash,
            semantic_hash=current.semantic_identity_hash,
            expected_status=LiveAccessBindingStatusV1.REVOKED,
        )
        return _result(binding, result, created_new_binding=False)

    def _resolve_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1:
        try:
            raw_connection = self._tenant_connection_port.get_connection(
                tenant_id=tenant_id, connection_ref=connection_ref,
            )
        except Exception as exc:
            raise WorkspaceLiveAccessBindingError("connection_unavailable") from exc
        if raw_connection is None:
            raise WorkspaceLiveAccessBindingError("connection_not_found")
        if not isinstance(raw_connection, BaseModel):
            raise WorkspaceLiveAccessBindingError("connection_unavailable")
        try:
            connection = SafeTenantConnectionV1.model_validate(raw_connection.model_dump())
        except Exception:
            raise WorkspaceLiveAccessBindingError("connection_unavailable") from None
        if connection.tenant_id.strip() != tenant_id:
            raise WorkspaceLiveAccessBindingError("connection_not_found")
        if connection.connection_ref.strip() != connection_ref:
            raise WorkspaceLiveAccessBindingError("connection_unavailable")
        if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
            raise WorkspaceLiveAccessBindingError("connection_unavailable")
        if (
            not connection.provider_id
            or connection.provider_id != connection.provider_id.strip()
            or not connection.safe_display_name
            or connection.safe_display_name != connection.safe_display_name.strip()
            or connection.tenant_id != connection.tenant_id.strip()
            or connection.connection_ref != connection.connection_ref.strip()
        ):
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
        if not isinstance(resource, BaseModel):
            raise WorkspaceLiveAccessBindingError("remote_resource_lookup_invalid")
        try:
            resource = RemoteResourceDescriptorV1.model_validate(resource.model_dump())
        except Exception:
            raise WorkspaceLiveAccessBindingError("remote_resource_lookup_invalid") from None
        if resource.remote_resource_id.strip() != remote_resource_id.strip():
            raise WorkspaceLiveAccessBindingError("remote_resource_lookup_invalid")
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


class LiveAccessRuntimeAuthorityPort(Protocol):
    def is_usable(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        live_access_binding_id: str,
        connection_ref: str,
        capability_id: str,
    ) -> bool:
        ...


def _probe_runtime(
    *,
    binding: WorkspaceLiveAccessBinding,
    configuration: object,
    tenant_connection_port: TenantConnectionPort,
    capability_catalog: TenantLiveCapabilityCatalogPort,
) -> tuple[bool, LiveAccessRuntimeBindingStateV1, str | None]:
    if binding.status is LiveAccessBindingStatusV1.REVOKED:
        return False, LiveAccessRuntimeBindingStateV1.DETACHED, None
    if binding.status is LiveAccessBindingStatusV1.DISABLED:
        return False, LiveAccessRuntimeBindingStateV1.DISABLED, None
    if binding.status is LiveAccessBindingStatusV1.UNAVAILABLE:
        return False, LiveAccessRuntimeBindingStateV1.UNAVAILABLE, "connection_unavailable"
    attachments = getattr(configuration, "connection_attachments", ())
    attachment = next(
        (item for item in attachments if item.connection_ref == binding.connection_ref),
        None,
    )
    if attachment is None or attachment.status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
        return False, LiveAccessRuntimeBindingStateV1.NOT_ATTACHED, "connection_not_attached"
    try:
        connection = tenant_connection_port.get_connection(
            tenant_id=binding.tenant_id,
            connection_ref=binding.connection_ref,
        )
    except Exception:  # noqa: BLE001 - runtime probing fails closed
        return False, LiveAccessRuntimeBindingStateV1.UNAVAILABLE, "connection_unavailable"
    if connection is None:
        return False, LiveAccessRuntimeBindingStateV1.NOT_FOUND, "connection_not_found"
    if (
        not isinstance(connection, BaseModel)
        or connection.tenant_id != binding.tenant_id
        or connection.connection_ref != binding.connection_ref
    ):
        return False, LiveAccessRuntimeBindingStateV1.UNAVAILABLE, "connection_unavailable"
    try:
        safe_connection = SafeTenantConnectionV1.model_validate(connection.model_dump())
    except Exception:  # noqa: BLE001 - runtime probing fails closed
        return False, LiveAccessRuntimeBindingStateV1.UNAVAILABLE, "connection_unavailable"
    if safe_connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
        return False, LiveAccessRuntimeBindingStateV1.DISABLED, "connection_unavailable"
    try:
        descriptors = capability_catalog.list_capabilities(
            tenant_id=binding.tenant_id,
            connection_ref=binding.connection_ref,
            remote_resource_id=binding.remote_resource_id,
        )
        _validate_catalog(descriptors, safe_connection, binding.allowed_capability_ids)
    except WorkspaceLiveAccessBindingError as exc:
        return False, LiveAccessRuntimeBindingStateV1.UNAVAILABLE, exc.error_code
    except Exception:  # noqa: BLE001 - runtime probing fails closed
        return False, LiveAccessRuntimeBindingStateV1.UNAVAILABLE, "capability_catalog_unavailable"
    return True, LiveAccessRuntimeBindingStateV1.USABLE, None


class WorkspaceLiveAccessRuntimeAuthority:
    """Reload configuration and runtime capability authority before every live call."""

    def __init__(
        self,
        *,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        tenant_connection_port: TenantConnectionPort,
        capability_catalog: TenantLiveCapabilityCatalogPort,
    ) -> None:
        self._configuration_service = configuration_service
        self._tenant_connection_port = tenant_connection_port
        self._capability_catalog = capability_catalog

    def is_usable(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        live_access_binding_id: str,
        connection_ref: str,
        capability_id: str,
    ) -> bool:
        configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            return False
        binding = next(
            (
                item
                for item in configuration.live_access_bindings
                if item.live_access_binding_id == live_access_binding_id
            ),
            None,
        )
        if (
            binding is None
            or binding.connection_ref != connection_ref
            or capability_id not in binding.allowed_capability_ids
            or binding.status is not LiveAccessBindingStatusV1.ACTIVE
        ):
            return False
        usable, _, _ = _probe_runtime(
            binding=binding,
            configuration=configuration,
            tenant_connection_port=self._tenant_connection_port,
            capability_catalog=self._capability_catalog,
        )
        return usable


class LiveAccessLifecycleService:
    """Provider-neutral façade over connection, configuration and runtime authorities."""

    def __init__(
        self,
        *,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        live_access_binding_service: WorkspaceLiveAccessBindingService,
        connection_attachment_service: WorkspaceConnectionAttachmentService,
        tenant_connection_port: TenantConnectionPort,
        capability_catalog: TenantLiveCapabilityCatalogPort,
    ) -> None:
        self._configuration_service = configuration_service
        self._live_access_binding_service = live_access_binding_service
        self._connection_attachment_service = connection_attachment_service
        self._tenant_connection_port = tenant_connection_port
        self._capability_catalog = capability_catalog

    async def attach(self, command: AttachLiveAccessCommand) -> LiveAccessLifecycleResultV1:
        attachment = self._connection_attachment_service.attach_connection(
            AttachWorkspaceConnectionCommand(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                connection_ref=command.connection_ref,
                expected_revision=command.expected_revision,
                idempotency_key_hash=command.idempotency_key_hash,
                requested_safe_display_label=command.requested_safe_display_label,
            )
        )
        result = await self._live_access_binding_service.create_live_access_binding(
            CreateWorkspaceLiveAccessBindingCommand(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                connection_ref=command.connection_ref,
                remote_resource_id=command.remote_resource_id,
                allowed_capability_ids=command.allowed_capability_ids,
                expected_revision=attachment.configuration_revision,
                idempotency_key_hash=command.idempotency_key_hash,
                audience_eligibility=command.audience_eligibility,
            )
        )
        return self._lifecycle_result(result)

    def get(self, command: GetLiveAccessCommand) -> LiveAccessLifecycleViewV1:
        return self._view(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            live_access_binding_id=command.live_access_binding_id,
        )

    def disable(
        self,
        command: DisableWorkspaceLiveAccessBindingCommand,
    ) -> LiveAccessLifecycleResultV1:
        result = self._live_access_binding_service.disable_live_access_binding(command)
        return self._lifecycle_result(result)

    async def enable(
        self,
        command: EnableWorkspaceLiveAccessBindingCommand,
    ) -> LiveAccessLifecycleResultV1:
        result = await self._live_access_binding_service.enable_live_access_binding(command)
        return self._lifecycle_result(result)

    def detach(
        self,
        command: DetachWorkspaceLiveAccessBindingCommand,
    ) -> LiveAccessLifecycleResultV1:
        result = self._live_access_binding_service.detach_live_access_binding(command)
        return self._lifecycle_result(result)

    def _lifecycle_result(
        self,
        result: WorkspaceLiveAccessBindingResult,
    ) -> LiveAccessLifecycleResultV1:
        return LiveAccessLifecycleResultV1(
            binding=result.binding,
            view=self._view(
                tenant_id=result.binding.tenant_id,
                workspace_id=result.binding.workspace_id,
                live_access_binding_id=result.binding.live_access_binding_id,
            ),
            configuration_revision=result.configuration_revision,
            disposition=result.disposition,
        )

    def _view(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        live_access_binding_id: str,
    ) -> LiveAccessLifecycleViewV1:
        configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise WorkspaceLiveAccessBindingError("live_access_not_found")
        binding = next(
            (
                item
                for item in configuration.live_access_bindings
                if item.live_access_binding_id == live_access_binding_id
            ),
            None,
        )
        if binding is None:
            raise WorkspaceLiveAccessBindingError("live_access_not_found")
        usable, runtime_state, error_code = _probe_runtime(
            binding=binding,
            configuration=configuration,
            tenant_connection_port=self._tenant_connection_port,
            capability_catalog=self._capability_catalog,
        )
        if binding.status is LiveAccessBindingStatusV1.REVOKED:
            lifecycle_state = LiveAccessLifecycleStateV1.DETACHED
        elif binding.status is LiveAccessBindingStatusV1.DISABLED:
            lifecycle_state = LiveAccessLifecycleStateV1.DISABLED
        elif usable:
            lifecycle_state = LiveAccessLifecycleStateV1.ACTIVE
        elif error_code in {"connection_not_found", "capability_catalog_invalid"}:
            lifecycle_state = LiveAccessLifecycleStateV1.ERROR
        else:
            lifecycle_state = LiveAccessLifecycleStateV1.READY
        return LiveAccessLifecycleViewV1(
            tenant_id=binding.tenant_id,
            workspace_id=binding.workspace_id,
            live_access_binding_id=binding.live_access_binding_id,
            connection_ref=binding.connection_ref,
            remote_resource_id=binding.remote_resource_id,
            allowed_capability_ids=binding.allowed_capability_ids,
            lifecycle_state=lifecycle_state,
            configuration_revision=configuration.configuration_revision,
            enabled=binding.status is not LiveAccessBindingStatusV1.REVOKED
            and binding.status is not LiveAccessBindingStatusV1.DISABLED,
            detached=binding.status is LiveAccessBindingStatusV1.REVOKED,
            runtime_available=usable,
            runtime_binding_state=runtime_state,
            last_error_code=error_code,
            updated_at=binding.updated_at,
        )
