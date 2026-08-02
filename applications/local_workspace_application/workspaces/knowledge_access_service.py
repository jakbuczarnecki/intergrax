# © Artur Czarnecki. All rights reserved.

"""Orchestration for connected workspace knowledge access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding
from local_workspace_application.workspaces.connected_source_candidate import (
    decode_slack_conversation_candidate_ref,
    validate_candidate_scope,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.connected_source_ids import connected_source_id
from local_workspace_application.workspaces.connected_source_discovery import (
    WorkspaceRemoteResourceDiscoveryService,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceBindingError,
    ConnectedSourceDiscoveryError,
    RemoteResourceDiscoveryPageV1,
    RemoteResourceTypeV1,
)
from local_workspace_application.workspaces.connected_source_tenant_binding import (
    SlackConversationTenantBindingRequest,
    WorkspaceConnectedSourceTenantBindingService,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    normalize_request_hash,
    semantic_identity_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    CreateIndexedSourceMutationIntent,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceKnowledgeMutationOperationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeMutationExecutionResult,
)
from local_workspace_application.workspaces.models import WorkspaceOperation
from local_workspace_application.workspaces.service import ManagedWorkspaceService


class TenantKnowledgeSourceBindingPort(Protocol):
    def get_binding(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSourceBinding | None:
        ...


@dataclass(frozen=True, slots=True)
class CreateConnectedIndexedSourceRequest:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    opaque_candidate_ref: str
    expected_revision: int
    idempotency_key_hash: str
    root_oldest: str
    root_latest: str
    start_sync: bool = False


@dataclass(frozen=True, slots=True)
class CreateConnectedIndexedSourceResult:
    binding_id: str
    source_id: str
    configuration_revision: int
    mutation_result: WorkspaceKnowledgeMutationExecutionResult
    sync_operation: WorkspaceOperation | None = None


class WorkspaceKnowledgeAccessService:
    def __init__(
        self,
        *,
        discovery_service: WorkspaceRemoteResourceDiscoveryService,
        tenant_binding_service: WorkspaceConnectedSourceTenantBindingService,
        mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
        workspace_service: ManagedWorkspaceService,
        tenant_binding_port: TenantKnowledgeSourceBindingPort,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
    ) -> None:
        self._discovery = discovery_service
        self._tenant_bindings = tenant_binding_service
        self._mutation_engine = mutation_engine
        self._workspace_service = workspace_service
        self._tenant_binding_port = tenant_binding_port
        self._codec = opaque_ref_codec

    async def list_slack_conversations(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        cursor: str | None = None,
        limit: int = 50,
    ) -> RemoteResourceDiscoveryPageV1:
        return await self._discovery.list_remote_resources(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
            cursor=cursor,
            limit=limit,
        )

    async def create_indexed_source_from_candidate(
        self,
        request: CreateConnectedIndexedSourceRequest,
    ) -> CreateConnectedIndexedSourceResult:
        payload = decode_slack_conversation_candidate_ref(
            self._codec,
            request.opaque_candidate_ref,
        )
        validate_candidate_scope(
            payload,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            connection_ref=request.connection_ref,
        )
        safe_label = await self._discovery.revalidate_candidate_label(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            connection_ref=request.connection_ref,
            conversation_id=payload.conversation_id,
            conversation_kind=payload.conversation_kind,
        )

        try:
            tenant_binding = self._tenant_bindings.create_or_get_equivalent_for_slack_conversation(
                SlackConversationTenantBindingRequest(
                    tenant_id=request.tenant_id,
                    connection_ref=request.connection_ref,
                    conversation_id=payload.conversation_id,
                    conversation_kind=payload.conversation_kind,
                    safe_display_name=safe_label,
                    root_oldest=request.root_oldest,
                    root_latest=request.root_latest,
                )
            )
        except ConnectedSourceBindingError as exc:
            raise ConnectedSourceDiscoveryError(exc.error_code) from exc

        sync_mode = IndexedSourceSyncModeV1.FULL
        audience_eligibility = IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY

        intent = CreateIndexedSourceMutationIntent(
            knowledge_source_binding_ref=tenant_binding.binding_id,
            sync_mode=sync_mode,
            audience_eligibility=audience_eligibility,
            cached_safe_display_label=safe_label,
        )
        normalized_hash = normalize_request_hash(
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            knowledge_source_binding_ref=tenant_binding.binding_id,
            sync_mode=sync_mode,
            audience_eligibility=audience_eligibility,
        )
        semantic_hash = semantic_identity_hash(
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            knowledge_source_binding_ref=tenant_binding.binding_id,
        )
        mutation_result = self._mutation_engine.execute(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            expected_revision=request.expected_revision,
            idempotency_key_hash=request.idempotency_key_hash,
            normalized_request_hash=normalized_hash,
            semantic_identity_hash=semantic_hash,
            intent=intent,
        )

        configuration = self._workspace_service.repository.get_knowledge_configuration_head(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
        )
        committed_revision = mutation_result.configuration_revision
        if committed_revision == 0 and configuration is not None:
            committed_revision = configuration.committed_revision

        durable_binding = self._tenant_binding_port.get_binding(
            tenant_id=request.tenant_id,
            binding_id=tenant_binding.binding_id,
        )
        if durable_binding is None:
            raise ConnectedSourceBindingError("knowledge_source_binding_not_found")

        sync_operation = None
        source_id = connected_source_id(
            request.tenant_id,
            request.workspace_id,
            tenant_binding.binding_id,
        )
        if request.start_sync:
            sync_operation = self._workspace_service.create_sync_operation(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                source_id=source_id,
            )

        return CreateConnectedIndexedSourceResult(
            binding_id=mutation_result.result_entity_id,
            source_id=source_id,
            configuration_revision=committed_revision,
            mutation_result=mutation_result,
            sync_operation=sync_operation,
        )

