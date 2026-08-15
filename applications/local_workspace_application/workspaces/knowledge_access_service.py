# © Artur Czarnecki. All rights reserved.

"""Orchestration for connected workspace knowledge access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from local_workspace_application.workspaces.connected_source_discovery import (
    WorkspaceRemoteResourceDiscoveryService,
)
from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
)
from local_workspace_application.workspaces.connected_source_models import (
    RemoteResourceDiscoveryPageV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeMutationExecutionResult,
)
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    ActivateWorkspaceIndexedSourceCommand,
    WorkspaceIndexedSourceLifecycleError,
    WorkspaceIndexedSourceLifecycleService,
)
from local_workspace_application.workspaces.models import WorkspaceOperation
from local_workspace_application.workspaces.service import ManagedWorkspaceService

from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding


class TenantKnowledgeSourceBindingPort(Protocol):
    def create_or_get_equivalent(
        self,
        binding: KnowledgeSourceBinding,
    ) -> KnowledgeSourceBinding:
        ...

    def get_binding(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSourceBinding | None:
        ...


class ConnectedSourceCandidateAdapter(Protocol):
    def build_binding(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        root_oldest: str,
        root_latest: str,
        safe_display_name: str | None = None,
    ) -> KnowledgeSourceBinding:
        ...

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
    ) -> str:
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
    created_new_source: bool = False
    sync_operation: WorkspaceOperation | None = None


def _lifecycle_execution_result(
    lifecycle_result,
    *,
    binding_id: str,
) -> WorkspaceKnowledgeMutationExecutionResult:
    return WorkspaceKnowledgeMutationExecutionResult(
        disposition=lifecycle_result.disposition,
        mutation=lifecycle_result.mutation,
        configuration_revision=lifecycle_result.configuration_revision,
        result_entity_type="indexed_source_binding",
        result_entity_id=binding_id,
    )


class WorkspaceKnowledgeAccessService:
    def __init__(
        self,
        *,
        discovery_service: WorkspaceRemoteResourceDiscoveryService,
        tenant_binding_service: TenantKnowledgeSourceBindingPort,
        indexed_source_lifecycle_service: WorkspaceIndexedSourceLifecycleService,
        workspace_service: ManagedWorkspaceService,
        tenant_binding_port: TenantKnowledgeSourceBindingPort,
        candidate_adapter: ConnectedSourceCandidateAdapter,
    ) -> None:
        self._discovery = discovery_service
        self._tenant_bindings = tenant_binding_service
        self._lifecycle = indexed_source_lifecycle_service
        self._workspace_service = workspace_service
        self._tenant_binding_port = tenant_binding_port
        self._candidate_adapter = candidate_adapter

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: object,
        cursor: str | None = None,
        limit: int = 50,
    ) -> RemoteResourceDiscoveryPageV1:
        return await self._discovery.list_remote_resources(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=resource_type,
            cursor=cursor,
            limit=limit,
        )

    async def create_indexed_source_from_candidate(
        self,
        request: CreateConnectedIndexedSourceRequest,
    ) -> CreateConnectedIndexedSourceResult:
        binding = self._candidate_adapter.build_binding(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            connection_ref=request.connection_ref,
            opaque_candidate_ref=request.opaque_candidate_ref,
            root_oldest=request.root_oldest,
            root_latest=request.root_latest,
        )
        expected_binding_id = binding.binding_id

        sync_mode = IndexedSourceSyncModeV1.FULL
        audience_eligibility = IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY
        activate_command = ActivateWorkspaceIndexedSourceCommand(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            knowledge_source_binding_ref=expected_binding_id,
            expected_revision=request.expected_revision,
            idempotency_key_hash=request.idempotency_key_hash,
            sync_mode=sync_mode,
            audience_eligibility=audience_eligibility,
        )
        replay = self._lifecycle.replay_activation_if_committed(activate_command)
        if replay is not None:
            source_id = connected_source_id(
                request.tenant_id,
                request.workspace_id,
                expected_binding_id,
            )
            return CreateConnectedIndexedSourceResult(
                binding_id=replay.binding.indexed_source_binding_id,
                source_id=source_id,
                configuration_revision=replay.configuration_revision,
                mutation_result=_lifecycle_execution_result(
                    replay,
                    binding_id=replay.binding.indexed_source_binding_id,
                ),
                created_new_source=False,
            )

        safe_label = await self._candidate_adapter.revalidate_candidate_label(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            connection_ref=request.connection_ref,
            opaque_candidate_ref=request.opaque_candidate_ref,
        )
        binding = self._candidate_adapter.build_binding(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            connection_ref=request.connection_ref,
            opaque_candidate_ref=request.opaque_candidate_ref,
            root_oldest=request.root_oldest,
            root_latest=request.root_latest,
            safe_display_name=safe_label,
        )
        try:
            tenant_binding = self._tenant_bindings.create_or_get_equivalent(binding)
        except Exception as exc:
            raise WorkspaceIndexedSourceLifecycleError(
                "knowledge_source_binding_unavailable"
            ) from exc
        if tenant_binding.binding_id != expected_binding_id:
            raise WorkspaceIndexedSourceLifecycleError("knowledge_source_binding_invalid")

        lifecycle_result = self._lifecycle.activate_indexed_source(
            ActivateWorkspaceIndexedSourceCommand(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                knowledge_source_binding_ref=tenant_binding.binding_id,
                expected_revision=request.expected_revision,
                idempotency_key_hash=request.idempotency_key_hash,
                sync_mode=sync_mode,
                audience_eligibility=audience_eligibility,
                cached_safe_display_label=safe_label,
            )
        )

        source_id = connected_source_id(
            request.tenant_id,
            request.workspace_id,
            tenant_binding.binding_id,
        )
        sync_operation = None
        if request.start_sync:
            sync_operation = self._workspace_service.create_sync_operation(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                source_id=source_id,
            )

        return CreateConnectedIndexedSourceResult(
            binding_id=lifecycle_result.binding.indexed_source_binding_id,
            source_id=source_id,
            configuration_revision=lifecycle_result.configuration_revision,
            mutation_result=_lifecycle_execution_result(
                lifecycle_result,
                binding_id=lifecycle_result.binding.indexed_source_binding_id,
            ),
            created_new_source=lifecycle_result.created_new_source,
            sync_operation=sync_operation,
        )
