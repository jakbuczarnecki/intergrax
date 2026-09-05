# © Artur Czarnecki. All rights reserved.

"""Provider-neutral remote resource discovery for connected workspace sources."""

from __future__ import annotations

from typing import Protocol

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceDiscoveryPageV1,
    RemoteResourceTypeV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.connected_source_discovery_strategy import (
    ConnectedSourceRevalidationLimits,
    RemoteResourceDiscoveryStrategyRegistry,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachmentStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.models import Workspace

from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError


class WorkspaceRemoteResourceDiscoveryPort(Protocol):
    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
        cursor: str | None,
        limit: int,
    ) -> RemoteResourceDiscoveryPageV1:
        ...


class WorkspaceKnowledgeWorkspaceLookupPort(Protocol):
    def require_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> Workspace | None:
        ...


class WorkspaceRemoteResourceDiscoveryService:
    def __init__(
        self,
        *,
        workspace_lookup: WorkspaceKnowledgeWorkspaceLookupPort,
        configuration_reader: WorkspaceKnowledgeConfigurationService,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
        strategy_registry: RemoteResourceDiscoveryStrategyRegistry,
        revalidation_limits: ConnectedSourceRevalidationLimits | None = None,
    ) -> None:
        self._workspace_lookup = workspace_lookup
        self._configuration_reader = configuration_reader
        self._codec = opaque_ref_codec
        self._strategy_registry = strategy_registry
        self._revalidation_limits = revalidation_limits or ConnectedSourceRevalidationLimits()

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
        cursor: str | None,
        limit: int,
    ) -> RemoteResourceDiscoveryPageV1:
        if limit < 1 or limit > 100:
            raise ConnectedSourceDiscoveryError("discovery_limit_invalid")

        self._require_attached_connection(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        strategy = self._strategy_registry.resolve(resource_type)

        provider_cursor: str | None = None
        if cursor is not None:
            provider_cursor = self._codec.decode_pagination_cursor(
                opaque_cursor=cursor,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=resource_type,
            )

        try:
            page = await strategy.list_remote_resources(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                provider_cursor=provider_cursor,
                limit=limit,
            )
        except VendorKnowledgeError as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc

        return RemoteResourceDiscoveryPageV1(
            items=page.items,
            next_cursor=self._codec.encode_pagination_cursor(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=resource_type,
                provider_cursor=page.provider_cursor,
            ),
        )

    def _require_attached_connection(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> None:
        workspace = self._workspace_lookup.require_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise ConnectedSourceDiscoveryError("workspace_not_found")
        configuration = self._configuration_reader.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise ConnectedSourceDiscoveryError("connection_not_attached")
        attachment = None
        for item in configuration.connection_attachments:
            if item.connection_ref == connection_ref:
                attachment = item
                break
        if attachment is None or attachment.status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
            raise ConnectedSourceDiscoveryError("connection_not_attached")

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
        opaque_candidate_ref: str,
    ) -> str:
        self._require_attached_connection(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        strategy = self._strategy_registry.resolve(resource_type)
        try:
            return await strategy.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                opaque_candidate_ref=opaque_candidate_ref,
                limits=self._revalidation_limits,
            )
        except VendorKnowledgeError as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc




