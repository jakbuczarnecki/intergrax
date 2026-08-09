# © Artur Czarnecki. All rights reserved.

"""Exact-resource Google Workspace discovery behind the LKW boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import RLock

from local_workspace_application.workspaces.connected_source_discovery_strategy import (
    ConnectedSourceRevalidationLimits,
    RemoteResourceStrategyPage,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceCandidateV1,
    RemoteResourceTypeV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    GoogleWorkspaceCandidatePayload,
    RemoteResourceOpaqueRefCodec,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError


@dataclass(frozen=True, slots=True)
class GoogleWorkspaceKnownResource:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    resource_type: RemoteResourceTypeV1
    remote_resource_id: str
    safe_display_label: str


class GoogleWorkspaceKnownResourceCatalog:
    """Host-owned exact resource configuration; never performs discovery."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._resources: dict[
            tuple[str, str, str, RemoteResourceTypeV1],
            GoogleWorkspaceKnownResource,
        ] = {}

    def register(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
        remote_resource_id: str,
        safe_display_label: str,
    ) -> GoogleWorkspaceKnownResource:
        values = (
            tenant_id,
            workspace_id,
            connection_ref,
            remote_resource_id,
            safe_display_label,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("google_workspace_known_resource_invalid")
        if resource_type not in {
            RemoteResourceTypeV1.GOOGLE_WORKSPACE_CALENDAR,
            RemoteResourceTypeV1.GOOGLE_WORKSPACE_DOCS,
            RemoteResourceTypeV1.GOOGLE_WORKSPACE_SHEETS,
        }:
            raise ValueError("google_workspace_resource_type_invalid")
        resource = GoogleWorkspaceKnownResource(
            tenant_id=tenant_id.strip(),
            workspace_id=workspace_id.strip(),
            connection_ref=connection_ref.strip(),
            resource_type=resource_type,
            remote_resource_id=remote_resource_id.strip(),
            safe_display_label=safe_display_label.strip(),
        )
        key = (
            resource.tenant_id,
            resource.workspace_id,
            resource.connection_ref,
            resource.resource_type,
        )
        with self._lock:
            self._resources[key] = resource
        return resource

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
    ) -> GoogleWorkspaceKnownResource | None:
        key = (
            tenant_id.strip(),
            workspace_id.strip(),
            connection_ref.strip(),
            resource_type,
        )
        with self._lock:
            return self._resources.get(key)


class GoogleWorkspaceKnownResourceDiscoveryStrategy:
    """Resolve one configured Google resource and never broaden its scope."""

    def __init__(
        self,
        *,
        connection_registry: KnowledgeConnectionRegistry,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
        known_resources: GoogleWorkspaceKnownResourceCatalog,
        resource_type: RemoteResourceTypeV1,
        safe_description: str,
    ) -> None:
        self._connection_registry = connection_registry
        self._codec = opaque_ref_codec
        self._known_resources = known_resources
        self.resource_type = resource_type
        self._safe_description = safe_description

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        provider_cursor: str | None,
        limit: int,
    ) -> RemoteResourceStrategyPage:
        if provider_cursor is not None:
            return RemoteResourceStrategyPage(items=(), provider_cursor=None)
        known = self._known_resources.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=self.resource_type,
        )
        if known is None:
            return RemoteResourceStrategyPage(items=(), provider_cursor=None)
        label = await self._resolve_label(known)
        return RemoteResourceStrategyPage(
            items=(
                RemoteResourceCandidateV1(
                    opaque_candidate_ref=self._codec.encode_google_workspace_candidate(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        connection_ref=connection_ref,
                        resource_type=self.resource_type,
                        remote_resource_id=known.remote_resource_id,
                        safe_display_label=label,
                    ),
                    resource_type=self.resource_type,
                    safe_display_label=label,
                    remote_resource_id=known.remote_resource_id,
                    safe_description=self._safe_description,
                ),
            )[:limit],
            provider_cursor=None,
        )

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        limits: ConnectedSourceRevalidationLimits,
    ) -> str:
        _ = limits
        payload = self._codec.decode_google_workspace_candidate(opaque_candidate_ref)
        self._validate_payload(
            payload,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        known = GoogleWorkspaceKnownResource(
            tenant_id=payload.tenant_id,
            workspace_id=payload.workspace_id,
            connection_ref=payload.connection_ref,
            resource_type=payload.resource_type,
            remote_resource_id=payload.remote_resource_id,
            safe_display_label=payload.safe_display_label,
        )
        return await self._resolve_label(known)

    async def _resolve_label(self, known: GoogleWorkspaceKnownResource) -> str:
        integration = self._resolve_integration(
            tenant_id=known.tenant_id,
            connection_ref=known.connection_ref,
        )
        try:
            if known.resource_type is RemoteResourceTypeV1.GOOGLE_WORKSPACE_CALENDAR:
                page = await asyncio.to_thread(
                    integration.list_calendar_events_page,
                    calendar_id=known.remote_resource_id,
                    max_results=1,
                )
                label = page.summary
            elif known.resource_type is RemoteResourceTypeV1.GOOGLE_WORKSPACE_DOCS:
                document = await asyncio.to_thread(
                    integration.read_docs_document,
                    document_id=known.remote_resource_id,
                )
                label = document.title
            else:
                spreadsheet = await asyncio.to_thread(
                    integration.read_sheets_spreadsheet,
                    spreadsheet_id=known.remote_resource_id,
                )
                label = spreadsheet.title
        except Exception as exc:
            raise ConnectedSourceDiscoveryError("candidate_inaccessible") from exc
        return label.strip() if isinstance(label, str) and label.strip() else known.safe_display_label

    def _resolve_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        try:
            integration = self._connection_registry.resolve(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            )
        except (VendorKnowledgeError, ValueError) as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
        if not isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration):
            raise ConnectedSourceDiscoveryError("connection_incompatible")
        return integration

    def _validate_payload(
        self,
        payload: GoogleWorkspaceCandidatePayload,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> None:
        if (
            payload.resource_type is not self.resource_type
            or payload.tenant_id != tenant_id
            or payload.workspace_id != workspace_id
            or payload.connection_ref != connection_ref
        ):
            raise ConnectedSourceDiscoveryError("candidate_inaccessible")
