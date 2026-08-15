"""Provider-owned discovery for the reference external provider."""

from __future__ import annotations

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
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.vendor_knowledge_extension_composition import (
    VendorKnowledgeApplicationExtensionContext,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry

from acme_reference_vk_plugin.constants import (
    ACME_COLLECTION_SCOPE_TYPE,
    ACME_DOCUMENTS_SOURCE_KIND,
    ACME_REFERENCE_PROVIDER_ID,
)
from acme_reference_vk_plugin.integration import AcmeReferenceWikiKnowledgeIntegration


class AcmeReferenceCollectionDiscoveryStrategy:
    resource_type = RemoteResourceTypeV1.VENDOR_KNOWLEDGE_SCOPED_SOURCE

    def __init__(
        self,
        *,
        connection_registry: KnowledgeConnectionRegistry,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
    ) -> None:
        self._connection_registry = connection_registry
        self._codec = opaque_ref_codec

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
        integration = self._resolve_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        items = []
        for collection in integration.list_collections()[: max(1, limit)]:
            items.append(
                RemoteResourceCandidateV1(
                    opaque_candidate_ref=self._codec.encode_vendor_knowledge_scoped_source_candidate(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        connection_ref=connection_ref,
                        provider_id=ACME_REFERENCE_PROVIDER_ID,
                        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE.value,
                        source_kind=ACME_DOCUMENTS_SOURCE_KIND,
                        scope_id=collection.collection_id,
                        scope_type=ACME_COLLECTION_SCOPE_TYPE,
                        safe_display_label=collection.safe_display_label,
                    ),
                    resource_type=self.resource_type,
                    safe_display_label=collection.safe_display_label,
                    remote_resource_id=collection.collection_id,
                    safe_description="Acme reference document collection",
                )
            )
        return RemoteResourceStrategyPage(
            items=tuple(items),
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
        _ = workspace_id, limits
        payload = self._codec.decode_vendor_knowledge_scoped_source_candidate(
            opaque_candidate_ref
        )
        if (
            payload.tenant_id != tenant_id
            or payload.connection_ref != connection_ref
            or payload.provider_id != ACME_REFERENCE_PROVIDER_ID
            or payload.source_kind != ACME_DOCUMENTS_SOURCE_KIND
        ):
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
        integration = self._resolve_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        for item in integration.list_collections():
            if item.collection_id == payload.scope_id:
                return item.safe_display_label
        raise ConnectedSourceDiscoveryError("candidate_inaccessible")

    def _resolve_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> AcmeReferenceWikiKnowledgeIntegration:
        integration = self._connection_registry.resolve(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=ACME_REFERENCE_PROVIDER_ID,
            integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
        )
        if not isinstance(integration, AcmeReferenceWikiKnowledgeIntegration):
            raise ConnectedSourceDiscoveryError("connection_unavailable")
        return integration


def build_acme_reference_discovery_strategy(
    context: VendorKnowledgeApplicationExtensionContext,
) -> AcmeReferenceCollectionDiscoveryStrategy:
    return AcmeReferenceCollectionDiscoveryStrategy(
        connection_registry=context.connection_registry,
        opaque_ref_codec=context.opaque_ref_codec,
    )
