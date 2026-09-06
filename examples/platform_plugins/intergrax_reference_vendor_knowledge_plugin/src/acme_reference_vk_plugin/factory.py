"""Tenant connection factory for the reference external provider."""

from __future__ import annotations

import json
from collections.abc import Mapping

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_contract import (
    EagerTenantConnectionIntegrationFactoryMixin,
)

from acme_reference_vk_plugin.backend import AcmeReferenceBackend
from acme_reference_vk_plugin.constants import ACME_REFERENCE_PROVIDER_ID
from acme_reference_vk_plugin.integration import (
    AcmeReferenceIntegrationConfig,
    AcmeReferenceWikiKnowledgeIntegration,
)


def _require_nonblank(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _parse_credentials(credential: str) -> str:
    try:
        parsed = json.loads(credential)
    except json.JSONDecodeError:
        raise ValueError("Acme reference credential payload is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError("Acme reference credential payload must be a JSON object")
    api_key = parsed.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Acme reference credential payload is missing api_key")
    return api_key.strip()


class AcmeReferenceTenantConnectionIntegrationFactory(
    EagerTenantConnectionIntegrationFactoryMixin,
):
    """Build reference integrations from secret-ref-backed credentials."""

    def create_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        credential: str,
        secret_free_config: Mapping[str, object],
    ) -> AcmeReferenceWikiKnowledgeIntegration:
        return self.create_integration_with_resolved_credential(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            resolved_credential=credential,
            secret_free_config=secret_free_config,
        )

    def create_integration_with_resolved_credential(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        resolved_credential: str,
        secret_free_config: Mapping[str, object],
    ) -> AcmeReferenceWikiKnowledgeIntegration:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        _require_nonblank(credential_ref, field_name="credential_ref")
        if provider_id != ACME_REFERENCE_PROVIDER_ID:
            raise ValueError("provider_id does not match acme_reference")
        if integration_kind is not IntegrationCategory.WIKI_KNOWLEDGE:
            raise ValueError("integration_kind does not match wiki_knowledge")
        if not isinstance(resolved_credential, str) or not resolved_credential.strip():
            raise ValueError("credential must be a nonblank string")

        endpoint = secret_free_config.get("collection_endpoint", "inmemory://collections")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("collection_endpoint must be a nonblank string")

        api_key = _parse_credentials(resolved_credential)
        config = AcmeReferenceIntegrationConfig(
            enabled=True,
            api_key=api_key,
            collection_endpoint=endpoint.strip(),
        )
        config.validate_for_runtime()
        return AcmeReferenceWikiKnowledgeIntegration.from_backend(
            AcmeReferenceBackend(),
            config=config,
        )
