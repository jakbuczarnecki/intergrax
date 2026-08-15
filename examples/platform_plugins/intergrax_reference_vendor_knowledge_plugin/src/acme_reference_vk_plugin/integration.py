"""Reference wiki-knowledge integration contract."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationCategory

from acme_reference_vk_plugin.backend import AcmeReferenceBackend, AcmeReferenceDocument
from acme_reference_vk_plugin.constants import ACME_REFERENCE_PROVIDER_ID


@dataclass(frozen=True, slots=True)
class AcmeReferenceIntegrationConfig:
    enabled: bool
    api_key: str
    collection_endpoint: str

    def validate_for_runtime(self) -> None:
        if not self.enabled:
            raise ValueError("integration disabled")
        if not self.api_key.strip():
            raise ValueError("api_key required")
        if not self.collection_endpoint.strip():
            raise ValueError("collection_endpoint required")


class AcmeReferenceWikiKnowledgeIntegration:
    """Secret-free integration surface backed by a bounded in-memory store."""

    provider_id = ACME_REFERENCE_PROVIDER_ID
    integration_kind = IntegrationCategory.WIKI_KNOWLEDGE

    def __init__(
        self,
        *,
        config: AcmeReferenceIntegrationConfig,
        backend: AcmeReferenceBackend,
    ) -> None:
        self._config = config
        self._backend = backend

    @classmethod
    def from_backend(
        cls,
        backend: AcmeReferenceBackend,
        *,
        enabled: bool = True,
        config: AcmeReferenceIntegrationConfig | None = None,
    ) -> AcmeReferenceWikiKnowledgeIntegration:
        resolved = config or AcmeReferenceIntegrationConfig(
            enabled=enabled,
            api_key="reference-api-key",
            collection_endpoint="inmemory://collections",
        )
        resolved.validate_for_runtime()
        return cls(config=resolved, backend=backend)

    def list_documents(
        self,
        *,
        collection_id: str,
    ) -> tuple[AcmeReferenceDocument, ...]:
        return self._backend.list_documents(collection_id=collection_id)

    def get_document(
        self,
        *,
        collection_id: str,
        remote_id: str,
    ) -> AcmeReferenceDocument | None:
        return self._backend.get_document(
            collection_id=collection_id,
            remote_id=remote_id,
        )

    def list_collections(self):
        return self._backend.list_collections()
