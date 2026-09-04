# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenAI managed retrieval integration (canonical catalog registration)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.managed_retrieval import (
    ManagedRetrievalBackend,
    ManagedRetrievalQueryRequest,
    ManagedRetrievalUploadResult,
)
from intergrax.runtime.integrations.categories.managed_retrieval import (
    ManagedRetrievalIntegrationContract,
)
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

OPENAI_MANAGED_RETRIEVAL_PROVIDER_ID = "openai"


class OpenAIManagedRetrievalIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for OpenAI managed retrieval integration."""

    pass


OpenAIManagedRetrievalClient = ManagedRetrievalBackend


class OpenAIManagedRetrievalIntegration(ManagedRetrievalIntegrationContract):
    """Single public OpenAI managed retrieval entrypoint."""

    config: OpenAIManagedRetrievalIntegrationConfig = OpenAIManagedRetrievalIntegrationConfig()
    _client: OpenAIManagedRetrievalClient | None = PrivateAttr(default=None)

    def ensure_store_exists(self, store_id: str) -> None:
        self._require_client().ensure_store_exists(store_id)

    def list_attached_file_ids(self, store_id: str) -> Sequence[str]:
        return self._require_client().list_attached_file_ids(store_id)

    def upload_folder(
        self,
        store_id: str,
        folder: str | Path,
        *,
        patterns: Sequence[str],
    ) -> ManagedRetrievalUploadResult:
        return self._require_client().upload_folder(store_id, folder, patterns=patterns)

    def clear_store(self, store_id: str) -> int:
        return self._require_client().clear_store(store_id)

    def query(self, request: ManagedRetrievalQueryRequest) -> str:
        return self._require_client().query(request)

    def _require_client(self) -> ManagedRetrievalBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client

    @classmethod
    def from_client(
        cls,
        client: OpenAIManagedRetrievalClient,
        *,
        enabled: bool = False,
    ) -> OpenAIManagedRetrievalIntegration:
        integration = cls.for_provider(
            provider_id=OPENAI_MANAGED_RETRIEVAL_PROVIDER_ID,
            display_name="OpenAI",
            config=OpenAIManagedRetrievalIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OpenAIManagedRetrievalClient | None:
        return self._client


ManagedRetrievalBackend.register(OpenAIManagedRetrievalIntegration)
