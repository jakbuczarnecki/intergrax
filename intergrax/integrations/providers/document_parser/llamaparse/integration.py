# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Llamaparse document parser integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID = "llamaparse"


class LlamaparseDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Llamaparse document parser integration."""

    pass


LlamaparseDocumentParserClient = DocumentParser

class LlamaparseDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Single public Llamaparse document parser entrypoint.

    Legacy catalog factory (create_llamaparse_document_parser) owns catalog behavior; legacy factories use from_client().
    """

    config: LlamaparseDocumentParserIntegrationConfig = LlamaparseDocumentParserIntegrationConfig()
    _client: LlamaparseDocumentParserClient | None = PrivateAttr(default=None)
    

    def is_available(self):
        return self._require_client().is_available()

    def parse_file(self, source):
        return self._require_client().parse_file(source)

    def parser_id(self):
        return self._require_client().parser_id()

    def _require_client(self) -> DocumentParser:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: LlamaparseDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> LlamaparseDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Llamaparse",
            config=LlamaparseDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LlamaparseDocumentParserClient | None:
        return self._client

DocumentParser.register(LlamaparseDocumentParserIntegration)
