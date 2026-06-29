# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pymupdf document parser integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID = "pymupdf"


class PymupdfDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pymupdf document parser integration."""

    pass


PymupdfDocumentParserClient = DocumentParser

class PymupdfDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Single public Pymupdf document parser entrypoint.

    Legacy catalog factory (create_pymupdf_document_parser) owns catalog behavior; legacy factories use from_client().
    """

    config: PymupdfDocumentParserIntegrationConfig = PymupdfDocumentParserIntegrationConfig()
    _client: PymupdfDocumentParserClient | None = PrivateAttr(default=None)
    

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
        client: PymupdfDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> PymupdfDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Pymupdf",
            config=PymupdfDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PymupdfDocumentParserClient | None:
        return self._client

DocumentParser.register(PymupdfDocumentParserIntegration)
