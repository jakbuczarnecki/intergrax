# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Openpyxl document parser integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID = "openpyxl"


class OpenpyxlDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Openpyxl document parser integration."""

    pass


OpenpyxlDocumentParserClient = DocumentParser

class OpenpyxlDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Single public Openpyxl document parser entrypoint.

    Legacy catalog factory (create_openpyxl_document_parser) owns catalog behavior; legacy factories use from_client().
    """

    config: OpenpyxlDocumentParserIntegrationConfig = OpenpyxlDocumentParserIntegrationConfig()
    _client: OpenpyxlDocumentParserClient | None = PrivateAttr(default=None)
    

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
        client: OpenpyxlDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> OpenpyxlDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Openpyxl",
            config=OpenpyxlDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OpenpyxlDocumentParserClient | None:
        return self._client

DocumentParser.register(OpenpyxlDocumentParserIntegration)
