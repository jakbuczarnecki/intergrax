# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Yt Dlp document parser integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

YT_DLP_DOCUMENT_PARSER_PROVIDER_ID = "yt_dlp"


class YtDlpDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Yt Dlp document parser integration."""

    pass


YtDlpDocumentParserClient = DocumentParser

class YtDlpDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Single public Yt Dlp document parser entrypoint.

    Legacy catalog factory (create_yt_dlp_document_parser) owns catalog behavior; legacy factories use from_client().
    """

    config: YtDlpDocumentParserIntegrationConfig = YtDlpDocumentParserIntegrationConfig()
    _client: YtDlpDocumentParserClient | None = PrivateAttr(default=None)
    

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
        client: YtDlpDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> YtDlpDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=YT_DLP_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Yt Dlp",
            config=YtDlpDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> YtDlpDocumentParserClient | None:
        return self._client

DocumentParser.register(YtDlpDocumentParserIntegration)
