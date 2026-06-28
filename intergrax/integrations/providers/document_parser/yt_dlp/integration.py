# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""yt-dlp document parser integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

YT_DLP_DOCUMENT_PARSER_PROVIDER_ID = "yt_dlp"


class YtDlpDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for yt-dlp document parser integration."""

    pass


@runtime_checkable
class YtDlpDocumentParserClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class YtDlpDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    yt-dlp document parser integration.

    The legacy facade (create_yt_dlp_document_parser) remains separate and backward-compatible.
    """

    config: YtDlpDocumentParserIntegrationConfig = YtDlpDocumentParserIntegrationConfig()
    _client: YtDlpDocumentParserClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: YtDlpDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> YtDlpDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=YT_DLP_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="yt-dlp",
            config=YtDlpDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> YtDlpDocumentParserClient | None:
        return self._client
