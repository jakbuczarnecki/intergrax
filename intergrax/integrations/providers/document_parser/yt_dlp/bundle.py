# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.yt_dlp.config import YtDlpIntegrationConfig
from intergrax.integrations.providers.document_parser.yt_dlp.manifest import MANIFEST
from intergrax.integrations.providers.document_parser.yt_dlp.opens import download_youtube_audio
from intergrax.integrations.registry.plugin_register import register_from_manifest


class YtDlpUrlDocumentParser:
    """Parses YouTube URLs into a single path fragment (download only; use whisper for transcription)."""

    def __init__(self, config: YtDlpIntegrationConfig) -> None:
        self._config = config

    def parser_id(self) -> str:
        return "yt_dlp"

    def is_available(self) -> bool:
        from intergrax.integrations.providers.document_parser.yt_dlp.opens import yt_dlp_is_available

        return yt_dlp_is_available()

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        if not (source.startswith("http://") or source.startswith("https://")):
            return []
        path = download_youtube_audio(self._config, source)
        return [
            ParsedDocumentFragment(
                text=str(path),
                metadata={"parser_backend": "yt_dlp", "downloaded_path": str(path), "source_url": source},
            )
        ]


def create_yt_dlp_document_parser(**config_overrides: object) -> DocumentParser:
    return YtDlpUrlDocumentParser(YtDlpIntegrationConfig.from_env(**config_overrides))

__all__ = ["create_yt_dlp_document_parser"
    "create_yt_dlp_document_parser_integration",
]

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_parser.yt_dlp.integration import (
    YT_DLP_DOCUMENT_PARSER_PROVIDER_ID,
    YtDlpDocumentParserIntegration,
    YtDlpDocumentParserIntegrationConfig,
    YtDlpDocumentParserClient,
)


def create_yt_dlp_document_parser_integration(
    *,
    client: YtDlpDocumentParserClient | None = None,
    enabled: bool = False,
) -> YtDlpDocumentParserIntegration:
    """
    Build a contract-based yt-dlp document parser integration.

    The legacy facade (create_yt_dlp_document_parser) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "yt-dlp document parser integration requires an injected client when enabled=True",
        )
    if client is not None:
        return YtDlpDocumentParserIntegration.from_client(client, enabled=enabled)
    return YtDlpDocumentParserIntegration.for_provider(
        provider_id=YT_DLP_DOCUMENT_PARSER_PROVIDER_ID,
        display_name="yt-dlp",
        config=YtDlpDocumentParserIntegrationConfig(enabled=enabled),
    )
