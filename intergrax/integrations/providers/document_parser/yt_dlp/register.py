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


def register_yt_dlp_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_yt_dlp_document_parser, override=override)
