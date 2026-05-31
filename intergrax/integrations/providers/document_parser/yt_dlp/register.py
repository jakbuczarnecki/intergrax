# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.yt_dlp.config import YtDlpIntegrationConfig
from intergrax.integrations.providers.document_parser.yt_dlp.opens import download_youtube_audio
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


class YtDlpUrlDocumentParser:
    """Parses YouTube URLs into a single path fragment (download only; use whisper for transcription)."""

    def __init__(self, config: YtDlpIntegrationConfig) -> None:
        self._config = config

    def parser_id(self) -> str:
        return "yt_dlp"

    def is_available(self) -> bool:
        try:
            import yt_dlp  # noqa: F401

            return True
        except Exception:
            return False

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
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.YT_DLP.value,
            categories=(IntegrationCategory.DOCUMENT_PARSER,),
            factory=create_yt_dlp_document_parser,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_YT_DLP",
            description="YouTube audio download via yt-dlp (pair with whisper for transcription)",
        ),
        override=override,
    )
