# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.document_parser.whisper.bundle import create_whisper_document_parser
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_whisper_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.WHISPER.value,
            categories=(IntegrationCategory.DOCUMENT_PARSER,),
            factory=create_whisper_document_parser,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_WHISPER",
            description="Audio transcription via OpenAI Whisper (optional yt-dlp for YouTube URLs)",
        ),
        override=override,
    )
