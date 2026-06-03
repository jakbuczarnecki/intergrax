# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``yt_dlp`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="yt_dlp",
    categories=(IntegrationCategory.DOCUMENT_PARSER,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_YT_DLP",
    description="YouTube audio download via yt-dlp (pair with whisper for transcription)",
)
