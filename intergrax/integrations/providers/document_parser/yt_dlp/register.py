# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register yt_dlp."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.yt_dlp.bundle import create_yt_dlp_document_parser
from intergrax.integrations.providers.document_parser.yt_dlp.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_yt_dlp_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_yt_dlp_document_parser, override=override)
