# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Deprecated — use ``intergrax.integrations.providers.document_parser.docling.opens``."""

from __future__ import annotations

from intergrax.integrations.providers.document_parser.docling.config import DoclingIntegrationConfig
from intergrax.integrations.providers.document_parser.docling.opens import open_docling_local_converter


def create_docling_converter():
    """Backward-compatible shim for legacy imports."""
    config = DoclingIntegrationConfig.from_env()
    return open_docling_local_converter(config)
