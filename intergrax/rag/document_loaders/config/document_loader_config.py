# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.integrations.providers.document_parser.docling.config import DoclingIntegrationConfig, DoclingMode

# Backward-compatible aliases for legacy RAG imports.
DoclingMode = DoclingMode


class DocumentLoaderConfig:
    """Shim — prefer ``DoclingIntegrationConfig`` and ``IntegrationProfile.document_parser``."""

    def __init__(self) -> None:
        integration = DoclingIntegrationConfig.from_env()
        self.docling_mode = integration.mode
        self.docling_simple_pdf_mode = integration.simple_pdf_mode
        self.docling_server_url = integration.server_url
        self.docling_server_timeout_seconds = integration.timeout_seconds
        self.default_builtin_handler_confidence = 0.8


GLOBAL_DOCUMENT_LOADER_CONFIG = DocumentLoaderConfig()
