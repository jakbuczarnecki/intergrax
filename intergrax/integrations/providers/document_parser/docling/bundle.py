# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.providers.document_parser.docling.adapter import DoclingDocumentParser
from intergrax.integrations.providers.document_parser.docling.config import DoclingIntegrationConfig


def create_docling_document_parser(**config_overrides: object) -> DocumentParser:
    config = DoclingIntegrationConfig.from_env(**config_overrides)
    return DoclingDocumentParser(config)
