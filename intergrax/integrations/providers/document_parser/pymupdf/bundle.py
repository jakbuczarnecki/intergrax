# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.providers.document_parser.pymupdf.adapter import PymupdfDocumentParser
from intergrax.integrations.providers.document_parser.pymupdf.config import PymupdfIntegrationConfig


def create_pymupdf_document_parser(**config_overrides: object) -> DocumentParser:
    return PymupdfDocumentParser(PymupdfIntegrationConfig.from_env(**config_overrides))
