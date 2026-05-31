# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.providers.document_parser.openpyxl.adapter import OpenpyxlDocumentParser
from intergrax.integrations.providers.document_parser.openpyxl.config import OpenpyxlIntegrationConfig


def create_openpyxl_document_parser(**config_overrides: object) -> DocumentParser:
    return OpenpyxlDocumentParser(OpenpyxlIntegrationConfig.from_env(**config_overrides))
