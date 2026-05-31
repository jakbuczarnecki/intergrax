# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.providers.document_parser.whisper.adapter import WhisperDocumentParser
from intergrax.integrations.providers.document_parser.whisper.config import WhisperIntegrationConfig


def create_whisper_document_parser(**config_overrides: object) -> DocumentParser:
    return WhisperDocumentParser(WhisperIntegrationConfig.from_env(**config_overrides))
