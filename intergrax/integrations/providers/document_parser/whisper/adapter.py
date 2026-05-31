# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.whisper.config import WhisperIntegrationConfig
from intergrax.integrations.providers.document_parser.whisper.opens import parse_whisper_audio


class WhisperDocumentParser:
    def __init__(self, config: WhisperIntegrationConfig) -> None:
        self._config = config

    def parser_id(self) -> str:
        return "whisper"

    def is_available(self) -> bool:
        from intergrax.integrations.providers.document_parser.whisper.opens import whisper_is_available

        return whisper_is_available()

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        return parse_whisper_audio(self._config, source)
