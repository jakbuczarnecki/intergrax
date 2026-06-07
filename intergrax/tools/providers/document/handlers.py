# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.document.contracts import (
    DocumentParseInput,
    DocumentParseOutput,
    DocumentParsePreviewInput,
    DocumentParsePreviewOutput,
)
from intergrax.tools.providers.document.service import document_parse, document_parse_preview


class DocumentParseHandler(ServiceToolHandler[DocumentParseInput, DocumentParseOutput]):
    _service = document_parse


class DocumentParsePreviewHandler(ServiceToolHandler[DocumentParsePreviewInput, DocumentParsePreviewOutput]):
    _service = document_parse_preview
