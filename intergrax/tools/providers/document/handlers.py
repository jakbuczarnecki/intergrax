# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.document.contracts import DocumentParseInput, DocumentParseOutput
from intergrax.tools.providers.document.service import document_parse


class DocumentParseHandler(ServiceToolHandler[DocumentParseInput, DocumentParseOutput]):
    _service = document_parse
