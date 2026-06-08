# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.contracts import RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.providers.rag.preview_service import rag_preview_retrieval


class RagPreviewRetrievalHandler(ServiceToolHandler[RagRetrieveInput, RagRetrieveOutput]):
    _service = rag_preview_retrieval
