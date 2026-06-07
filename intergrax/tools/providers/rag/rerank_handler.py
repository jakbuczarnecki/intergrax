# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.rerank_contracts import RagRerankInput, RagRerankOutput
from intergrax.tools.providers.rag.rerank_service import rag_rerank


class RagRerankHandler(ServiceToolHandler[RagRerankInput, RagRerankOutput]):
    _service = rag_rerank
