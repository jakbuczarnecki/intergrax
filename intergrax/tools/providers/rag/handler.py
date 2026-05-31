# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.contracts import RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.providers.rag.service import perform_rag_retrieve


class RagRetrieveHandler(ServiceToolHandler[RagRetrieveInput, RagRetrieveOutput]):
    """Tool handler for ``rag.retrieve``."""

    _service = perform_rag_retrieve
