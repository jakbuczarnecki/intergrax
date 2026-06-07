# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.index_lifecycle_contracts import (
    RagCheckIndexStatusInput,
    RagCheckIndexStatusOutput,
    RagGetDocumentInput,
    RagGetDocumentOutput,
    RagListDocumentsInput,
    RagListDocumentsOutput,
)
from intergrax.tools.providers.rag.index_lifecycle_service import (
    perform_rag_check_index_status,
    perform_rag_get_document,
    perform_rag_list_documents,
)


class RagListDocumentsHandler(ServiceToolHandler[RagListDocumentsInput, RagListDocumentsOutput]):
    _service = perform_rag_list_documents


class RagGetDocumentHandler(ServiceToolHandler[RagGetDocumentInput, RagGetDocumentOutput]):
    _service = perform_rag_get_document


class RagCheckIndexStatusHandler(ServiceToolHandler[RagCheckIndexStatusInput, RagCheckIndexStatusOutput]):
    _service = perform_rag_check_index_status
