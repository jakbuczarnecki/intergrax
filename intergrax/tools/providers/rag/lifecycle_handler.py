# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.lifecycle_contracts import (
    RagDeleteDocumentsInput,
    RagDeleteDocumentsOutput,
    RagDescribeCollectionInput,
    RagDescribeCollectionOutput,
)
from intergrax.tools.providers.rag.lifecycle_service import (
    perform_rag_delete_documents,
    perform_rag_describe_collection,
)


class RagDeleteDocumentsHandler(ServiceToolHandler[RagDeleteDocumentsInput, RagDeleteDocumentsOutput]):
    _service = perform_rag_delete_documents


class RagDescribeCollectionHandler(ServiceToolHandler[RagDescribeCollectionInput, RagDescribeCollectionOutput]):
    _service = perform_rag_describe_collection
