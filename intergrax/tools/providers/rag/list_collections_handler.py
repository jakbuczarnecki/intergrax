# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.list_collections_contracts import RagListCollectionsInput
from intergrax.tools.providers.rag.list_collections_service import perform_rag_list_collections


class RagListCollectionsHandler(ServiceToolHandler[RagListCollectionsInput, object]):
    _service = perform_rag_list_collections
