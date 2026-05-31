# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.tools.providers.rag.list_collections_contracts import RagListCollectionsInput
from intergrax.tools.providers.rag.list_collections_service import perform_rag_list_collections
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_rag_list_collections_returns_inmemory_name() -> None:
    ctx = ToolWiringContext(
        vectorstore_manager=VectorstoreManager(InMemoryVectorStore(tenant_id="lab")),
    )
    out = perform_rag_list_collections(ctx, RagListCollectionsInput())
    assert out.used is True
    assert out.collections == ["inmemory:lab"]
