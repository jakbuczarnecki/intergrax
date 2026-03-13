# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional

from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.providers.inmemory_vectorstore import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


def create_default_vectorstore_manager(*, tenant_id: Optional[str] = None)-> BaseVectorstoreManager:
    if tenant_id is None:
        tenant_id = "in_memory_tenant_id"

    manager = VectorstoreManager(store=InMemoryVectorStore(tenant_id=tenant_id))
    return manager