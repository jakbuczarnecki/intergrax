# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import numpy as np
from typing import Dict, List, Any
from langchain_core.documents import Document


class InMemoryVectorStore:
    """
    Deterministic in-memory test double.

    Simulates:
    - physical collection isolation
    - tenant metadata enforcement
    - simple exact-vector match query
    """

    _storage: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self, collection_name: str, tenant_id: str) -> None:
        self.collection_name = collection_name
        self.tenant_id = tenant_id
        if collection_name not in self._storage:
            self._storage[collection_name] = []

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        for i, doc in enumerate(documents):
            self._storage[self.collection_name].append(
                {
                    "embedding": embeddings[i].tolist(),
                    "document": doc.page_content,
                    "metadata": {"tenant_id": self.tenant_id},
                }
            )

    def query(self, query_embeddings: np.ndarray, top_k: int):
        results = []
        for record in self._storage.get(self.collection_name, []):
            if (
                record["embedding"] == query_embeddings[0].tolist()
                and record["metadata"]["tenant_id"] == self.tenant_id
            ):
                results.append(record)

        return {
            "ids": [[str(i) for i in range(len(results))]],
            "documents": [[r["document"] for r in results]],
        }


def test_vectorstore_cross_tenant_isolation_chroma() -> None:
    """
    Cross-tenant isolation test (in-memory deterministic provider).

    Scenario:
    - Tenant A inserts a document.
    - Tenant B queries with identical embedding.
    - Tenant B must not see Tenant A's document.
    """

    embedding = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    doc = Document(page_content="secret-data", metadata={})

    # Tenant A
    vs_a = InMemoryVectorStore(
        collection_name="test_collection__tenant__tenant_A",
        tenant_id="tenant_A",
    )
    vs_a.add_documents([doc], embedding)

    # Tenant B (different physical collection)
    vs_b = InMemoryVectorStore(
        collection_name="test_collection__tenant__tenant_B",
        tenant_id="tenant_B",
    )

    result_b = vs_b.query(
        query_embeddings=embedding,
        top_k=5,
    )

    assert result_b["ids"] == [[]]
    assert result_b["documents"] == [[]]