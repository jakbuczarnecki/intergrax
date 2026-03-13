# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

import pytest
from langchain_core.documents import Document

from intergrax.rag.indexing.indexing_manager import IndexingManager
from intergrax.rag.indexing.strategies.single_index_strategy import SingleIndexStrategy


pytestmark = pytest.mark.unit


class FakeEmbeddingManager:

    def embed_documents(self, documents):
        vectors = [[0.1, 0.2] for _ in documents]
        return vectors, documents


class FakeVectorstore:

    def __init__(self):
        self.docs = []

    def add_documents(self, documents, embeddings, batch_size=512):
        self.docs.extend(documents)

    def count(self):
        return len(self.docs)


def test_indexing_manager_indexes_documents():

    docs = [Document(page_content="doc")]

    embed = FakeEmbeddingManager()
    store = FakeVectorstore()

    manager = IndexingManager(
        embed_manager=embed,
        vectorstore=store,
        strategy=SingleIndexStrategy(),
    )

    manager.index_documents(docs)

    assert store.count() == 1