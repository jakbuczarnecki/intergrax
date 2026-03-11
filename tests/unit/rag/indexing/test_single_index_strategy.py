# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

import pytest
from langchain_core.documents import Document

from intergrax.rag.indexing.strategies.single_index_strategy import SingleIndexStrategy


pytestmark = pytest.mark.unit


class FakeEmbeddingManager:

    def embed_documents(self, documents):
        vectors = [[0.1, 0.2, 0.3] for _ in documents]
        return vectors, documents


class FakeVectorstore:

    def __init__(self):
        self.docs = []

    def add_documents(self, documents, embeddings, batch_size=512):
        self.docs.extend(documents)

    def count(self):
        return len(self.docs)


def test_single_index_strategy_inserts_documents():

    docs = [
        Document(page_content="A"),
        Document(page_content="B"),
    ]

    embed_manager = FakeEmbeddingManager()
    vectorstore = FakeVectorstore()

    strategy = SingleIndexStrategy()

    strategy.build_index(
        documents=docs,
        embed_manager=embed_manager,
        vectorstore=vectorstore,
    )

    assert vectorstore.count() == 2