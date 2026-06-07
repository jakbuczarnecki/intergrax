# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Sequence

import pytest

from intergrax.tools.providers.rag.lifecycle_contracts import RagDeleteDocumentsInput, RagDescribeCollectionInput
from intergrax.tools.providers.rag.lifecycle_service import perform_rag_delete_documents, perform_rag_describe_collection
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeVectorstoreManager:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self._count = 3

    def delete(self, ids: Sequence[str]) -> None:
        self.deleted.extend(list(ids))
        self._count = max(0, self._count - len(ids))

    def count(self) -> int:
        return self._count

    def list_collections(self) -> list[str]:
        return ["default", "archive"]


def test_rag_delete_documents() -> None:
    manager = FakeVectorstoreManager()
    ctx = ToolWiringContext(vectorstore_manager=manager)
    out = perform_rag_delete_documents(ctx, RagDeleteDocumentsInput(document_ids=["doc-1", "doc-2"]))
    assert out.used is True
    assert out.deleted_count == 2
    assert manager.deleted == ["doc-1", "doc-2"]


def test_rag_describe_collection() -> None:
    ctx = ToolWiringContext(vectorstore_manager=FakeVectorstoreManager())
    out = perform_rag_describe_collection(ctx, RagDescribeCollectionInput())
    assert out.used is True
    assert out.document_count == 3
    assert out.collections == ["default", "archive"]
    assert out.collection == "default"
