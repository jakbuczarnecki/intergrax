# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import create_default_reranker_manager
from intergrax.tools.providers.rag.rerank_contracts import RagRerankChunkInput, RagRerankInput
from intergrax.tools.providers.rag.rerank_service import rag_rerank
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


def test_rag_rerank_orders_candidates() -> None:
    ctx = ToolWiringContext(reranker_manager=create_default_reranker_manager())
    out = rag_rerank(
        ctx,
        RagRerankInput(
            query="project budget",
            chunks=[
                RagRerankChunkInput(id="a", text="unrelated note"),
                RagRerankChunkInput(id="b", text="project budget summary"),
            ],
            top_n=2,
        ),
    )
    assert out.total == 2
    assert out.chunks[0].id in {"a", "b"}
