# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-13 — ToolCatalogEmbedder acceptance tests."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest
from langchain_core.documents import Document
from numpy.typing import NDArray
from pydantic import BaseModel

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.tools.tool_catalog_embedder import ToolCatalogEmbedder
from intergrax.runtime.nexus.tools.tool_selection import (
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit

_DIM = 32


class _In(BaseModel):
    query: str = ""


class _Out(BaseModel):
    result: str = ""


class BagOfWordsEmbeddingManager(BaseEmbeddingManager):
    """Deterministic bag-of-words vectors for rank tests."""

    def embed_one(self, text: str) -> NDArray[np.float32]:
        return _bag_vector(text)

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.stack([_bag_vector(text) for text in texts])

    def embed_documents(self, documents: Sequence[Document]) -> EmbeddingResult:
        vectors = [list(_bag_vector(doc.page_content)) for doc in documents]
        return EmbeddingResult(vectors=vectors, model_name="test")


def _bag_vector(text: str) -> NDArray[np.float32]:
    vec = np.zeros(_DIM, dtype=np.float32)
    for token in text.lower().split():
        vec[hash(token) % _DIM] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    specs = (
        ("jira.search_tasks", "Search Jira issues and tasks in the issue tracker"),
        ("notify.send", "Send email notification to users"),
        ("rag.retrieve", "Retrieve documents from vector index"),
    )
    for tool_id, description in specs:
        contract = ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=description,
            input_schema=_In,
            output_schema=_Out,
            error_mapping={},
            side_effects=False,
        )
        registry.register(
            contract,
            type("H", (), {"execute": lambda self, request: _Out(result="ok")})(),
        )
    return registry


def test_semantic_rank_prefers_jira_for_jira_query() -> None:
    registry = _registry()
    embedder = BagOfWordsEmbeddingManager()
    ranks = ToolCatalogEmbedder(embedder).search_registry(
        registry,
        "search jira tasks in issue tracker",
        top_k=2,
    )
    assert ranks
    assert ranks[0].tool_id == "jira.search_tasks"


def test_semantic_selection_strategy_via_resolver() -> None:
    registry = _registry()
    embedder = BagOfWordsEmbeddingManager()
    ids = resolve_planner_allowed_tool_ids(
        ToolSelectionMode.SEMANTIC,
        ToolSelectionContext(
            registry=registry,
            query="search jira tasks",
            top_k=1,
            embedding_manager=embedder,
        ),
    )
    assert ids == ("jira.search_tasks",)
