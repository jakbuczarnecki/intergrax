# © Artur Czarnecki. All rights reserved.

"""Semantic tool catalog index (TOOL-ENG-13 · ADR-TOOL-004)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.runtime import RegisteredTool

HARNESS_TOOL_CATALOG_COLLECTION = "__harness_tool_catalog__"


@dataclass(frozen=True, slots=True)
class ToolCatalogRank:
    tool_id: str
    score: float


@dataclass(slots=True)
class ToolCatalogIndex:
    """In-memory cosine index for tool metadata embeddings."""

    entries: list[tuple[str, NDArray[np.float32]]]

    def search_query(
        self,
        query_vector: NDArray[np.float32],
        *,
        top_k: int,
        allowed_tool_ids: frozenset[str] | None = None,
    ) -> list[ToolCatalogRank]:
        if not self.entries:
            return []
        query = _normalize(query_vector)
        scored: list[ToolCatalogRank] = []
        for tool_id, vector in self.entries:
            if allowed_tool_ids is not None and tool_id not in allowed_tool_ids:
                continue
            score = float(np.dot(query, _normalize(vector)))
            scored.append(ToolCatalogRank(tool_id=tool_id, score=score))
        scored.sort(key=lambda item: (-item.score, item.tool_id))
        return scored[: max(1, top_k)]


class ToolCatalogEmbedder:
    """Builds and caches semantic indexes over runtime tool registries."""

    def __init__(self, embedding_manager: BaseEmbeddingManager) -> None:
        self._embedding_manager = embedding_manager
        self._cache: dict[tuple[str, ...], ToolCatalogIndex] = {}

    def index_for_registry(self, registry: ToolRegistry) -> ToolCatalogIndex:
        fingerprint = tuple(sorted(registry.tool_ids()))
        cached = self._cache.get(fingerprint)
        if cached is not None:
            return cached
        entries: list[tuple[str, NDArray[np.float32]]] = []
        for registered in registry.list():
            text = contract_embedding_text(registered.contract)
            vector = self._embedding_manager.embed_one(text)
            entries.append((registered.contract.tool_id, np.asarray(vector, dtype=np.float32)))
        index = ToolCatalogIndex(entries=entries)
        self._cache[fingerprint] = index
        return index

    def search_registry(
        self,
        registry: ToolRegistry,
        query: str,
        *,
        top_k: int,
        allowed_tool_ids: Sequence[str] | None = None,
    ) -> list[ToolCatalogRank]:
        index = self.index_for_registry(registry)
        query_vector = self._embedding_manager.embed_one(query)
        allowed = frozenset(allowed_tool_ids) if allowed_tool_ids is not None else None
        present = frozenset(registry.tool_ids())
        if allowed is not None:
            allowed = frozenset(tool_id for tool_id in allowed if tool_id in present)
        else:
            allowed = present
        return index.search_query(
            np.asarray(query_vector, dtype=np.float32),
            top_k=top_k,
            allowed_tool_ids=allowed,
        )


def contract_embedding_text(contract: ToolContract) -> str:
    return " ".join(
        part
        for part in (
            contract.tool_id,
            contract.name,
            contract.description,
            contract.description_short or "",
            contract.category,
            " ".join(contract.tags),
        )
        if part
    ).strip()


def _normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector
    return vector / norm
