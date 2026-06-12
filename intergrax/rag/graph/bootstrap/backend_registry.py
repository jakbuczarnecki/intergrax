# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RAG GraphStore backend registry (M-RAG.38) — mirror vectorstore bootstrap pattern."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.rag.graph.contracts.graph_store import GraphStore

GraphStoreFactory = Callable[..., GraphStore]

_REGISTRY: dict[str, GraphStoreFactory] = {}


def register_graph_store_backend(backend_id: str, factory: GraphStoreFactory) -> None:
    """Register a factory for ``backend_id`` (lowercase slug)."""
    key = backend_id.strip().lower()
    if not key:
        raise ValueError("backend_id must be non-empty")
    _REGISTRY[key] = factory


def resolve_graph_store_backend(backend_id: str) -> Optional[GraphStoreFactory]:
    return _REGISTRY.get(backend_id.strip().lower())


def list_graph_store_backends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))


def create_graph_store_from_registry(
    backend_id: str,
    *,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    factory = resolve_graph_store_backend(backend_id)
    if factory is None:
        known = ", ".join(list_graph_store_backends()) or "(none)"
        raise ValueError(f"unknown_rag_graph_store_backend:{backend_id}; known={known}")
    return factory(
        integration_graph_store=integration_graph_store,
        tenant_id=tenant_id,
    )
