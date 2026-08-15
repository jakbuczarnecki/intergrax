# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Native Graph RAG batch validation shared by graph indexers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.graph.contracts.graph_store import GraphStore


@dataclass(frozen=True)
class GraphDocumentScope:
    """Logical scope that must remain isolated inside one graph boundary."""

    tenant_id: str
    namespace: str | None
    workspace_id: str | None


class GraphScopeFence:
    """Bind a graph boundary to its first fully validated document scope."""

    def __init__(self) -> None:
        self._bound_scope: GraphDocumentScope | None = None

    @property
    def bound_scope(self) -> GraphDocumentScope | None:
        return self._bound_scope

    def bind(self, scope: GraphDocumentScope) -> None:
        if self._bound_scope is not None and self._bound_scope != scope:
            raise ValueError(
                "graph scope cannot change after binding: "
                "tenant_id, namespace and workspace_id must remain identical"
            )
        self._bound_scope = scope


_STORE_SCOPE_FENCES: WeakKeyDictionary[object, GraphScopeFence] = WeakKeyDictionary()
_NON_WEAKREF_STORE_SCOPE_FENCES: dict[int, tuple[object, GraphScopeFence]] = {}


def _scope_fence_for_store(store: GraphStore) -> GraphScopeFence:
    try:
        fence = _STORE_SCOPE_FENCES.get(store)
        if fence is None:
            fence = GraphScopeFence()
            _STORE_SCOPE_FENCES[store] = fence
        return fence
    except TypeError:
        key = id(store)
        existing = _NON_WEAKREF_STORE_SCOPE_FENCES.get(key)
        if existing is not None and existing[0] is store:
            return existing[1]
        fence = GraphScopeFence()
        _NON_WEAKREF_STORE_SCOPE_FENCES[key] = (store, fence)
        return fence


def validate_graph_index_batch(
    store: GraphStore,
    documents: Sequence[KnowledgeDocument],
    chunk_ids: Sequence[str] | None,
    *,
    scope_fence: GraphScopeFence | None = None,
) -> tuple[tuple[KnowledgeDocument, ...], tuple[str, ...]]:
    """Validate the complete native batch before any GraphStore write."""
    materialized = tuple(documents)
    if not materialized:
        return (), ()

    validated: list[KnowledgeDocument] = []
    for document in materialized:
        if not isinstance(document, KnowledgeDocument):
            raise TypeError("documents must contain only KnowledgeDocument values")
        try:
            validated.append(
                KnowledgeDocument.model_validate(document.model_dump(mode="python"))
            )
        except Exception as exc:
            raise ValueError("document failed full revalidation") from exc

    first_scope = validated[0].scope
    if any(
        document.scope.tenant_id != first_scope.tenant_id
        or document.scope.namespace != first_scope.namespace
        or document.scope.workspace_id != first_scope.workspace_id
        for document in validated[1:]
    ):
        raise ValueError(
            "documents must share the same tenant and namespace and workspace"
        )

    store_tenant = store.tenant_id
    if store_tenant is not None and first_scope.tenant_id != store_tenant:
        raise ValueError("document tenant_id differs from bound graph store")

    if chunk_ids is None:
        resolved_ids = tuple(document.identity.document_id for document in validated)
    else:
        if len(chunk_ids) != len(validated):
            raise ValueError("chunk_ids length must match documents length")
        resolved_ids = tuple(chunk_ids)

    if any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in resolved_ids):
        raise ValueError("chunk_ids must contain only non-empty strings")
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("chunk_ids must be unique")

    effective_fence = scope_fence or _scope_fence_for_store(store)
    effective_fence.bind(
        GraphDocumentScope(
            tenant_id=first_scope.tenant_id,
            namespace=first_scope.namespace,
            workspace_id=first_scope.workspace_id,
        )
    )

    return tuple(validated), resolved_ids
