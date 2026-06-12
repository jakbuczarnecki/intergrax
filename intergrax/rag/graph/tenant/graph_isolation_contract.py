# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cross-tenant isolation contract for document knowledge graphs (M-RAG.41)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document

from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer

GRAPH_ISOLATION_CONTRACT_BACKENDS: tuple[str, ...] = (
    "inmemory",
    "neo4j",
    "memgraph",
    "falkordb",
)

GraphStoreFactory = Callable[[str], GraphStore]


@dataclass(frozen=True)
class GraphIsolationContractResult:
    slug: str
    cross_query_isolated: bool
    reason: str = ""


def run_graph_isolation_contract(
    factory: GraphStoreFactory,
    *,
    slug: str,
    tenant_a: str = "tenant_graph_A",
    tenant_b: str = "tenant_graph_B",
) -> GraphIsolationContractResult:
    """
    Index a secret entity under tenant A; tenant B graph queries must not see it.
    """
    store_a = factory(tenant_a)
    store_b = factory(tenant_b)
    indexer = HeuristicGraphIndexer(store_a)
    secret_doc = Document(
        page_content="Acme Corp signed a secret partnership with Intergrax Harness.",
        metadata={"tenant_id": tenant_a},
    )
    indexer.index_documents([secret_doc], chunk_ids=["chunk-secret-graph"])

    leaked = store_b.find_nodes(label_contains="Acme", limit=5)
    if leaked:
        return GraphIsolationContractResult(
            slug=slug,
            cross_query_isolated=False,
            reason="tenant_b_leaked_tenant_a_graph_nodes",
        )

    chunk_ids = store_b.chunk_ids_for_nodes({node.id for node in leaked})
    if chunk_ids:
        return GraphIsolationContractResult(
            slug=slug,
            cross_query_isolated=False,
            reason="tenant_b_leaked_tenant_a_chunk_links",
        )

    return GraphIsolationContractResult(
        slug=slug,
        cross_query_isolated=True,
        reason="ok",
    )
