# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Graph-store production SLO soak contract (M-RAG.55)."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Sequence

from intergrax.integrations.contracts.base import IntegrationStatus
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore

STABLE_GRAPH_SOAK_SLUGS: tuple[str, ...] = ("neo4j",)

GRAPH_BETA_PROMOTION_SLUGS: tuple[str, ...] = ("memgraph", "falkordb")

_SLUG_MANIFEST_MODULE = "intergrax.integrations.providers.graph_store.{slug}.manifest"


@dataclass(frozen=True)
class GraphSoakConfig:
    node_count: int = 12
    neighbor_rounds: int = 4
    find_rounds: int = 3
    max_neighbor_latency_ms: float = 250.0


@dataclass(frozen=True)
class GraphSoakResult:
    passed: bool
    slug: str = ""
    nodes_indexed: int = 0
    neighbor_queries: int = 0
    p95_neighbor_ms: float = 0.0
    reason: str = ""


def manifest_status_for_graph_slug(slug: str) -> IntegrationStatus:
    module = importlib.import_module(_SLUG_MANIFEST_MODULE.format(slug=slug))
    manifest = module.MANIFEST
    return manifest.status


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


def run_graph_store_soak(
    store: GraphStore,
    *,
    config: GraphSoakConfig | None = None,
    slug: str = "",
) -> GraphSoakResult:
    """Execute upsert → link → neighbor → find → unlink soak on any ``GraphStore``."""
    cfg = config or GraphSoakConfig()
    nodes_indexed = 0

    try:
        for index in range(cfg.node_count):
            node = GraphNode(
                id=f"ent:soak_{index}",
                label=f"Soak Entity {index}",
                node_type="entity",
                metadata={"batch": "soak"},
            )
            store.upsert_node(node)
            store.link_chunk(node.id, f"chunk-soak-{index}")
            nodes_indexed += 1
            if index > 0:
                store.upsert_edge(
                    GraphEdge(
                        source_id=f"ent:soak_{index - 1}",
                        target_id=node.id,
                        relation="related_to",
                    )
                )
    except Exception as exc:
        return GraphSoakResult(passed=False, slug=slug, reason=f"index_failed:{exc}")

    latencies_ms: list[float] = []
    neighbor_queries = 0

    try:
        for round_idx in range(cfg.neighbor_rounds):
            node_id = f"ent:soak_{round_idx % cfg.node_count}"
            started = time.perf_counter()
            neighbors = store.neighbors(node_id, max_hops=1)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            neighbor_queries += 1
            if round_idx > 0 and not neighbors:
                return GraphSoakResult(
                    passed=False,
                    slug=slug,
                    nodes_indexed=nodes_indexed,
                    neighbor_queries=neighbor_queries,
                    reason="empty_neighbor_expansion",
                )

        for round_idx in range(cfg.find_rounds):
            found = store.find_nodes(label_contains="Soak", limit=5)
            if not found:
                return GraphSoakResult(
                    passed=False,
                    slug=slug,
                    nodes_indexed=nodes_indexed,
                    neighbor_queries=neighbor_queries,
                    reason="find_nodes_empty",
                )

        chunk_ids = store.chunk_ids_for_nodes({f"ent:soak_0"})
        if not chunk_ids or chunk_ids[0] != "chunk-soak-0":
            return GraphSoakResult(
                passed=False,
                slug=slug,
                nodes_indexed=nodes_indexed,
                neighbor_queries=neighbor_queries,
                reason="chunk_link_mismatch",
            )

        removed = store.unlink_chunks(["chunk-soak-0"])
        if removed <= 0:
            return GraphSoakResult(
                passed=False,
                slug=slug,
                nodes_indexed=nodes_indexed,
                neighbor_queries=neighbor_queries,
                reason="unlink_chunks_noop",
            )
    except Exception as exc:
        return GraphSoakResult(
            passed=False,
            slug=slug,
            nodes_indexed=nodes_indexed,
            neighbor_queries=neighbor_queries,
            reason=f"query_failed:{exc}",
        )

    p95_ms = _p95(latencies_ms)
    if p95_ms > cfg.max_neighbor_latency_ms:
        return GraphSoakResult(
            passed=False,
            slug=slug,
            nodes_indexed=nodes_indexed,
            neighbor_queries=neighbor_queries,
            p95_neighbor_ms=p95_ms,
            reason=f"slo_latency_exceeded:p95={p95_ms:.2f}>{cfg.max_neighbor_latency_ms}",
        )

    return GraphSoakResult(
        passed=True,
        slug=slug,
        nodes_indexed=nodes_indexed,
        neighbor_queries=neighbor_queries,
        p95_neighbor_ms=p95_ms,
        reason="ok",
    )
