# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationStatus
from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.graph.soak.prod_slo import (
    GRAPH_BETA_PROMOTION_SLUGS,
    STABLE_GRAPH_SOAK_SLUGS,
    GraphSoakConfig,
    manifest_status_for_graph_slug,
    run_graph_store_soak,
)
from intergrax.rag.profiles.rag_profile import APPROVED_PRODUCTION_GRAPH_STORE_SLUGS, RagProfile


class _FakeCypherIntegration:
    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._chunks: dict[str, dict] = {}
        self._has_chunk: set[tuple[str, str]] = set()
        self._edges: set[tuple[str, str]] = set()

    def run_query(self, statement: str, *, parameters: dict | None = None) -> object:
        from intergrax.integrations.contracts.graph_store import GraphQueryResult

        params = parameters or {}
        stmt = " ".join(statement.lower().split())
        if "merge (n:ragentity" in stmt and "set n.label" in stmt:
            node_id = str(params["id"])
            self._nodes[node_id] = {
                "id": node_id,
                "label": params["label"],
                "node_type": params["node_type"],
                "metadata": params.get("metadata") or {},
                "tenant_id": params.get("tenant_id"),
            }
            return GraphQueryResult(records=[{"id": node_id}], raw={})
        if "merge (a)-[r:rag_rel" in stmt:
            self._edges.add((str(params["source_id"]), str(params["target_id"])))
            return GraphQueryResult(records=[], raw={})
        if "match (n:ragentity" in stmt and "return distinct m.id" in stmt:
            node_id = str(params.get("node_id", ""))
            rows = []
            for source_id, target_id in self._edges:
                if source_id == node_id and target_id in self._nodes:
                    rows.append(self._nodes[target_id])
                elif target_id == node_id and source_id in self._nodes:
                    rows.append(self._nodes[source_id])
            return GraphQueryResult(records=rows, raw={})
        if "match (n:ragentity)-[:has_chunk]->(c:ragchunk)" in stmt:
            node_ids = set(params.get("node_ids") or [])
            rows = []
            for node_id, chunk_id in self._has_chunk:
                if node_id in node_ids:
                    rows.append({"chunk_id": chunk_id})
            return GraphQueryResult(records=rows, raw={})
        if "merge (n)-[:has_chunk]->(c)" in stmt:
            node_id = str(params["node_id"])
            chunk_id = str(params["chunk_id"])
            self._chunks[chunk_id] = {"id": chunk_id}
            self._has_chunk.add((node_id, chunk_id))
            return GraphQueryResult(records=[], raw={})
        if "contains tolower($needle)" in stmt or "tolower(n.label) contains tolower($needle)" in stmt:
            needle = str(params.get("needle", "")).lower()
            limit = int(params.get("limit", 20))
            rows = []
            for node in self._nodes.values():
                if needle in str(node.get("label", "")).lower():
                    rows.append(node)
                if len(rows) >= limit:
                    break
            return GraphQueryResult(records=rows, raw={})
        if "detach delete c" in stmt and "chunk_ids" in params:
            removed = 0
            for chunk_id in params["chunk_ids"]:
                if chunk_id in self._chunks:
                    del self._chunks[chunk_id]
                    removed += 1
            return GraphQueryResult(records=[{"removed_chunks": removed}], raw={})
        if "pruned_entities" in stmt:
            return GraphQueryResult(records=[{"pruned_entities": 0}], raw={})
        return GraphQueryResult(records=[], raw={})


pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_stable_graph_slug_manifest_is_stable() -> None:
    for slug in STABLE_GRAPH_SOAK_SLUGS:
        assert manifest_status_for_graph_slug(slug) is IntegrationStatus.STABLE, slug


def test_graph_beta_promotion_candidates_remain_beta() -> None:
    for slug in GRAPH_BETA_PROMOTION_SLUGS:
        assert manifest_status_for_graph_slug(slug) is IntegrationStatus.BETA, slug


def test_inmemory_graph_soak_passes() -> None:
    store = InMemoryGraphStore(tenant_id="graph-soak")
    result = run_graph_store_soak(
        store,
        slug="inmemory",
        config=GraphSoakConfig(node_count=8, neighbor_rounds=3, max_neighbor_latency_ms=100.0),
    )
    assert result.passed is True
    assert result.reason == "ok"
    assert result.nodes_indexed == 8


def test_cypher_adapter_graph_soak_passes() -> None:
    profile = RagProfile(graph_store_backend="falkordb", graph_rag_enabled=True)
    store = create_rag_graph_store(
        profile=profile,
        integration_graph_store=_FakeCypherIntegration(),
        tenant_id="soak-fk",
    )
    assert isinstance(store, CypherRagGraphStore)
    result = run_graph_store_soak(store, slug="falkordb")
    assert result.passed is True


def test_falkordb_in_approved_production_graph_store_slugs() -> None:
    assert "falkordb" in APPROVED_PRODUCTION_GRAPH_STORE_SLUGS
