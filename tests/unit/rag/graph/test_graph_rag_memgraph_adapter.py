# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile


class _FakeMemgraphIntegration:
    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._chunks: dict[str, dict] = {}
        self._has_chunk: set[tuple[str, str]] = set()

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
        if "merge (n)-[:has_chunk]->(c)" in stmt:
            node_id = str(params["node_id"])
            chunk_id = str(params["chunk_id"])
            self._chunks[chunk_id] = {"id": chunk_id, "tenant_id": params.get("tenant_id")}
            self._has_chunk.add((node_id, chunk_id))
            return GraphQueryResult(records=[], raw={})
        if "contains tolower($needle)" in stmt:
            needle = str(params.get("needle", "")).lower()
            limit = int(params.get("limit", 20))
            rows = []
            for node in self._nodes.values():
                if params.get("tenant_id") and node.get("tenant_id") != params.get("tenant_id"):
                    continue
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


@pytest.mark.gate
def test_memgraph_adapter_via_registry() -> None:
    profile = RagProfile(graph_store_backend="memgraph", graph_rag_enabled=True)
    store = create_rag_graph_store(
        profile=profile,
        integration_graph_store=_FakeMemgraphIntegration(),
        tenant_id="tenant-mg",
    )
    assert isinstance(store, CypherRagGraphStore)
    assert store.tenant_id == "tenant-mg"
