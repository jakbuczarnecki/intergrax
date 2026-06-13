# © Artur Czarnecki. All rights reserved.

"""Shared fake Cypher integration for GraphRAG adapter gate tests."""

from __future__ import annotations


class FakeCypherGraphIntegration:
    def run_query(self, statement: str, *, parameters: dict | None = None) -> object:
        from intergrax.integrations.contracts.graph_store import GraphQueryResult

        params = parameters or {}
        stmt = " ".join(statement.lower().split())
        if "merge (n:ragentity" in stmt and "set n.label" in stmt:
            return GraphQueryResult(records=[{"id": str(params["id"])}], raw={})
        if "contains tolower($needle)" in stmt or "tolower(n.label) contains tolower($needle)" in stmt:
            return GraphQueryResult(records=[{"id": "ent:test", "label": "Test", "node_type": "entity", "metadata": {}}], raw={})
        return GraphQueryResult(records=[], raw={})
