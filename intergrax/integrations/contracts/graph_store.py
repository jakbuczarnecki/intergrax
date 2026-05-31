# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Graph store integration contract (§7.1.2, Phase M.7)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class GraphNodeRecord(BaseModel):
    id: str
    labels: Sequence[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphQueryResult(BaseModel):
    records: Sequence[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


@runtime_checkable
class GraphStore(Protocol):
    """Property-graph facade for agent memory and dependency graphs."""

    def run_query(
        self,
        statement: str,
        *,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> GraphQueryResult:
        """Execute Cypher or vendor-neutral graph query."""

    def get_node(self, node_id: str) -> Optional[GraphNodeRecord]:
        """Fetch a node by internal id."""
