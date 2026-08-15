# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared publication-generation visibility law for GraphRAG evidence."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.distributed.source_operation import (
    RagSourceOperationKey,
    SourceOperationCoordinator,
)
from intergrax.rag.graph.contracts.graph_store import GraphScope

_EVIDENCE_VISIBLE_PREDICATE = """(
    {alias}.generation IS NULL
    OR (
        $coordinator_bound = true
        AND any(
            pair IN $active_pairs
            WHERE pair.source_id = {alias}.source_id
              AND pair.generation = {alias}.generation
        )
    )
)"""


def cypher_evidence_visible(*, alias: str = "e") -> str:
    """Return a Cypher predicate for one RagEvidence alias."""
    return _EVIDENCE_VISIBLE_PREDICATE.format(alias=alias)


def cypher_node_visible(*, node_alias: str = "n") -> str:
    """Return a Cypher predicate matching InMemory ``_node_visible`` semantics."""
    evidence_visible = cypher_evidence_visible(alias="nev")
    return f"""(
        NOT EXISTS {{
            MATCH (nev:RagEvidence {{scope_key: $scope_key}})-[:EVIDENCES_NODE]->({node_alias})
        }}
        OR EXISTS {{
            MATCH (nev:RagEvidence {{scope_key: $scope_key}})-[:EVIDENCES_NODE]->({node_alias})
            WHERE {evidence_visible}
        }}
    )"""


def graph_evidence_visible(
    *,
    versioned: bool,
    generation: str | None,
    source_key: RagSourceOperationKey | None,
    coordinator: SourceOperationCoordinator | None,
) -> bool:
    """Return whether one evidence record is visible under the canonical law."""
    if not versioned:
        return True
    if (
        generation is None
        or source_key is None
        or coordinator is None
    ):
        return False
    try:
        active = coordinator.active_publication_generation(key=source_key)
    except Exception:
        return False
    return active == generation


def resolve_scope_active_generations(
    coordinator: SourceOperationCoordinator,
    scope: GraphScope,
    source_ids: Sequence[str],
) -> dict[str, str]:
    """Resolve authoritative active publication generations for bounded sources."""
    active: dict[str, str] = {}
    for raw_source_id in source_ids:
        source_id = raw_source_id.strip()
        if not source_id or source_id == "__legacy__":
            continue
        key = RagSourceOperationKey(
            tenant_id=scope.tenant_id,
            namespace=scope.namespace,
            workspace_id=scope.workspace_id,
            source_id=source_id,
        )
        try:
            generation = coordinator.active_publication_generation(key=key)
        except Exception as exc:
            raise RuntimeError("graph generation resolution failed") from exc
        if generation is not None:
            active[source_id] = generation
    return active


def visibility_query_params(
    *,
    scope: GraphScope,
    coordinator: SourceOperationCoordinator | None,
    source_ids: Sequence[str],
) -> dict[str, object]:
    """Build Cypher parameters for generation-aware evidence reads."""
    if coordinator is None:
        return {
            "scope_key": scope.key,
            "coordinator_bound": False,
            "active_pairs": [],
        }
    active = resolve_scope_active_generations(coordinator, scope, source_ids)
    return {
        "scope_key": scope.key,
        "coordinator_bound": True,
        "active_pairs": [
            {"source_id": source_id, "generation": generation}
            for source_id, generation in sorted(active.items())
        ],
    }
