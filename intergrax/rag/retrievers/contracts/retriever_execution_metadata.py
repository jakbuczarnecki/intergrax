# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Execution metadata for retriever engine calls (contract surface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrieverExecutionMetadata:
    requested_retriever_id: str
    used_retriever_id: str
    attempted_retriever_ids: list[str] = field(default_factory=list)
    fallback_applied: bool = False
    retries_exhausted: bool = False
    channel_contributions: dict[str, list[str]] | None = None
    graph_expanded_node_ids: list[str] | None = None
    graph_provenance_summary: str | None = None
    graph_provenance_records: list[dict[str, Any]] | None = None


__all__ = ["RetrieverExecutionMetadata"]
