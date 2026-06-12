# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution metadata for retriever engine calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RetrieverExecutionMetadata:
    requested_retriever_id: str
    used_retriever_id: str
    attempted_retriever_ids: List[str] = field(default_factory=list)
    fallback_applied: bool = False
    retries_exhausted: bool = False
    channel_contributions: Optional[Dict[str, List[str]]] = None
    graph_expanded_node_ids: Optional[List[str]] = None
    graph_provenance_summary: Optional[str] = None
