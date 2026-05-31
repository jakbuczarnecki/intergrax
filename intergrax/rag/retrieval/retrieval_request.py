# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RetrievalRequest:
    """Canonical retrieval input for tools, Nexus, and AnswerPipeline adapters."""

    query: str
    top_k: Optional[int] = None
    metadata_filter: Any = None
    score_threshold: Optional[float] = None
    retriever_id: Optional[str] = None
    route_tier_override: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
