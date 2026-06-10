# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution metadata for retriever engine calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RetrieverExecutionMetadata:
    requested_retriever_id: str
    used_retriever_id: str
    attempted_retriever_ids: List[str] = field(default_factory=list)
    fallback_applied: bool = False
    retries_exhausted: bool = False
