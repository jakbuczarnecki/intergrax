# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read criteria for knowledge audit persistence (SELF-HEALING R5.6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeChangeRecordQuery:
    tenant_id: str
    strategy_id: str
    context_fingerprint: str
    limit: int = 100

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeUpdatedEventQuery:
    tenant_id: str
    strategy_id: str
    context_fingerprint: str
    limit: int = 100

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")


__all__ = [
    "StrategyKnowledgeChangeRecordQuery",
    "StrategyKnowledgeUpdatedEventQuery",
]
