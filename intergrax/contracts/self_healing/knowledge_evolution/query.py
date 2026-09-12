# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read criteria for strategy knowledge persistence (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeProfileQuery:
    tenant_id: str
    strategy_id: str
    context_fingerprint: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeRevisionQuery:
    tenant_id: str
    strategy_id: str
    context_fingerprint: str
    limit: int = 50

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
class StrategyKnowledgeVersionQuery:
    tenant_id: str
    strategy_id: str
    context_fingerprint: str
    knowledge_version: int

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")
        if self.knowledge_version < 1:
            raise ValueError("knowledge_version must be >= 1")


__all__ = [
    "StrategyKnowledgeProfileQuery",
    "StrategyKnowledgeRevisionQuery",
    "StrategyKnowledgeVersionQuery",
]
