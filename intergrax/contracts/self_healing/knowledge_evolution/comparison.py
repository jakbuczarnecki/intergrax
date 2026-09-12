# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy comparison policy port (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.contextual.operating_context import (
    StrategyKnowledgeOperatingContext,
)
from intergrax.contracts.self_healing.knowledge_evolution.metrics import StrategyMetricBundle
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment


class StrategyComparisonPreference(StrEnum):
    PREFER_LEFT = "PREFER_LEFT"
    PREFER_RIGHT = "PREFER_RIGHT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class StrategyComparisonScope:
    tenant_id: str
    context_fingerprint: str
    dimension_weights_ref: str | None
    operating_context: StrategyKnowledgeOperatingContext | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")


@dataclass(frozen=True, slots=True)
class StrategyComparisonSubject:
    strategy_id: str
    metric_bundle: StrategyMetricBundle
    quality_assessment: StrategyQualityAssessment | None
    operating_context: StrategyKnowledgeOperatingContext | None = None
    knowledge_freshness_score: float | None = None

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if self.knowledge_freshness_score is not None and not (
            0.0 <= self.knowledge_freshness_score <= 1.0
        ):
            raise ValueError("knowledge_freshness_score must be in [0.0, 1.0] when set")


@dataclass(frozen=True, slots=True)
class StrategyComparisonResult:
    policy_id: str
    preference: StrategyComparisonPreference
    dimension_weights_ref: str | None
    evidence_refs: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id required")
        if not self.rationale.strip():
            raise ValueError("rationale required")


@runtime_checkable
class StrategyComparisonPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def compare(
        self,
        scope: StrategyComparisonScope,
        left: StrategyComparisonSubject,
        right: StrategyComparisonSubject,
    ) -> StrategyComparisonResult:
        """Contrast strategies — no execution or selection authority."""
        ...


__all__ = [
    "StrategyComparisonPolicy",
    "StrategyComparisonPreference",
    "StrategyComparisonResult",
    "StrategyComparisonScope",
    "StrategyComparisonSubject",
]
