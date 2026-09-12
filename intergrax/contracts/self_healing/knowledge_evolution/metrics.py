# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy metric collection port (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience


@dataclass(frozen=True, slots=True)
class StrategyMetricScope:
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
class StrategyMetricValue:
    name: str
    value: float
    unit: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric name required")
        if not self.unit.strip():
            raise ValueError("metric unit required")


@dataclass(frozen=True, slots=True)
class StrategyMetricBundle:
    provider_id: str
    scope: StrategyMetricScope
    metrics: tuple[StrategyMetricValue, ...]

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("provider_id required")


@runtime_checkable
class StrategyMetricProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def collect(
        self,
        scope: StrategyMetricScope,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
    ) -> StrategyMetricBundle:
        """Measure strategy performance signals — no learning or execution."""
        ...


__all__ = [
    "StrategyMetricBundle",
    "StrategyMetricProvider",
    "StrategyMetricScope",
    "StrategyMetricValue",
]
