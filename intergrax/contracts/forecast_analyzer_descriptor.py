# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Governance metadata for statistical forecast analyzers (PREDICTIVE R3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ForecastResourceBudget:
    """Bounded analyzer resources — enterprise isolation."""

    max_execution_time_ms: int
    max_input_points: int
    max_memory_kb: int

    def __post_init__(self) -> None:
        if self.max_execution_time_ms <= 0:
            raise ValueError("max_execution_time_ms must be positive")
        if self.max_input_points <= 0:
            raise ValueError("max_input_points must be positive")
        if self.max_memory_kb <= 0:
            raise ValueError("max_memory_kb must be positive")


@dataclass(frozen=True, slots=True)
class ForecastAnalyzerDescriptor:
    """Registry and operator-facing forecast analyzer identity."""

    analyzer_id: str
    version: str
    namespace: str
    priority: int
    supported_features: tuple[str, ...]
    resource_budget: ForecastResourceBudget
    supported_risk_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.namespace.strip():
            raise ValueError("namespace must be non-empty")
        if not self.supported_features:
            raise ValueError("supported_features must be non-empty")
        if not self.supported_risk_types:
            raise ValueError("supported_risk_types must be non-empty")


__all__ = ["ForecastAnalyzerDescriptor", "ForecastResourceBudget"]
