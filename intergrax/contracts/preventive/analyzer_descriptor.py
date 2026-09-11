# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive analyzer governance metadata (PREVENTIVE R6-Q)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreventiveResourceBudget:
    max_execution_time_ms: int
    max_candidates_per_run: int

    def __post_init__(self) -> None:
        if self.max_execution_time_ms <= 0:
            raise ValueError("max_execution_time_ms must be positive")
        if self.max_candidates_per_run <= 0:
            raise ValueError("max_candidates_per_run must be positive")


@dataclass(frozen=True, slots=True)
class PreventiveAnalyzerDescriptor:
    """Production plugin identity — required for registry admission."""

    analyzer_id: str
    namespace: str
    version: str
    owner: str
    capabilities: tuple[str, ...]
    quality_profile_id: str
    resource_budget: PreventiveResourceBudget

    def __post_init__(self) -> None:
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if not self.namespace.strip():
            raise ValueError("namespace must be non-empty")
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.owner.strip():
            raise ValueError("owner must be non-empty")
        if not self.capabilities:
            raise ValueError("capabilities must be non-empty")
        if not self.quality_profile_id.strip():
            raise ValueError("quality_profile_id must be non-empty")


__all__ = [
    "PreventiveAnalyzerDescriptor",
    "PreventiveResourceBudget",
]
