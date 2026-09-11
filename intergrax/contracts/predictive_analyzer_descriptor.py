# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Analyzer governance metadata (PREDICTIVE R2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalyzerDescriptor:
    """Registry and operator-facing analyzer identity — not diagnostic authority."""

    analyzer_id: str
    version: str
    owner: str
    scope: str
    required_evidence: tuple[str, ...]
    supported_risk_types: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.owner.strip():
            raise ValueError("owner must be non-empty")
        if not self.scope.strip():
            raise ValueError("scope must be non-empty")
        if not self.supported_risk_types:
            raise ValueError("supported_risk_types must be non-empty")


__all__ = ["AnalyzerDescriptor"]
