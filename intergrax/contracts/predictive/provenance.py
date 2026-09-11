# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive context provenance (PREDICTIVE R4 governance)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class PredictiveContextProvenance:
    """Where one context slice originated — audit reconstruction."""

    source: str
    version: str
    generated_at: datetime
    tenant_scope: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.tenant_scope.strip():
            raise ValueError("tenant_scope must be non-empty")


__all__ = ["PredictiveContextProvenance"]
