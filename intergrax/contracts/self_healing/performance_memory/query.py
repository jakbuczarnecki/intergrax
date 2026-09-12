# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read criteria for strategy performance memory (SELF-HEALING R5.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyPerformanceMemoryQuery:
    tenant_id: str
    strategy_id: str | None = None
    workflow_id: str | None = None
    diagnostic_investigation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if self.workflow_id is not None and not self.workflow_id.startswith("sh_wf_"):
            raise ValueError("workflow_id must be sh_wf_* when set")


__all__ = ["StrategyPerformanceMemoryQuery"]
