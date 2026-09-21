# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R3 — research host runtime composition (not declarative settings)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.tools.providers.websearch.executor_contract import WebSearchQueryExecutor


@dataclass(frozen=True, slots=True)
class ResearchHostRuntimeComposition:
    """Host-scoped runtime overrides for research_application."""

    orchestration_decision_requirement_policy: DecisionRequirementPolicy | None = None
    websearch_executor: WebSearchQueryExecutor | None = None


__all__ = [
    "ResearchHostRuntimeComposition",
]
