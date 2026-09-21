# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R3 — research host runtime composition (not declarative settings)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.websearch.schemas.web_search_result import WebSearchResult


@runtime_checkable
class ResearchWebSearchExecutor(Protocol):
    """Structural contract for ``websearch.query`` tool wiring (host composition boundary)."""

    def search_sync(
        self,
        query: str,
        top_k: Optional[int] = None,
        locale: Optional[str] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        safe_search: Optional[bool] = None,
        top_n_fetch: Optional[int] = None,
    ) -> list[WebSearchResult]:
        ...


@dataclass(frozen=True, slots=True)
class ResearchHostRuntimeComposition:
    """Host-scoped runtime overrides for research_application."""

    orchestration_decision_requirement_policy: DecisionRequirementPolicy | None = None
    websearch_executor: ResearchWebSearchExecutor | None = None


__all__ = [
    "ResearchHostRuntimeComposition",
    "ResearchWebSearchExecutor",
]
