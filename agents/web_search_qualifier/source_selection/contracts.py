# © Artur Czarnecki. All rights reserved.

"""Immutable contracts for pluginable web source selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from web_search_qualifier.web_search import WebSearchCandidate


class SourceSelectionOutcome(StrEnum):
    ABSTAIN = "abstain"
    SELECT = "select"


class SourceSelectionMode(StrEnum):
    POLICY = "policy"
    LLM = "llm"


@dataclass(frozen=True, slots=True)
class SourceSelectionPolicyId:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip()
        if not normalized:
            raise ValueError("SourceSelectionPolicyId must be non-empty")
        object.__setattr__(self, "value", normalized)


@dataclass(frozen=True, slots=True)
class SourceSelectionPolicyDescriptor:
    policy_id: SourceSelectionPolicyId
    display_name: str


@dataclass(frozen=True, slots=True)
class SourceSelectionContext:
    task_message: str
    candidates: tuple[WebSearchCandidate, ...]


@dataclass(frozen=True, slots=True)
class SourceSelectionPolicyDecision:
    outcome: SourceSelectionOutcome
    selected_url: str | None = None
    reason_code: str = ""


@dataclass(frozen=True, slots=True)
class SourceSelectionProvenance:
    selection_mode: SourceSelectionMode
    selected_url: str | None
    policy_id: SourceSelectionPolicyId | None = None
    raw_llm_response: str | None = None
    reason_code: str = ""


@dataclass(frozen=True, slots=True)
class SourceSelectionEngineDecision:
    selected_url: str | None
    provenance: SourceSelectionProvenance
    ordered_candidates: tuple[WebSearchCandidate, ...]


__all__ = [
    "SourceSelectionContext",
    "SourceSelectionEngineDecision",
    "SourceSelectionMode",
    "SourceSelectionOutcome",
    "SourceSelectionPolicyDecision",
    "SourceSelectionPolicyDescriptor",
    "SourceSelectionPolicyId",
    "SourceSelectionProvenance",
]
