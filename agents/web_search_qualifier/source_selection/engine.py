# © Artur Czarnecki. All rights reserved.

"""Generic pluginable source selection engine."""

from __future__ import annotations

from typing import Protocol

from web_search_qualifier.source_selection.contracts import (
    SourceSelectionContext,
    SourceSelectionEngineDecision,
    SourceSelectionMode,
    SourceSelectionOutcome,
    SourceSelectionPolicyDecision,
    SourceSelectionPolicyDescriptor,
    SourceSelectionProvenance,
)
from web_search_qualifier.source_selection.llm_selector import LLMSourceSelector
from web_search_qualifier.source_selection.matching import resolve_candidate_url
from web_search_qualifier.web_search import WebSearchCandidate


class SourceSelectionPolicy(Protocol):
    @property
    def descriptor(self) -> SourceSelectionPolicyDescriptor:
        ...

    def evaluate(self, context: SourceSelectionContext) -> SourceSelectionPolicyDecision:
        ...


class SourceSelectionContractError(ValueError):
    """Raised when a policy or LLM result violates source-selection contracts."""


class SourceSelectionEngine:
    def __init__(
        self,
        *,
        policies: tuple[SourceSelectionPolicy, ...],
        llm_selector: LLMSourceSelector | None = None,
    ) -> None:
        self._policies = policies
        self._llm_selector = llm_selector

    @property
    def policies(self) -> tuple[SourceSelectionPolicy, ...]:
        return self._policies

    def select(
        self,
        *,
        run_id: str,
        task_message: str,
        candidates: tuple[WebSearchCandidate, ...],
        llm_candidates: tuple[WebSearchCandidate, ...] | None = None,
        llm_highlight_top_rank: bool = False,
    ) -> SourceSelectionEngineDecision:
        ordered = candidates
        llm_ordered = llm_candidates if llm_candidates is not None else candidates
        context = SourceSelectionContext(task_message=task_message, candidates=candidates)

        for policy in self._policies:
            decision = policy.evaluate(context)
            if decision.outcome is not SourceSelectionOutcome.SELECT:
                continue
            selected_url = decision.selected_url
            if selected_url is None:
                raise SourceSelectionContractError(
                    f"policy {policy.descriptor.policy_id.value} returned SELECT without selected_url",
                )
            resolved = resolve_candidate_url(selected_url, candidates)
            if resolved is None:
                raise SourceSelectionContractError(
                    f"policy {policy.descriptor.policy_id.value} selected URL not in candidates: {selected_url}",
                )
            return SourceSelectionEngineDecision(
                selected_url=resolved,
                provenance=SourceSelectionProvenance(
                    selection_mode=SourceSelectionMode.POLICY,
                    selected_url=resolved,
                    policy_id=policy.descriptor.policy_id,
                    reason_code=decision.reason_code,
                ),
                ordered_candidates=ordered,
            )

        if self._llm_selector is None:
            return SourceSelectionEngineDecision(
                selected_url=None,
                provenance=SourceSelectionProvenance(
                    selection_mode=SourceSelectionMode.LLM,
                    selected_url=None,
                    reason_code="no_policy_match_and_no_llm_selector",
                ),
                ordered_candidates=llm_ordered,
            )

        llm_result = self._llm_selector.select(
            run_id=run_id,
            task_message=task_message,
            candidates=llm_ordered,
            highlight_top_rank=llm_highlight_top_rank,
        )
        selected_url = llm_result.selected_url
        if selected_url is not None:
            resolved = resolve_candidate_url(selected_url, llm_ordered)
            if resolved is None:
                raise SourceSelectionContractError(
                    f"llm selected URL not in candidates: {selected_url}",
                )
            selected_url = resolved

        return SourceSelectionEngineDecision(
            selected_url=selected_url,
            provenance=SourceSelectionProvenance(
                selection_mode=SourceSelectionMode.LLM,
                selected_url=selected_url,
                raw_llm_response=llm_result.raw_response,
                reason_code="llm_source_selection",
            ),
            ordered_candidates=llm_ordered,
        )


__all__ = [
    "SourceSelectionContractError",
    "SourceSelectionEngine",
    "SourceSelectionPolicy",
]
