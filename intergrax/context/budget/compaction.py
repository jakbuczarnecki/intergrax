# © Artur Czarnecki. All rights reserved.

"""Compaction strategy contract — distinct from truncation (CE-02)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.context.contracts import ContextFragment


@dataclass(frozen=True, slots=True)
class ContextCompactionProvenance:
    strategy_id: str
    reason_code: str
    source_fragment_ids: tuple[str, ...]
    input_token_estimate: int
    output_token_estimate: int
    input_content_hash: str
    output_content_hash: str


@dataclass(frozen=True, slots=True)
class ContextCompactionResult:
    fragment: ContextFragment
    provenance: ContextCompactionProvenance
    lossy: bool


@dataclass(frozen=True, slots=True)
class ContextCompactionInput:
    fragment: ContextFragment
    target_token_budget: int


@runtime_checkable
class ContextCompactionStrategy(Protocol):
    @property
    def strategy_id(self) -> str: ...

    def compact(self, item: ContextCompactionInput) -> ContextCompactionResult | None: ...


class NoOpContextCompactionStrategy:
    """Default: no compaction (selection/budget only)."""

    @property
    def strategy_id(self) -> str:
        return "noop_context_compaction.v1"

    def compact(self, item: ContextCompactionInput) -> ContextCompactionResult | None:
        _ = item
        return None


class DeterministicTailCompactionStrategy:
    """Mechanical tail-preserving shrink for optional fragments (no LLM)."""

    @property
    def strategy_id(self) -> str:
        return "deterministic_tail_compaction.v1"

    def compact(self, item: ContextCompactionInput) -> ContextCompactionResult | None:
        from dataclasses import replace

        from intergrax.context.contracts import content_hash_for_text

        fragment = item.fragment
        if fragment.mandatory:
            return None
        budget = max(1, item.target_token_budget)
        chars_budget = budget * 4
        content = fragment.content or ""
        if len(content) <= chars_budget:
            return None
        compacted_text = content[-chars_budget:]
        compacted = replace(
            fragment,
            content=compacted_text,
            token_estimate=max(1, len(compacted_text) // 4),
            content_hash=content_hash_for_text(compacted_text),
        )
        return ContextCompactionResult(
            fragment=compacted,
            provenance=ContextCompactionProvenance(
                strategy_id=self.strategy_id,
                reason_code="budget.compacted.fragment",
                source_fragment_ids=(fragment.fragment_id,),
                input_token_estimate=fragment.token_estimate,
                output_token_estimate=max(1, len(compacted_text) // 4),
                input_content_hash=fragment.content_hash or content_hash_for_text(content),
                output_content_hash=content_hash_for_text(compacted_text),
            ),
            lossy=True,
        )
