# © Artur Czarnecki. All rights reserved.

"""Fragment cost attribution for CONTEXT_ASSEMBLED payloads (CE-MAINT-02)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.context.contracts import AssembledContext, ContextFragment

# Harness reference rate: 1 microusd per token (test-stable; not billing truth).
_COST_MICROUSD_PER_TOKEN = 1


@dataclass(frozen=True)
class AssemblyCostAttribution:
    fragment_token_cost: int
    estimated_cost_microusd: int


def assembly_cost_from_fragments(fragments: Sequence[ContextFragment]) -> AssemblyCostAttribution:
    token_cost = sum(max(0, fragment.token_estimate) for fragment in fragments)
    return AssemblyCostAttribution(
        fragment_token_cost=token_cost,
        estimated_cost_microusd=token_cost * _COST_MICROUSD_PER_TOKEN,
    )


def assembly_cost_from_assembled(assembled: AssembledContext) -> AssemblyCostAttribution:
    return assembly_cost_from_fragments(assembled.fragments_included)
