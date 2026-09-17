# © Artur Czarnecki. All rights reserved.

"""Default score normalization strategies (MEM-XINT-5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from intergrax.context.contracts import (
    ContextFragmentSource,
    ContextNormalizationInput,
    replace_context_fragment,
)


def _clamp_unit_interval(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True, slots=True)
class ContextSourceNormalizationPolicy:
    source: ContextFragmentSource
    min_raw: float
    max_raw: float


DEFAULT_SOURCE_NORMALIZATION_POLICIES: tuple[ContextSourceNormalizationPolicy, ...] = (
    ContextSourceNormalizationPolicy(ContextFragmentSource.RAG, 0.0, 1.0),
    ContextSourceNormalizationPolicy(ContextFragmentSource.WEBSEARCH, 0.0, 1.0),
    ContextSourceNormalizationPolicy(ContextFragmentSource.LONGTERM_MEMORY, 0.0, 1.0),
    ContextSourceNormalizationPolicy(ContextFragmentSource.TOOL_OUTPUT, 0.0, 1.0),
    ContextSourceNormalizationPolicy(ContextFragmentSource.SESSION_HISTORY_SEMANTIC, 0.0, 1.0),
)


class DefaultContextScoreNormalizer:
    """Deterministic bounded normalizer — no LLM, preserves raw signal."""

    @property
    def strategy_id(self) -> str:
        return "default_context_score_normalizer.v1"

    def __init__(
        self,
        *,
        policies: tuple[ContextSourceNormalizationPolicy, ...] = DEFAULT_SOURCE_NORMALIZATION_POLICIES,
    ) -> None:
        self._policies = {policy.source: policy for policy in policies}

    def normalize(self, item: ContextNormalizationInput):
        fragment = item.fragment
        raw = fragment.raw_relevance_signal
        if raw is None or math.isnan(raw) or math.isinf(raw):
            raw = 0.0
        policy = self._policies.get(fragment.source)
        if policy is None:
            normalized = _clamp_unit_interval(float(raw))
        else:
            span = policy.max_raw - policy.min_raw
            if span <= 0.0:
                normalized = _clamp_unit_interval(float(raw))
            else:
                scaled = (float(raw) - policy.min_raw) / span
                normalized = _clamp_unit_interval(scaled)
        return replace_context_fragment(
            fragment,
            raw_relevance_signal=float(raw),
            normalized_relevance_score=normalized,
            relevance_score=normalized,
        )
