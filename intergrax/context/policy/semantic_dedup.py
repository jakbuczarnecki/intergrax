# © Artur Czarnecki. All rights reserved.

"""Default normalized fingerprint dedup strategy (MEM-XINT-5 / MEM-XINT-5-R)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextPolicyReasonCode,
    ContextSemanticDedupDecision,
)
from intergrax.context.policy.authority import authority_rank


class DefaultContextSemanticDeduper:
    """Deterministic normalized-content duplicate detection — not embedding similarity."""

    @property
    def strategy_id(self) -> str:
        return "default_context_normalized_fingerprint_deduper.v1"

    def deduplicate(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> tuple[list[ContextFragment], tuple[ContextSemanticDedupDecision, ...]]:
        _ = request
        grouped: dict[str, list[ContextFragment]] = {}
        for fragment in fragments:
            scope = fragment.scope_ref.tenant_id if fragment.scope_ref is not None else ""
            if fragment.source is ContextFragmentSource.SESSION_HISTORY:
                key = f"{scope}|session|{fragment.fragment_id}|{fragment.semantic_fingerprint}"
            else:
                key = f"{scope}|{fragment.semantic_fingerprint}"
            grouped.setdefault(key, []).append(fragment)

        kept: list[ContextFragment] = []
        decisions: list[ContextSemanticDedupDecision] = []
        for bucket in grouped.values():
            if len(bucket) == 1:
                kept.append(bucket[0])
                continue
            ordered = sorted(
                bucket,
                key=lambda item: (
                    authority_rank(item.authority_class),
                    item.normalized_relevance_score,
                    item.fragment_id,
                ),
                reverse=True,
            )
            winner = ordered[0]
            suppressed = tuple(item.fragment_id for item in ordered[1:])
            if suppressed:
                decisions.append(
                    ContextSemanticDedupDecision(
                        kept_fragment_id=winner.fragment_id,
                        suppressed_fragment_ids=suppressed,
                        reason_code=ContextPolicyReasonCode.SEMANTIC_DUPLICATE,
                        strategy_id=self.strategy_id,
                    ),
                )
            kept.append(winner)
        return kept, tuple(decisions)
