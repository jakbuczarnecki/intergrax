# © Artur Czarnecki. All rights reserved.

"""Default cross-source conflict resolver (MEM-XINT-5)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextAuthorityClass,
    ContextConflictAction,
    ContextConflictDecision,
    ContextFragment,
    ContextPolicyReasonCode,
    replace_context_fragment,
)
from intergrax.context.policy.authority import authority_rank


class DefaultContextConflictResolver:
    """Deterministic resolver — never mutates Memory stores."""

    @property
    def strategy_id(self) -> str:
        return "default_context_conflict_resolver.v1"

    def resolve(
        self,
        fragments: list[ContextFragment],
        request: ContextAssemblyRequest,
    ) -> tuple[list[ContextFragment], tuple[ContextConflictDecision, ...]]:
        _ = request
        by_key: dict[str, list[ContextFragment]] = {}
        for fragment in fragments:
            if not fragment.conflict_key:
                continue
            scope = fragment.scope_ref.tenant_id if fragment.scope_ref is not None else ""
            key = f"{scope}|{fragment.conflict_key}"
            by_key.setdefault(key, []).append(fragment)

        downrank_ids: set[str] = set()
        decisions: list[ContextConflictDecision] = []
        for group in by_key.values():
            if len(group) < 2:
                continue
            ordered = sorted(
                group,
                key=lambda item: (
                    authority_rank(item.authority_class),
                    item.trust_score,
                    item.freshness_score,
                    item.fragment_id,
                ),
                reverse=True,
            )
            left = ordered[0]
            right = ordered[1]
            action = ContextConflictAction.PREFER_LEFT
            kept_ids = (left.fragment_id,)
            if left.authority_class is ContextAuthorityClass.CANONICAL_MEMORY:
                action = ContextConflictAction.KEEP_BOTH
                kept_ids = (left.fragment_id, right.fragment_id)
            else:
                downrank_ids.add(right.fragment_id)
            decisions.append(
                ContextConflictDecision(
                    left_fragment_id=left.fragment_id,
                    right_fragment_id=right.fragment_id,
                    action=action,
                    kept_fragment_ids=kept_ids,
                    reason_code=ContextPolicyReasonCode.CONFLICT_RESOLVED,
                    strategy_id=self.strategy_id,
                ),
            )

        resolved: list[ContextFragment] = []
        for fragment in fragments:
            if fragment.fragment_id in downrank_ids:
                resolved.append(
                    replace_context_fragment(
                        fragment,
                        normalized_relevance_score=max(0.0, fragment.normalized_relevance_score - 0.25),
                        relevance_score=max(0.0, fragment.relevance_score - 0.25),
                    ),
                )
            else:
                resolved.append(fragment)
        return resolved, tuple(decisions)
