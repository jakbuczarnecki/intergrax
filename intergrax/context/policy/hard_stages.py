# © Artur Czarnecki. All rights reserved.

"""Platform-owned hard policy stages (MEM-XINT-5-R2)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextFragment,
    ContextPolicyDecision,
    ContextPolicyReasonCode,
    ContextPolicyStage,
)
from intergrax.context.policy.canonicalization import canonicalize_fragment_for_policy
from intergrax.context.policy.exact_dedup import exact_dedup_fragments


def _fragment_ids(fragments: list[ContextFragment]) -> tuple[str, ...]:
    return tuple(fragment.fragment_id for fragment in fragments)


def _decision(
    *,
    stage: ContextPolicyStage,
    strategy_id: str,
    before: list[ContextFragment],
    after: list[ContextFragment],
    reason_code: ContextPolicyReasonCode,
    detail: str = "",
) -> ContextPolicyDecision:
    return ContextPolicyDecision(
        stage=stage,
        strategy_id=strategy_id,
        input_fragment_ids=_fragment_ids(before),
        output_fragment_ids=_fragment_ids(after),
        reason_code=reason_code,
        detail=detail,
    )


def run_hard_policy_pre_stages(
    fragments: list[ContextFragment],
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]], tuple[ContextPolicyDecision, ...]]:
    """Canonicalize and exact-dedup before replaceable behavioral policy."""
    decisions: list[ContextPolicyDecision] = []
    excluded: list[tuple[ContextFragment, str]] = []
    working = list(fragments)

    before = list(working)
    working = [canonicalize_fragment_for_policy(fragment) for fragment in working]
    decisions.append(
        _decision(
            stage=ContextPolicyStage.CANONICALIZE,
            strategy_id="platform.canonicalize.v1",
            before=before,
            after=working,
            reason_code=ContextPolicyReasonCode.CONFLICT_RESOLVED,
            detail="canonical_content",
        ),
    )

    before = list(working)
    working, exact_dropped, _exact_audit = exact_dedup_fragments(working)
    excluded.extend(exact_dropped)
    decisions.append(
        _decision(
            stage=ContextPolicyStage.EXACT_DEDUP,
            strategy_id="platform.exact_dedup.v1",
            before=before,
            after=working,
            reason_code=ContextPolicyReasonCode.EXACT_DUPLICATE_CONTENT,
        ),
    )

    return working, excluded, tuple(decisions)
