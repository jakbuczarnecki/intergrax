# © Artur Czarnecki. All rights reserved.

"""Exact identity/content deduplication (MEM-XINT-5)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.context.contracts import (
    ContextFragment,
    ContextFragmentSource,
    ContextPolicyReasonCode,
)
from intergrax.context.policy.authority import authority_rank


def _exact_identity_key(fragment: ContextFragment) -> str:
    if fragment.source is ContextFragmentSource.SESSION_HISTORY:
        return f"session:{fragment.source_id}:{fragment.fragment_id}"
    if fragment.source_id:
        return f"source:{fragment.source_id}:{fragment.fragment_id}"
    return f"id:{fragment.fragment_id}"


def _scope_key(fragment: ContextFragment) -> str:
    if fragment.scope_ref is None:
        return ""
    return (
        f"{fragment.scope_ref.tenant_id}:"
        f"{fragment.scope_ref.user_id}:"
        f"{fragment.scope_ref.execution_scope_key}"
    )


def _retention_key(fragment: ContextFragment) -> tuple[int, float, float, float, str]:
    return (
        authority_rank(fragment.authority_class),
        fragment.trust_score,
        fragment.freshness_score,
        fragment.normalized_relevance_score,
        fragment.fragment_id,
    )


def _dedup_pass(
    fragments: list[ContextFragment],
    *,
    key_builder: Callable[[ContextFragment], str],
    reason: ContextPolicyReasonCode,
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]], list[tuple[ContextFragment, ContextPolicyReasonCode]]]:
    best: dict[str, ContextFragment] = {}
    order: list[str] = []
    dropped: list[tuple[ContextFragment, str]] = []
    audit: list[tuple[ContextFragment, ContextPolicyReasonCode]] = []
    for fragment in fragments:
        key = key_builder(fragment)
        existing = best.get(key)
        if existing is None:
            best[key] = fragment
            order.append(key)
            continue
        if _retention_key(fragment) > _retention_key(existing):
            dropped.append((existing, reason.value))
            audit.append((existing, reason))
            best[key] = fragment
        else:
            dropped.append((fragment, reason.value))
            audit.append((fragment, reason))
    kept = [best[key] for key in order]
    return kept, dropped, audit


def exact_dedup_fragments(
    fragments: list[ContextFragment],
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]], list[tuple[ContextFragment, ContextPolicyReasonCode]]]:
    identity_kept, identity_dropped, identity_audit = _dedup_pass(
        fragments,
        key_builder=lambda fragment: f"{_scope_key(fragment)}|{_exact_identity_key(fragment)}",
        reason=ContextPolicyReasonCode.EXACT_DUPLICATE_IDENTITY,
    )
    content_kept, content_dropped, content_audit = _dedup_pass(
        identity_kept,
        key_builder=lambda fragment: (
            f"{_scope_key(fragment)}|session|{fragment.fragment_id}|{fragment.content_hash}"
            if fragment.source is ContextFragmentSource.SESSION_HISTORY
            else f"{_scope_key(fragment)}|{fragment.content_hash}"
        ),
        reason=ContextPolicyReasonCode.EXACT_DUPLICATE_CONTENT,
    )
    return (
        content_kept,
        identity_dropped + content_dropped,
        identity_audit + content_audit,
    )
