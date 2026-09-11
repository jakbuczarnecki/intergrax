"""Scoped contradiction evaluation for one identity hypothesis."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    ProductIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.pair_evidence import (
    is_blocking_identity_contradiction,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    ContradictionRelationScope,
    IdentityContradictionEvaluation,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.scope import (
    classify_contradiction_scope,
)


def build_contradiction_evaluation(
    hypothesis: ProductIdentityHypothesis,
) -> IdentityContradictionEvaluation:
    """Classify contradictions by scope and blocking authority from 5C8."""

    member_refs = frozenset(member.source_ref for member in hypothesis.members)
    deduped = _dedupe_contradictions(hypothesis.contradictions)

    internal_blocking: list[IdentityContradiction] = []
    internal_nonblocking: list[IdentityContradiction] = []
    external_blocking: list[IdentityContradiction] = []
    external_nonblocking: list[IdentityContradiction] = []

    for contradiction in deduped:
        scope = classify_contradiction_scope(contradiction, member_refs=member_refs)
        blocking = is_blocking_identity_contradiction(contradiction)
        if scope is ContradictionRelationScope.INTERNAL:
            if blocking:
                internal_blocking.append(contradiction)
            else:
                internal_nonblocking.append(contradiction)
            continue
        if scope is ContradictionRelationScope.EXTERNAL:
            if blocking:
                external_blocking.append(contradiction)
            else:
                external_nonblocking.append(contradiction)

    external_separation_count = len(external_blocking) + len(external_nonblocking)
    return IdentityContradictionEvaluation(
        internal_blocking=tuple(internal_blocking),
        internal_nonblocking=tuple(internal_nonblocking),
        external_blocking=tuple(external_blocking),
        external_nonblocking=tuple(external_nonblocking),
        external_separation_count=external_separation_count,
    )


def _dedupe_contradictions(
    contradictions: tuple[IdentityContradiction, ...],
) -> tuple[IdentityContradiction, ...]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[IdentityContradiction] = []
    for item in contradictions:
        key = (
            item.contradiction_type.value,
            source_ref_sort_key(item.source_refs[0]),
            source_ref_sort_key(item.source_refs[1]),
            item.attribute_key,
            item.left_normalized_value,
            item.right_normalized_value,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return tuple(
        sorted(
            deduped,
            key=lambda item: (
                source_ref_sort_key(item.source_refs[0]),
                source_ref_sort_key(item.source_refs[1]),
                item.contradiction_type.value,
                item.attribute_key.casefold(),
                item.left_normalized_value,
                item.right_normalized_value,
            ),
        )
    )
