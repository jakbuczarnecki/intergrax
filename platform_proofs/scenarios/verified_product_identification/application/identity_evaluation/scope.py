"""Evidence and contradiction relation scope classification."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityEvidence,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    ContradictionRelationScope,
    EvidenceRelationScope,
)


def classify_evidence_scope(
    evidence: IdentityEvidence,
    *,
    member_refs: frozenset[SourceRecordRef],
) -> EvidenceRelationScope:
    """Classify whether evidence relates to hypothesis members."""

    left_is_member = evidence.source_refs[0] in member_refs
    right_is_member = evidence.source_refs[1] in member_refs
    if left_is_member and right_is_member:
        return EvidenceRelationScope.INTERNAL
    if left_is_member or right_is_member:
        return EvidenceRelationScope.EXTERNAL
    return EvidenceRelationScope.INVALID


def classify_contradiction_scope(
    contradiction: IdentityContradiction,
    *,
    member_refs: frozenset[SourceRecordRef],
) -> ContradictionRelationScope:
    """Classify whether a contradiction relates to hypothesis members."""

    left_is_member = contradiction.source_refs[0] in member_refs
    right_is_member = contradiction.source_refs[1] in member_refs
    if left_is_member and right_is_member:
        return ContradictionRelationScope.INTERNAL
    if left_is_member or right_is_member:
        return ContradictionRelationScope.EXTERNAL
    return ContradictionRelationScope.INVALID
