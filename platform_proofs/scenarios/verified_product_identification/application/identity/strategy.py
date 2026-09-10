"""Pluggable identity hypothesis grouping strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source_identity import (
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidate,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityEvidence,
    IdentityHypothesisMember,
    ProductIdentityHypothesis,
    ProductIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.pair_evidence import (
    OfferPairIdentityAssessment,
    pair_has_grouping_eligibility,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.profile import (
    SourceOfferIdentityProfile,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.source_identity_facts import (
    project_source_identity_facts,
)


class ProductIdentityHypothesisStrategy(Protocol):
    """Strategy seam for grouping fused offers into identity hypotheses."""

    def group(
        self,
        candidates: tuple[FusedOfferCandidate, ...],
        profiles: dict[SourceRecordRef, SourceOfferIdentityProfile],
        pair_assessments: dict[tuple[SourceRecordRef, SourceRecordRef], OfferPairIdentityAssessment],
    ) -> ProductIdentityHypothesisCollection:
        ...


@dataclass(frozen=True, slots=True)
class DeterministicEvidenceIdentityHypothesisStrategy:
    """Conservative complete-link grouping from discrete pair evidence."""

    def group(
        self,
        candidates: tuple[FusedOfferCandidate, ...],
        profiles: dict[SourceRecordRef, SourceOfferIdentityProfile],
        pair_assessments: dict[tuple[SourceRecordRef, SourceRecordRef], OfferPairIdentityAssessment],
    ) -> ProductIdentityHypothesisCollection:
        ordered_candidates = tuple(
            sorted(candidates, key=lambda candidate: (candidate.fused_rank, source_ref_sort_key(candidate.source_ref)))
        )
        unassigned = set(candidate.source_ref for candidate in ordered_candidates)
        candidate_by_ref = {candidate.source_ref: candidate for candidate in ordered_candidates}
        groups: list[tuple[SourceRecordRef, ...]] = []

        while unassigned:
            seed_ref = _pick_seed_ref(unassigned, candidate_by_ref=candidate_by_ref)
            group_refs: list[SourceRecordRef] = [seed_ref]
            unassigned.remove(seed_ref)

            for candidate_ref in sorted(unassigned, key=source_ref_sort_key):
                if _is_complete_link_compatible(
                    candidate_ref,
                    group_refs,
                    profiles=profiles,
                    pair_assessments=pair_assessments,
                ):
                    group_refs.append(candidate_ref)
                    unassigned.remove(candidate_ref)

            groups.append(tuple(sorted(group_refs, key=source_ref_sort_key)))

        hypotheses = tuple(
            _build_hypothesis(
                member_refs=group_refs,
                candidates=candidate_by_ref,
                pair_assessments=pair_assessments,
                all_candidate_refs=tuple(candidate.source_ref for candidate in ordered_candidates),
                profiles=profiles,
            )
            for group_refs in groups
        )
        return ProductIdentityHypothesisCollection(
            hypotheses=_sort_hypotheses(hypotheses),
        )


def _pick_seed_ref(
    unassigned: set[SourceRecordRef],
    *,
    candidate_by_ref: dict[SourceRecordRef, FusedOfferCandidate],
) -> SourceRecordRef:
    return min(
        unassigned,
        key=lambda source_ref: (
            candidate_by_ref[source_ref].fused_rank,
            source_ref_sort_key(source_ref),
        ),
    )


def _is_complete_link_compatible(
    candidate_ref: SourceRecordRef,
    group_refs: list[SourceRecordRef],
    *,
    profiles: dict[SourceRecordRef, SourceOfferIdentityProfile],
    pair_assessments: dict[tuple[SourceRecordRef, SourceRecordRef], OfferPairIdentityAssessment],
) -> bool:
    for member_ref in group_refs:
        assessment = pair_assessments[_pair_key(member_ref, candidate_ref)]
        if not pair_has_grouping_eligibility(
            assessment,
            left_profile=profiles[assessment.left_source_ref],
            right_profile=profiles[assessment.right_source_ref],
        ):
            return False
    return True


def _build_hypothesis(
    *,
    member_refs: tuple[SourceRecordRef, ...],
    candidates: dict[SourceRecordRef, FusedOfferCandidate],
    pair_assessments: dict[tuple[SourceRecordRef, SourceRecordRef], OfferPairIdentityAssessment],
    all_candidate_refs: tuple[SourceRecordRef, ...],
    profiles: dict[SourceRecordRef, SourceOfferIdentityProfile],
) -> ProductIdentityHypothesis:
    members = tuple(
        IdentityHypothesisMember(
            source_ref=member_ref,
            fused_rank=candidates[member_ref].fused_rank,
            fusion_evidence=candidates[member_ref].evidence,
        )
        for member_ref in sorted(member_refs, key=source_ref_sort_key)
    )

    evidence: list[IdentityEvidence] = []
    contradictions: list[IdentityContradiction] = []
    member_set = set(member_refs)

    for left_index, left_ref in enumerate(member_refs):
        for right_ref in member_refs[left_index + 1 :]:
            assessment = pair_assessments[_pair_key(left_ref, right_ref)]
            evidence.extend(assessment.evidence)
            contradictions.extend(assessment.contradictions)

    for member_ref in member_refs:
        for other_ref in all_candidate_refs:
            if other_ref in member_set or other_ref == member_ref:
                continue
            assessment = pair_assessments[_pair_key(member_ref, other_ref)]
            contradictions.extend(assessment.contradictions)

    return ProductIdentityHypothesis(
        hypothesis_id=source_ref_set_sha256(member_refs),
        members=members,
        evidence=_dedupe_evidence(evidence),
        contradictions=_dedupe_contradictions(contradictions),
        source_identity_facts=project_source_identity_facts(member_refs, profiles),
    )


def _pair_key(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
) -> tuple[SourceRecordRef, SourceRecordRef]:
    ordered = tuple(sorted((left_ref, right_ref), key=source_ref_sort_key))
    return ordered[0], ordered[1]


def _dedupe_evidence(items: list[IdentityEvidence]) -> tuple[IdentityEvidence, ...]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[IdentityEvidence] = []
    for item in items:
        key = (
            item.evidence_type.value,
            source_ref_sort_key(item.source_refs[0]),
            source_ref_sort_key(item.source_refs[1]),
            item.attribute_key,
            item.normalized_value,
            item.provenance.source_field,
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
                item.evidence_type.value,
                item.attribute_key.casefold(),
                item.normalized_value,
            ),
        )
    )


def _dedupe_contradictions(
    items: list[IdentityContradiction],
) -> tuple[IdentityContradiction, ...]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[IdentityContradiction] = []
    for item in items:
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


def _sort_hypotheses(
    hypotheses: tuple[ProductIdentityHypothesis, ...],
) -> tuple[ProductIdentityHypothesis, ...]:
    return tuple(
        sorted(
            hypotheses,
            key=lambda hypothesis: (
                min(member.fused_rank for member in hypothesis.members),
                -len(hypothesis.members),
                hypothesis.hypothesis_id,
            ),
        )
    )
