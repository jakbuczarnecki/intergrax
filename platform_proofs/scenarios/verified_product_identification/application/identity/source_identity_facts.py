"""Project per-offer identity profiles into direct source facts on hypotheses."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.contracts.source_identity_fact import (
    SourceIdentityFact,
    SourceIdentityFactKind,
    SourceIdentityFactProvenance,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.profile import (
    SourceOfferIdentityProfile,
    brand_normalization_rule,
    identifier_normalization_rule,
    structured_normalization_rule,
)


def project_source_identity_facts(
    member_refs: tuple[SourceRecordRef, ...],
    profiles: dict[SourceRecordRef, SourceOfferIdentityProfile],
) -> tuple[SourceIdentityFact, ...]:
    facts: list[SourceIdentityFact] = []
    for member_ref in sorted(member_refs, key=source_ref_sort_key):
        profile = profiles[member_ref]
        for identifier in profile.identifiers:
            facts.append(
                SourceIdentityFact(
                    source_ref=profile.source_ref,
                    fact_kind=SourceIdentityFactKind.IDENTIFIER,
                    attribute_key=identifier.identifier_type.value,
                    normalized_value=identifier.normalized_value,
                    identifier_type=identifier.identifier_type,
                    provenance=SourceIdentityFactProvenance(
                        source_field=identifier.source_field,
                        normalization_rule=identifier_normalization_rule(),
                        source_value=identifier.source_value,
                    ),
                )
            )
        for attribute in profile.structured_attributes:
            facts.append(
                SourceIdentityFact(
                    source_ref=profile.source_ref,
                    fact_kind=SourceIdentityFactKind.STRUCTURED_ATTRIBUTE,
                    attribute_key=attribute.canonical_key,
                    normalized_value=attribute.normalized_text_value,
                    identifier_type=None,
                    provenance=SourceIdentityFactProvenance(
                        source_field=attribute.source_field,
                        normalization_rule=structured_normalization_rule(),
                        source_value=attribute.source_value,
                    ),
                )
            )
        if (
            profile.brand is not None
            and profile.brand_source_field is not None
            and profile.brand_source_value is not None
        ):
            facts.append(
                SourceIdentityFact(
                    source_ref=profile.source_ref,
                    fact_kind=SourceIdentityFactKind.STRUCTURED_ATTRIBUTE,
                    attribute_key="brand",
                    normalized_value=profile.brand,
                    identifier_type=None,
                    provenance=SourceIdentityFactProvenance(
                        source_field=profile.brand_source_field,
                        normalization_rule=brand_normalization_rule(),
                        source_value=profile.brand_source_value,
                    ),
                )
            )
    return tuple(
        sorted(
            facts,
            key=lambda item: (
                source_ref_sort_key(item.source_ref),
                item.fact_kind.value,
                item.attribute_key.casefold(),
                item.normalized_value,
            ),
        )
    )
