"""Query-to-source evaluation using per-offer source identity facts."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    normalize_exact_lookup_value,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingRequirementOrigin,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.source_identity_fact import (
    SourceIdentityFact,
    SourceIdentityFactKind,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductIdentifierIdentityScope,
    ProductIdentifierType,
    identity_scope_for_identifier_type,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    ProductIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ContradictedRequirementEvidence,
    MissingRequirement,
    VerifiedRequirementEvidence,
)


def facts_for_hypothesis(
    hypothesis: ProductIdentityHypothesis,
) -> tuple[SourceIdentityFact, ...]:
    return hypothesis.source_identity_facts


def sort_source_identity_facts(
    facts: tuple[SourceIdentityFact, ...],
) -> tuple[SourceIdentityFact, ...]:
    return tuple(
        sorted(
            facts,
            key=lambda item: (
                source_ref_sort_key(item.source_ref),
                item.attribute_key.casefold(),
                item.normalized_value,
                item.provenance.source_field,
                item.provenance.source_value,
            ),
        )
    )


def identifier_facts_by_type(
    facts: tuple[SourceIdentityFact, ...],
    identifier_type: ProductIdentifierType,
) -> tuple[SourceIdentityFact, ...]:
    return tuple(
        item
        for item in facts
        if item.fact_kind is SourceIdentityFactKind.IDENTIFIER
        and item.identifier_type is identifier_type
    )


def structured_facts_for_attribute(
    facts: tuple[SourceIdentityFact, ...],
    *,
    attribute_key: str,
) -> tuple[SourceIdentityFact, ...]:
    key = attribute_key.casefold()
    return tuple(
        item
        for item in facts
        if item.fact_kind is SourceIdentityFactKind.STRUCTURED_ATTRIBUTE
        and item.attribute_key.casefold() == key
    )


def evaluate_requested_identifier(
    *,
    requested: ProductIdentifier,
    facts: tuple[SourceIdentityFact, ...],
) -> tuple[
    str,
    VerifiedRequirementEvidence | None,
    ContradictedRequirementEvidence | None,
    MissingRequirement | None,
]:
    """Return status token: supported | contradicted | missing | incompatible."""

    scope = identity_scope_for_identifier_type(requested.identifier_type)
    if scope is ProductIdentifierIdentityScope.SOURCE_LOCAL:
        return ("incompatible", None, None, None)

    normalized_query = normalize_exact_lookup_value(
        requested.identifier_type,
        requested.value,
    )
    compatible = identifier_facts_by_type(facts, requested.identifier_type)
    if not compatible:
        return (
            "missing",
            None,
            None,
            MissingRequirement(
                attribute_name=requested.identifier_type.value,
                origin=MissingRequirementOrigin.CATALOG,
                requirement_id=f"requested_identifier:{requested.identifier_type.value}",
            ),
        )

    for fact in compatible:
        if fact.normalized_value == normalized_query:
            evidence_row = _verified_from_fact(
                requested=requested,
                fact=fact,
                catalog_value=fact.normalized_value,
            )
            return ("supported", evidence_row, None, None)

    contradicting_facts = sort_source_identity_facts(
        tuple(
            fact
            for fact in compatible
            if fact.normalized_value != normalized_query
        )
    )
    contradicted_value = contradicting_facts[0].normalized_value
    return (
        "contradicted",
        None,
        ContradictedRequirementEvidence(
            attribute_name=requested.identifier_type.value,
            expected_value=requested.value,
            catalog_value=contradicted_value,
            contradicting_identity_evidence=(),
            contradicting_contradictions=(),
            contradicting_source_facts=contradicting_facts,
        ),
        None,
    )


def direct_global_gtin_supported(
    *,
    requested_gtin: ProductIdentifier,
    facts: tuple[SourceIdentityFact, ...],
) -> bool:
    status, ok_row, _, _ = evaluate_requested_identifier(requested=requested_gtin, facts=facts)
    return status == "supported" and ok_row is not None


def direct_manufacturer_mpn_present(facts: tuple[SourceIdentityFact, ...]) -> bool:
    return bool(identifier_facts_by_type(facts, ProductIdentifierType.MPN))


def _verified_from_fact(
    *,
    requested: ProductIdentifier,
    fact: SourceIdentityFact,
    catalog_value: str,
) -> VerifiedRequirementEvidence:
    return VerifiedRequirementEvidence(
        attribute_name=requested.identifier_type.value,
        expected_value=requested.value,
        catalog_value=catalog_value,
        supporting_identity_evidence=(),
        supporting_source_facts=(fact,),
    )
