"""Evaluate authoritative query constraints against hypothesis evidence."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingRequirementOrigin,
    NegativeAttributeConstraint,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.source_identity_fact import (
    SourceIdentityFact,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityContradictionType,
    IdentityEvidence,
    IdentityEvidenceStrengthClass,
    IdentityEvidenceType,
    ProductIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ConstraintRequirementStatus,
    ContradictedRequirementEvidence,
    MissingRequirement,
    VerifiedRequirementEvidence,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.direct_source_evidence import (
    facts_for_hypothesis,
    structured_facts_for_attribute,
)


def evaluate_required_constraint(
    hypothesis: ProductIdentityHypothesis,
    constraint: StructuredAttributeConstraint,
) -> tuple[
    ConstraintRequirementStatus,
    VerifiedRequirementEvidence | None,
    ContradictedRequirementEvidence | None,
    MissingRequirement | None,
]:
    attribute_key = constraint.attribute_name.casefold()
    expected = constraint.value.strip()
    facts = facts_for_hypothesis(hypothesis)
    direct_facts = structured_facts_for_attribute(facts, attribute_key=attribute_key)
    if direct_facts:
        return _evaluate_required_from_direct_facts(
            constraint=constraint,
            expected=expected,
            direct_facts=direct_facts,
        )

    supporting, contradicting_values = _structured_values_for_attribute(
        hypothesis,
        attribute_key=attribute_key,
    )
    contradictions = _contradictions_for_attribute(hypothesis, attribute_key=attribute_key)

    if contradictions:
        return (
            ConstraintRequirementStatus.CONTRADICTED,
            None,
            ContradictedRequirementEvidence(
                attribute_name=constraint.attribute_name,
                expected_value=expected,
                catalog_value=contradictions[0].left_normalized_value,
                contradicting_evidence=(),
                contradicting_contradictions=contradictions,
            ),
            None,
        )

    if not supporting and not contradicting_values:
        return (
            ConstraintRequirementStatus.MISSING,
            None,
            None,
            MissingRequirement(
                attribute_name=constraint.attribute_name,
                origin=MissingRequirementOrigin.CATALOG,
                requirement_id=f"required:{attribute_key}",
            ),
        )

    matched_evidence = _matching_evidence(
        supporting,
        constraint=constraint,
    )
    if matched_evidence:
        return (
            ConstraintRequirementStatus.SUPPORTED,
            VerifiedRequirementEvidence(
                attribute_name=constraint.attribute_name,
                expected_value=expected,
                catalog_value=matched_evidence[0].normalized_value,
                supporting_evidence=matched_evidence,
            ),
            None,
            None,
        )

    if supporting:
        catalog_value = supporting[0].normalized_value
        return (
            ConstraintRequirementStatus.CONTRADICTED,
            None,
            ContradictedRequirementEvidence(
                attribute_name=constraint.attribute_name,
                expected_value=expected,
                catalog_value=catalog_value,
                contradicting_evidence=_evidence_with_values(supporting, catalog_value),
                contradicting_contradictions=(),
            ),
            None,
        )

    if contradicting_values:
        return (
            ConstraintRequirementStatus.CONTRADICTED,
            None,
            ContradictedRequirementEvidence(
                attribute_name=constraint.attribute_name,
                expected_value=expected,
                catalog_value=contradicting_values[0],
                contradicting_evidence=_evidence_with_values(supporting, contradicting_values[0]),
                contradicting_contradictions=(),
            ),
            None,
        )

    return (
        ConstraintRequirementStatus.MISSING,
        None,
        None,
        MissingRequirement(
            attribute_name=constraint.attribute_name,
            origin=MissingRequirementOrigin.CATALOG,
            requirement_id=f"required:{attribute_key}",
        ),
    )


def evaluate_negative_constraint(
    hypothesis: ProductIdentityHypothesis,
    constraint: NegativeAttributeConstraint,
) -> tuple[
    ConstraintRequirementStatus,
    ContradictedRequirementEvidence | None,
]:
    attribute_key = constraint.attribute_name.casefold()
    excluded = constraint.excluded_value.strip()
    facts = facts_for_hypothesis(hypothesis)
    direct_facts = structured_facts_for_attribute(facts, attribute_key=attribute_key)
    if direct_facts:
        for fact in direct_facts:
            if _constraint_value_matches(
                catalog_value=fact.normalized_value,
                constraint_value=excluded,
                operator=constraint.operator,
            ):
                return (
                    ConstraintRequirementStatus.CONTRADICTED,
                    ContradictedRequirementEvidence(
                        attribute_name=constraint.attribute_name,
                        expected_value=f"not:{excluded}",
                        catalog_value=fact.normalized_value,
                        contradicting_evidence=(),
                        contradicting_contradictions=(),
                    ),
                )
        return (ConstraintRequirementStatus.SUPPORTED, None)

    supporting, contradicting_values = _structured_values_for_attribute(
        hypothesis,
        attribute_key=attribute_key,
    )
    seen_values: list[str] = []
    for item in supporting:
        if item.normalized_value not in seen_values:
            seen_values.append(item.normalized_value)
    all_values = tuple(seen_values) + contradicting_values

    for value in all_values:
        if _constraint_value_matches(
            catalog_value=value,
            constraint_value=excluded,
            operator=constraint.operator,
        ):
            return (
                ConstraintRequirementStatus.CONTRADICTED,
                ContradictedRequirementEvidence(
                    attribute_name=constraint.attribute_name,
                    expected_value=f"not:{excluded}",
                    catalog_value=value,
                    contradicting_evidence=_evidence_with_values(supporting, value),
                    contradicting_contradictions=(),
                ),
            )
    return (ConstraintRequirementStatus.SUPPORTED, None)


def _evaluate_required_from_direct_facts(
    *,
    constraint: StructuredAttributeConstraint,
    expected: str,
    direct_facts: tuple[SourceIdentityFact, ...],
) -> tuple[
    ConstraintRequirementStatus,
    VerifiedRequirementEvidence | None,
    ContradictedRequirementEvidence | None,
    MissingRequirement | None,
]:
    matched = tuple(
        fact
        for fact in direct_facts
        if _constraint_value_matches(
            catalog_value=fact.normalized_value,
            constraint_value=expected,
            operator=constraint.operator,
        )
    )
    if matched:
        return (
            ConstraintRequirementStatus.SUPPORTED,
            VerifiedRequirementEvidence(
                attribute_name=constraint.attribute_name,
                expected_value=expected,
                catalog_value=matched[0].normalized_value,
                supporting_evidence=(),
            ),
            None,
            None,
        )
    catalog_value = direct_facts[0].normalized_value
    return (
        ConstraintRequirementStatus.CONTRADICTED,
        None,
        ContradictedRequirementEvidence(
            attribute_name=constraint.attribute_name,
            expected_value=expected,
            catalog_value=catalog_value,
            contradicting_evidence=(),
            contradicting_contradictions=(),
        ),
        None,
    )


def _structured_values_for_attribute(
    hypothesis: ProductIdentityHypothesis,
    *,
    attribute_key: str,
) -> tuple[tuple[IdentityEvidence, ...], tuple[str, ...]]:
    supporting: list[IdentityEvidence] = []
    contradicting_values: list[str] = []
    for item in hypothesis.evidence:
        if item.evidence_type is not IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH:
            continue
        if item.strength_class is not IdentityEvidenceStrengthClass.STRONG:
            continue
        if item.attribute_key.casefold() != attribute_key:
            continue
        supporting.append(item)
    for item in hypothesis.contradictions:
        if item.contradiction_type is not IdentityContradictionType.STRUCTURED_ATTRIBUTE_CONFLICT:
            continue
        if item.attribute_key.casefold() != attribute_key:
            continue
        contradicting_values.append(item.left_normalized_value)
        contradicting_values.append(item.right_normalized_value)
    return (tuple(supporting), tuple(dict.fromkeys(contradicting_values).keys()))


def _contradictions_for_attribute(
    hypothesis: ProductIdentityHypothesis,
    *,
    attribute_key: str,
) -> tuple[IdentityContradiction, ...]:
    return tuple(
        item
        for item in hypothesis.contradictions
        if item.contradiction_type is IdentityContradictionType.STRUCTURED_ATTRIBUTE_CONFLICT
        and item.attribute_key.casefold() == attribute_key
    )


def _matching_evidence(
    evidence_rows: tuple[IdentityEvidence, ...],
    *,
    constraint: StructuredAttributeConstraint,
) -> tuple[IdentityEvidence, ...]:
    matched = tuple(
        item
        for item in evidence_rows
        if _constraint_value_matches(
            catalog_value=item.normalized_value,
            constraint_value=constraint.value.strip(),
            operator=constraint.operator,
        )
    )
    return matched


def _evidence_with_values(
    evidence_rows: tuple[IdentityEvidence, ...],
    catalog_value: str,
) -> tuple[IdentityEvidence, ...]:
    return tuple(
        item for item in evidence_rows if item.normalized_value == catalog_value
    )


def _constraint_value_matches(
    *,
    catalog_value: str,
    constraint_value: str,
    operator: StructuredConstraintOperator,
) -> bool:
    left = catalog_value.casefold()
    right = constraint_value.casefold()
    if operator is StructuredConstraintOperator.EQUALS:
        return left == right
    if operator is StructuredConstraintOperator.CONTAINS:
        return right in left
    raise ValueError(f"unsupported operator: {operator}")


def _missing_origin_for_attribute(_attribute_name: str) -> MissingRequirementOrigin:
    return MissingRequirementOrigin.CATALOG
