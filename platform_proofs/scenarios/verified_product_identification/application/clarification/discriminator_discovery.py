"""Derive clarification candidates from source identity facts."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingRequirementOrigin,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.source_identity_fact import (
    SourceIdentityFact,
    SourceIdentityFactKind,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
    identity_scope_for_identifier_type,
    ProductIdentifierIdentityScope,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    EvaluatedIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.answerability_policy import (
    ClarificationAnswerabilityPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationDiscriminationMetrics,
    ClarificationRequirement,
    ClarificationRequirementKind,
    ClarificationRequirementProvenance,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.materiality_policy import (
    ClarificationMaterialityPolicy,
)


@dataclass(frozen=True, slots=True)
class _FactGroupKey:
    kind: ClarificationRequirementKind
    attribute_name: str
    identifier_type: ProductIdentifierType | None

    def sort_key(self) -> tuple[str, str, str]:
        id_part = self.identifier_type.value if self.identifier_type else ""
        return (self.kind.value, self.attribute_name.casefold(), id_part)


@dataclass(frozen=True, slots=True)
class _FactGroupState:
    key: _FactGroupKey
    values_by_hypothesis: dict[str, tuple[str, ...]]
    facts_by_hypothesis: dict[str, tuple[SourceIdentityFact, ...]]


def _user_supplied_attribute_keys(query_context: ProductIdentificationQueryContext) -> frozenset[str]:
    keys: set[str] = set()
    for constraint in query_context.required_constraints:
        keys.add(constraint.attribute_name.casefold())
    for identifier in query_context.requested_identifiers:
        keys.add(identifier.identifier_type.value.casefold())
    return frozenset(keys)


def unresolved_hypothesis_ids(
    *,
    competing_ids: tuple[str, ...],
    evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...],
) -> tuple[str, ...]:
    eligible = [item for item in competing_ids if item.strip()]
    if not eligible:
        return ()
    by_id = {item.hypothesis.hypothesis_id for item in evaluated_hypotheses}
    return tuple(
        hypothesis_id for hypothesis_id in sorted(eligible) if hypothesis_id in by_id
    )


def build_fact_groups(
    *,
    hypothesis_ids: tuple[str, ...],
    evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...],
    query_context: ProductIdentificationQueryContext,
    materiality_policy: ClarificationMaterialityPolicy,
) -> tuple[_FactGroupState, ...]:
    if len(hypothesis_ids) < 2:
        return ()

    by_id = {item.hypothesis.hypothesis_id: item for item in evaluated_hypotheses}
    user_known = _user_supplied_attribute_keys(query_context)
    groups: dict[_FactGroupKey, _FactGroupState] = {}

    for hypothesis_id in hypothesis_ids:
        evaluated = by_id.get(hypothesis_id)
        if evaluated is None:
            continue
        facts = evaluated.hypothesis.source_identity_facts
        seen_keys: set[tuple[str, str, str]] = set()
        for fact in facts:
            key = _fact_group_key_for_fact(fact, materiality_policy=materiality_policy)
            if key is None:
                continue
            if key.attribute_name.casefold() in user_known:
                continue
            if key.identifier_type is not None:
                scope = identity_scope_for_identifier_type(key.identifier_type)
                if scope is ProductIdentifierIdentityScope.SOURCE_LOCAL:
                    continue
            dedupe_token = (
                key.kind.value,
                key.attribute_name.casefold(),
                key.identifier_type.value if key.identifier_type else "",
                fact.normalized_value.casefold(),
            )
            if dedupe_token in seen_keys:
                continue
            seen_keys.add(dedupe_token)

            state = groups.get(key)
            if state is None:
                state = _FactGroupState(key=key, values_by_hypothesis={}, facts_by_hypothesis={})
                groups[key] = state
            values = state.values_by_hypothesis.get(hypothesis_id, ())
            if fact.normalized_value not in values:
                values = values + (fact.normalized_value,)
            facts_for_hyp = state.facts_by_hypothesis.get(hypothesis_id, ())
            facts_for_hyp = facts_for_hyp + (fact,)
            state.values_by_hypothesis[hypothesis_id] = values
            state.facts_by_hypothesis[hypothesis_id] = facts_for_hyp

    return tuple(sorted(groups.values(), key=lambda item: item.key.sort_key()))


def _fact_group_key_for_fact(
    fact: SourceIdentityFact,
    *,
    materiality_policy: ClarificationMaterialityPolicy,
) -> _FactGroupKey | None:
    if fact.fact_kind is SourceIdentityFactKind.IDENTIFIER:
        if fact.identifier_type is None:
            return None
        return _FactGroupKey(
            kind=ClarificationRequirementKind.IDENTIFIER_VALUE,
            attribute_name=fact.attribute_key,
            identifier_type=fact.identifier_type,
        )
    if not materiality_policy.is_material_attribute(fact.attribute_key):
        return None
    return _FactGroupKey(
        kind=ClarificationRequirementKind.IDENTITY_DISCRIMINATOR,
        attribute_name=fact.attribute_key,
        identifier_type=None,
    )


def _metrics_for_group(
    state: _FactGroupState,
    *,
    hypothesis_ids: tuple[str, ...],
) -> ClarificationDiscriminationMetrics | None:
    total = len(hypothesis_ids)
    known_count = 0
    distinct_values: set[str] = set()
    value_to_hypotheses: dict[str, set[str]] = {}

    for hypothesis_id in hypothesis_ids:
        values = state.values_by_hypothesis.get(hypothesis_id, ())
        if values:
            known_count += 1
            for value in values:
                distinct_values.add(value)
                bucket = value_to_hypotheses.setdefault(value, set())
                bucket.add(hypothesis_id)

    if len(distinct_values) < 2:
        return None

    eliminable = 0
    for hypothesis_id in hypothesis_ids:
        values = state.values_by_hypothesis.get(hypothesis_id, ())
        if not values:
            continue
        if len(values) == 1:
            value = values[0]
            if len(value_to_hypotheses.get(value, set())) < total:
                eliminable += 1

    has_complete = known_count == total
    return ClarificationDiscriminationMetrics(
        known_hypothesis_count=known_count,
        total_competing_hypothesis_count=total,
        distinct_known_value_count=len(distinct_values),
        eliminable_hypothesis_count=eliminable,
        has_complete_coverage=has_complete,
    )


def requirements_from_fact_groups(
    *,
    groups: tuple[_FactGroupState, ...],
    hypothesis_ids: tuple[str, ...],
    answerability_policy: ClarificationAnswerabilityPolicy,
    query_context: ProductIdentificationQueryContext,
    origin: MissingRequirementOrigin | None,
    reason: str,
) -> tuple[ClarificationRequirement, ...]:
    candidates: list[ClarificationRequirement] = []
    for state in groups:
        metrics = _metrics_for_group(state, hypothesis_ids=hypothesis_ids)
        if metrics is None:
            continue
        key = state.key
        if key.kind is ClarificationRequirementKind.IDENTIFIER_VALUE:
            if key.identifier_type is None:
                continue
            answerability = answerability_policy.classify_identifier(
                key.identifier_type,
                query_context=query_context,
            )
        else:
            answerability = answerability_policy.classify_attribute(key.attribute_name)
        if not answerability_policy.is_selectable(answerability):
            continue

        affected = tuple(
            sorted(
                hypothesis_id
                for hypothesis_id in hypothesis_ids
                if hypothesis_id in state.values_by_hypothesis
            )
        )
        if len(affected) < 2:
            continue

        supporting: list[SourceIdentityFact] = []
        candidate_values: list[str] = []
        for hypothesis_id in affected:
            supporting.extend(state.facts_by_hypothesis.get(hypothesis_id, ()))
            for value in state.values_by_hypothesis.get(hypothesis_id, ()):
                if value not in candidate_values:
                    candidate_values.append(value)
        candidate_values_sorted = tuple(sorted(candidate_values, key=lambda item: item.casefold()))

        requirement_id = _requirement_id_for_key(key)
        kind = (
            ClarificationRequirementKind.IDENTITY_DISCRIMINATOR
            if key.kind is ClarificationRequirementKind.IDENTITY_DISCRIMINATOR
            else ClarificationRequirementKind.IDENTIFIER_VALUE
        )
        candidates.append(
            ClarificationRequirement(
                requirement_id=requirement_id,
                kind=kind,
                attribute_name=key.attribute_name,
                origin=origin,
                reason=reason,
                discrimination=metrics,
                provenance=ClarificationRequirementProvenance(
                    affected_hypothesis_ids=affected,
                    supporting_source_facts=tuple(supporting),
                    origin=origin,
                ),
                identifier_type=key.identifier_type,
                candidate_values=candidate_values_sorted,
            )
        )
    return tuple(candidates)


def _requirement_id_for_key(key: _FactGroupKey) -> str:
    if key.identifier_type is not None:
        return f"identifier:{key.identifier_type.value}:{key.attribute_name.casefold()}"
    return f"attribute:{key.attribute_name.casefold()}"
