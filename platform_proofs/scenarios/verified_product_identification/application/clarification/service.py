"""Clarification requirement selection orchestration (5C11)."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingRequirementOrigin,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.answerability_policy import (
    ClarificationAnswerabilityPolicy,
    DeterministicClarificationAnswerabilityPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationDiscriminationMetrics,
    ClarificationRequirement,
    ClarificationRequirementKind,
    ClarificationRequirementProvenance,
    ClarificationSelectionRequest,
    ClarificationSelectionResult,
    NoClarificationReason,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.discriminator_discovery import (
    build_fact_groups,
    requirements_from_fact_groups,
    unresolved_hypothesis_ids,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.materiality_policy import (
    ClarificationMaterialityPolicy,
    DeterministicClarificationMaterialityPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.selection_strategy import (
    ClarificationRequirementSelectionStrategy,
    DeterministicClarificationRequirementSelectionStrategy,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationDecisionReasonCode,
    ProductIdentificationOutcome,
)


_NON_SELECTABLE_MISSING_KEYS = frozenset(
    {
        "competing_identity",
        "variant",
        "identity_evidence",
        "evaluated_hypotheses",
    }
)


@dataclass(frozen=True, slots=True)
class ClarificationRequirementSelectionService:
    selection_strategy: ClarificationRequirementSelectionStrategy
    answerability_policy: ClarificationAnswerabilityPolicy
    materiality_policy: ClarificationMaterialityPolicy

    def select(self, request: ClarificationSelectionRequest) -> ClarificationSelectionResult:
        decision = request.decision
        query_context = request.query_context
        outcome = decision.outcome

        if outcome in (
            ProductIdentificationOutcome.VERIFIED,
            ProductIdentificationOutcome.NO_MATCH,
        ):
            return ClarificationSelectionResult(
                clarification_required=False,
                primary_requirement=None,
                alternate_requirements=(),
                no_clarification_reason=NoClarificationReason.DECISION_ALREADY_TERMINAL,
            )

        if outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION:
            catalog_only = _is_catalog_evidence_only_gap(decision, query_context)
            if catalog_only:
                return ClarificationSelectionResult(
                    clarification_required=False,
                    primary_requirement=None,
                    alternate_requirements=(),
                    no_clarification_reason=NoClarificationReason.CATALOG_EVIDENCE_ONLY_GAP,
                )

        competing_ids = _competing_hypothesis_ids(decision)
        if len(competing_ids) < 2:
            user_missing = _user_missing_requirements(decision, query_context)
            if user_missing:
                primary = user_missing[0]
                alternates = user_missing[1:]
                return ClarificationSelectionResult(
                    clarification_required=True,
                    primary_requirement=primary,
                    alternate_requirements=alternates,
                    no_clarification_reason=None,
                )
            return ClarificationSelectionResult(
                clarification_required=False,
                primary_requirement=None,
                alternate_requirements=(),
                no_clarification_reason=_no_competitor_reason(decision),
            )

        unresolved_ids = unresolved_hypothesis_ids(
            competing_ids=competing_ids,
            evaluated_hypotheses=decision.evaluated_hypotheses,
        )
        if len(unresolved_ids) < 2:
            return ClarificationSelectionResult(
                clarification_required=False,
                primary_requirement=None,
                alternate_requirements=(),
                no_clarification_reason=NoClarificationReason.NO_UNRESOLVED_COMPETITOR,
            )

        candidates: list[ClarificationRequirement] = []
        candidates.extend(
            _user_missing_requirements(decision, query_context),
        )

        groups = build_fact_groups(
            hypothesis_ids=unresolved_ids,
            evaluated_hypotheses=decision.evaluated_hypotheses,
            query_context=query_context,
            materiality_policy=self.materiality_policy,
        )
        if not groups and not candidates:
            reason = (
                NoClarificationReason.NO_DISCRIMINATOR_AVAILABLE
                if decision.decision_reason_code
                is ProductIdentificationDecisionReasonCode.UNRESOLVED_COMPETING_IDENTITY
                else NoClarificationReason.NO_DISCRIMINATING_FACT
            )
            return ClarificationSelectionResult(
                clarification_required=False,
                primary_requirement=None,
                alternate_requirements=(),
                no_clarification_reason=reason,
            )

        origin = _origin_for_reason(decision.decision_reason_code)
        candidates.extend(
            requirements_from_fact_groups(
                groups=groups,
                hypothesis_ids=unresolved_ids,
                answerability_policy=self.answerability_policy,
                query_context=query_context,
                origin=origin,
                reason=decision.decision_reason_code.value,
            )
        )
        candidates = list(_dedupe_requirements(candidates))
        if not candidates:
            return ClarificationSelectionResult(
                clarification_required=False,
                primary_requirement=None,
                alternate_requirements=(),
                no_clarification_reason=NoClarificationReason.NO_USER_ANSWERABLE_REQUIREMENT,
            )

        primary, alternates = self.selection_strategy.select(tuple(candidates))
        if primary is None:
            return ClarificationSelectionResult(
                clarification_required=False,
                primary_requirement=None,
                alternate_requirements=(),
                no_clarification_reason=NoClarificationReason.NO_USER_ANSWERABLE_REQUIREMENT,
            )
        return ClarificationSelectionResult(
            clarification_required=True,
            primary_requirement=primary,
            alternate_requirements=alternates,
            no_clarification_reason=None,
        )


def _competing_hypothesis_ids(decision) -> tuple[str, ...]:
    if decision.outcome is ProductIdentificationOutcome.AMBIGUOUS:
        return tuple(sorted(decision.ambiguity_candidates))
    if (
        decision.decision_reason_code
        is ProductIdentificationDecisionReasonCode.UNRESOLVED_COMPETING_IDENTITY
    ):
        return tuple(
            sorted(
                item.hypothesis.hypothesis_id
                for item in decision.evaluated_hypotheses
                if not item.ranking_key.has_internal_blocking_contradiction
            )
        )
    return ()


def _no_competitor_reason(decision) -> NoClarificationReason:
    if (
        decision.decision_reason_code
        is ProductIdentificationDecisionReasonCode.MISSING_REQUIRED_CATALOG_EVIDENCE
    ):
        return NoClarificationReason.CATALOG_EVIDENCE_ONLY_GAP
    return NoClarificationReason.NO_UNRESOLVED_COMPETITOR


def _origin_for_reason(
    reason: ProductIdentificationDecisionReasonCode,
) -> MissingRequirementOrigin | None:
    if reason is ProductIdentificationDecisionReasonCode.MISSING_DISTINGUISHING_FACT:
        return MissingRequirementOrigin.USER
    if reason is ProductIdentificationDecisionReasonCode.MISSING_REQUIRED_CATALOG_EVIDENCE:
        return MissingRequirementOrigin.CATALOG
    return None


def _user_known_keys(query_context) -> frozenset[str]:
    keys: set[str] = set()
    for item in query_context.required_constraints:
        keys.add(item.attribute_name.casefold())
    return frozenset(keys)


def _is_catalog_evidence_only_gap(decision, query_context) -> bool:
    if decision.outcome is not ProductIdentificationOutcome.INSUFFICIENT_INFORMATION:
        return False
    user_known = _user_known_keys(query_context)
    if not user_known:
        return False
    catalog_missing = [
        item
        for item in decision.missing_requirements
        if item.origin is MissingRequirementOrigin.CATALOG
    ]
    if not catalog_missing:
        return False
    if all(item.attribute_name.casefold() in user_known for item in catalog_missing):
        return True
    return (
        decision.decision_reason_code
        is ProductIdentificationDecisionReasonCode.MISSING_REQUIRED_CATALOG_EVIDENCE
        and any(item.attribute_name.casefold() in user_known for item in catalog_missing)
    )


def _user_missing_requirements(
    decision,
    query_context,
) -> tuple[ClarificationRequirement, ...]:
    answerability = DeterministicClarificationAnswerabilityPolicy()
    materiality = DeterministicClarificationMaterialityPolicy()
    user_known = _user_known_keys(query_context)
    items: list[ClarificationRequirement] = []

    for missing in decision.missing_requirements:
        if missing.origin is not MissingRequirementOrigin.USER:
            continue
        key = missing.attribute_name.casefold()
        if key in _NON_SELECTABLE_MISSING_KEYS:
            continue
        if key in user_known:
            continue
        if not materiality.is_material_attribute(missing.attribute_name):
            continue
        if not answerability.is_selectable(answerability.classify_attribute(missing.attribute_name)):
            continue
        metrics = ClarificationDiscriminationMetrics(
            known_hypothesis_count=0,
            total_competing_hypothesis_count=max(len(decision.ambiguity_candidates), 1),
            distinct_known_value_count=0,
            eliminable_hypothesis_count=0,
            has_complete_coverage=False,
        )
        items.append(
            ClarificationRequirement(
                requirement_id=missing.requirement_id,
                kind=ClarificationRequirementKind.USER_MISSING_FACT,
                attribute_name=missing.attribute_name,
                origin=missing.origin,
                reason=decision.decision_reason_code.value,
                discrimination=metrics,
                provenance=ClarificationRequirementProvenance(
                    affected_hypothesis_ids=(),
                    supporting_source_facts=(),
                    origin=missing.origin,
                ),
                candidate_values=(),
            )
        )

    for item in query_context.missing_user_distinguishing_requirements:
        key = item.attribute_name.casefold()
        if key in user_known or key in _NON_SELECTABLE_MISSING_KEYS:
            continue
        if not materiality.is_material_attribute(item.attribute_name):
            continue
        if not answerability.is_selectable(answerability.classify_attribute(item.attribute_name)):
            continue
        metrics = ClarificationDiscriminationMetrics(
            known_hypothesis_count=0,
            total_competing_hypothesis_count=max(len(decision.ambiguity_candidates), 1),
            distinct_known_value_count=0,
            eliminable_hypothesis_count=0,
            has_complete_coverage=False,
        )
        items.append(
            ClarificationRequirement(
                requirement_id=item.requirement_id,
                kind=ClarificationRequirementKind.USER_MISSING_FACT,
                attribute_name=item.attribute_name,
                origin=item.origin,
                reason=decision.decision_reason_code.value,
                discrimination=metrics,
                provenance=ClarificationRequirementProvenance(
                    affected_hypothesis_ids=(),
                    supporting_source_facts=(),
                    origin=item.origin,
                ),
                candidate_values=(),
            )
        )

    return tuple(_dedupe_requirements(items))


def _dedupe_requirements(
    items: list[ClarificationRequirement],
) -> tuple[ClarificationRequirement, ...]:
    seen: set[tuple[str, str, str]] = set()
    ordered: list[ClarificationRequirement] = []
    for item in sorted(items, key=lambda row: (row.kind.value, row.attribute_name.casefold(), row.requirement_id)):
        token = (
            item.kind.value,
            item.attribute_name.casefold(),
            item.identifier_type.value if item.identifier_type else "",
        )
        if token in seen:
            continue
        seen.add(token)
        ordered.append(item)
    return tuple(ordered)
