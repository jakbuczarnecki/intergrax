# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.evidence_claims import validate_evidence_reference_id
from intergrax.decision_system.completion_eligibility import (
    CompletionEligibilityStatus,
    evaluate_completion_eligibility,
    evaluate_completion_eligibility_with_provider,
)
from intergrax.decision_system.evidence_requirements import (
    EvidenceRequirement,
    EvidenceRequirementAssessment,
    EvidenceRequirementCriticality,
    EvidenceRequirementOutcome,
    EvidenceRequirementProviderAbsentError,
    EvidenceRequirementSet,
    EvidenceSufficiencyAssessment,
    validate_evidence_requirement_id,
    validate_evidence_requirement_waiver_ref,
)

pytestmark = pytest.mark.unit


def _requirement(
    requirement_id: str,
    *,
    criticality: EvidenceRequirementCriticality = EvidenceRequirementCriticality.MANDATORY,
) -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=validate_evidence_requirement_id(requirement_id),
        criticality=criticality,
    )


def _assessment(
    requirement_id: str,
    outcome: EvidenceRequirementOutcome,
    *,
    supporting: tuple[str, ...] = (),
    waiver_ref: str | None = None,
) -> EvidenceRequirementAssessment:
    return EvidenceRequirementAssessment(
        requirement_id=validate_evidence_requirement_id(requirement_id),
        outcome=outcome,
        supporting_evidence_refs=tuple(
            validate_evidence_reference_id(ref) for ref in supporting
        ),
        waiver_ref=(
            validate_evidence_requirement_waiver_ref(waiver_ref) if waiver_ref is not None else None
        ),
    )


class _StaticProvider:
    def __init__(self, requirements: tuple[EvidenceRequirement, ...]) -> None:
        self._requirements = requirements

    def requirements_for(self) -> EvidenceRequirementSet:
        return EvidenceRequirementSet(requirements=self._requirements)


def test_mandatory_satisfied_is_eligible() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.alpha",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.alpha",),
            ),
        )
    )
    decision = evaluate_completion_eligibility(requirement_set, assessment)
    assert decision.status is CompletionEligibilityStatus.ELIGIBLE
    assert decision.unresolved_mandatory_requirement_ids == ()


def test_mandatory_unsatisfied_is_ineligible() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment("domain.alpha", EvidenceRequirementOutcome.UNSATISFIED),
        )
    )
    decision = evaluate_completion_eligibility(requirement_set, assessment)
    assert decision.status is CompletionEligibilityStatus.INELIGIBLE
    assert decision.unresolved_mandatory_requirement_ids == (
        validate_evidence_requirement_id("domain.alpha"),
    )


def test_mandatory_waived_with_provenance_is_eligible() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.alpha",
                EvidenceRequirementOutcome.WAIVED,
                waiver_ref="policy.decision.alpha",
            ),
        )
    )
    decision = evaluate_completion_eligibility(requirement_set, assessment)
    assert decision.status is CompletionEligibilityStatus.ELIGIBLE


def test_optional_unsatisfied_does_not_block() -> None:
    requirement_set = EvidenceRequirementSet(
        requirements=(
            _requirement("domain.mandatory"),
            _requirement("domain.optional", criticality=EvidenceRequirementCriticality.OPTIONAL),
        )
    )
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.mandatory",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.mandatory",),
            ),
            _assessment("domain.optional", EvidenceRequirementOutcome.UNSATISFIED),
        )
    )
    decision = evaluate_completion_eligibility(requirement_set, assessment)
    assert decision.status is CompletionEligibilityStatus.ELIGIBLE
    assert decision.optional_unsatisfied_count == 1


def test_mixed_requirements_block_only_unresolved_mandatory() -> None:
    requirement_set = EvidenceRequirementSet(
        requirements=(
            _requirement("domain.one"),
            _requirement("domain.two"),
            _requirement("domain.three"),
            _requirement("domain.opt.a", criticality=EvidenceRequirementCriticality.OPTIONAL),
            _requirement("domain.opt.b", criticality=EvidenceRequirementCriticality.OPTIONAL),
            _requirement("domain.opt.c", criticality=EvidenceRequirementCriticality.OPTIONAL),
        )
    )
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.one",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.one",),
            ),
            _assessment(
                "domain.two",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.two",),
            ),
            _assessment("domain.three", EvidenceRequirementOutcome.UNSATISFIED),
            _assessment(
                "domain.opt.a",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.opt.a",),
            ),
            _assessment("domain.opt.b", EvidenceRequirementOutcome.UNSATISFIED),
            _assessment(
                "domain.opt.c",
                EvidenceRequirementOutcome.WAIVED,
                waiver_ref="policy.opt.c",
            ),
        )
    )
    decision = evaluate_completion_eligibility(requirement_set, assessment)
    assert decision.status is CompletionEligibilityStatus.INELIGIBLE
    assert decision.unresolved_mandatory_requirement_ids == (
        validate_evidence_requirement_id("domain.three"),
    )


def test_empty_requirement_set_is_eligible() -> None:
    decision = evaluate_completion_eligibility(
        EvidenceRequirementSet(requirements=()),
        EvidenceSufficiencyAssessment(assessments=()),
    )
    assert decision.status is CompletionEligibilityStatus.ELIGIBLE


def test_provider_absence_fail_closed() -> None:
    with pytest.raises(EvidenceRequirementProviderAbsentError):
        evaluate_completion_eligibility_with_provider(
            provider=None,
            assessment=EvidenceSufficiencyAssessment(assessments=()),
        )


def test_provider_pluginability_without_branching() -> None:
    provider_a = _StaticProvider((_requirement("provider.a.requirement"),))
    provider_b = _StaticProvider(
        (
            _requirement("provider.b.one"),
            _requirement("provider.b.two"),
        )
    )
    decision_a = evaluate_completion_eligibility_with_provider(
        provider=provider_a,
        assessment=EvidenceSufficiencyAssessment(
            assessments=(
                _assessment(
                    "provider.a.requirement",
                    EvidenceRequirementOutcome.SATISFIED,
                    supporting=("evidence.a",),
                ),
            )
        ),
    )
    decision_b = evaluate_completion_eligibility_with_provider(
        provider=provider_b,
        assessment=EvidenceSufficiencyAssessment(
            assessments=(
                _assessment(
                    "provider.b.one",
                    EvidenceRequirementOutcome.SATISFIED,
                    supporting=("evidence.b1",),
                ),
                _assessment("provider.b.two", EvidenceRequirementOutcome.UNSATISFIED),
            )
        ),
    )
    assert decision_a.status is CompletionEligibilityStatus.ELIGIBLE
    assert decision_b.status is CompletionEligibilityStatus.INELIGIBLE
    assert decision_b.unresolved_mandatory_requirement_ids == (
        validate_evidence_requirement_id("provider.b.two"),
    )


def test_tool_count_does_not_affect_eligibility_evaluator() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.alpha",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.alpha",),
            ),
        )
    )
    for _tool_count in (1, 2, 6):
        decision = evaluate_completion_eligibility(requirement_set, assessment)
        assert decision.status is CompletionEligibilityStatus.ELIGIBLE
