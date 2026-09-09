# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.evidence_claims import validate_evidence_reference_id
from intergrax.decision_system.evidence_requirements import (
    EvidenceRequirement,
    EvidenceRequirementAssessment,
    EvidenceRequirementContractError,
    EvidenceRequirementCriticality,
    EvidenceRequirementOutcome,
    EvidenceRequirementProviderAbsentError,
    EvidenceRequirementSet,
    EvidenceSufficiencyAssessment,
    validate_evidence_requirement_id,
    validate_evidence_requirement_waiver_ref,
    validate_evidence_sufficiency_assessment,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERIC_MODULES = (
    _REPO_ROOT / "intergrax" / "decision_system" / "evidence_requirements.py",
    _REPO_ROOT / "intergrax" / "decision_system" / "completion_eligibility.py",
)
_FORBIDDEN_IMPORT_ROOTS = (
    "platform_proofs",
    "intergrax.runtime.nexus",
    "intergrax.runtime.execution",
    "intergrax.agent_distribution",
)


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


def test_duplicate_requirement_ids_fail_closed() -> None:
    requirement_set = EvidenceRequirementSet(
        requirements=(
            _requirement("domain.alpha"),
            _requirement("domain.alpha"),
        )
    )
    with pytest.raises(EvidenceRequirementContractError, match="duplicate requirement"):
        validate_evidence_sufficiency_assessment(
            requirement_set,
            EvidenceSufficiencyAssessment(assessments=()),
        )


def test_waived_without_provenance_invalid() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment("domain.alpha", EvidenceRequirementOutcome.WAIVED),
        )
    )
    with pytest.raises(EvidenceRequirementContractError, match="waiver provenance"):
        validate_evidence_sufficiency_assessment(requirement_set, assessment)


def test_satisfied_without_evidence_provenance_invalid() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment("domain.alpha", EvidenceRequirementOutcome.SATISFIED),
        )
    )
    with pytest.raises(EvidenceRequirementContractError, match="supporting evidence"):
        validate_evidence_sufficiency_assessment(requirement_set, assessment)


def test_satisfied_with_waiver_provenance_invalid() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.alpha",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.alpha",),
                waiver_ref="policy.waiver.alpha",
            ),
        )
    )
    with pytest.raises(EvidenceRequirementContractError, match="waiver provenance"):
        validate_evidence_sufficiency_assessment(requirement_set, assessment)


def test_missing_assessment_fail_closed() -> None:
    requirement_set = EvidenceRequirementSet(
        requirements=(
            _requirement("domain.alpha"),
            _requirement("domain.beta"),
        )
    )
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.alpha",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.alpha",),
            ),
        )
    )
    with pytest.raises(EvidenceRequirementContractError, match="missing assessments"):
        validate_evidence_sufficiency_assessment(requirement_set, assessment)


def test_unknown_assessment_fail_closed() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.beta",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.beta",),
            ),
        )
    )
    with pytest.raises(EvidenceRequirementContractError, match="unknown requirement"):
        validate_evidence_sufficiency_assessment(requirement_set, assessment)


def test_duplicate_assessment_fail_closed() -> None:
    requirement_set = EvidenceRequirementSet(requirements=(_requirement("domain.alpha"),))
    assessment = EvidenceSufficiencyAssessment(
        assessments=(
            _assessment(
                "domain.alpha",
                EvidenceRequirementOutcome.SATISFIED,
                supporting=("evidence.alpha",),
            ),
            _assessment(
                "domain.alpha",
                EvidenceRequirementOutcome.UNSATISFIED,
            ),
        )
    )
    with pytest.raises(EvidenceRequirementContractError, match="duplicate assessment"):
        validate_evidence_sufficiency_assessment(requirement_set, assessment)


def test_provider_absence_fail_closed() -> None:
    with pytest.raises(EvidenceRequirementProviderAbsentError):
        raise EvidenceRequirementProviderAbsentError(
            "completion eligibility gate requires an EvidenceRequirementProvider"
        )


@pytest.mark.parametrize("module_path", _GENERIC_MODULES)
def test_generic_modules_forbid_runtime_and_scenario_imports(module_path: Path) -> None:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    for root in _FORBIDDEN_IMPORT_ROOTS:
        assert not any(
            item == root or item.startswith(f"{root}.") for item in imported
        ), f"{module_path.name} must not import {root}"
