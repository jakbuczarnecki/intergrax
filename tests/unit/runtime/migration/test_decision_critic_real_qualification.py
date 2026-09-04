# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from intergrax.runtime.migration.critic_retirement_qualification import (
    CriticRetirementEvidenceProvenance,
    FROZEN_CRITIC_RETIREMENT_EVIDENCE,
    proven_critic_retirement_qualification,
)
from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadiness,
    CriticRetirementReadinessEvidence,
    DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    ParityCapabilityRequirement,
    ParityCapabilityRequirementMode,
    ParityHostScope,
    ParityVerificationCapability,
    evaluate_critic_retirement_readiness_evidence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE_PATH = Path("intergrax/runtime/migration/critic_retirement_qualification.py")
_FORBIDDEN_FRAGMENTS = (
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect.",
    "exec(",
    "eval(",
    "dict[str, Any]",
    "readiness=CriticRetirementReadiness.READY",
)

_REQUIRED_SCOPES = frozenset({
    ParityHostScope.GRAPH_FINAL,
    ParityHostScope.UAEP_STEP,
})


def _evaluate_frozen(
    evidence: CriticRetirementReadinessEvidence,
    *,
    capability_requirements: tuple[ParityCapabilityRequirement, ...] = (
        DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS
    ),
    required_scopes: frozenset[ParityHostScope] = _REQUIRED_SCOPES,
):
    return evaluate_critic_retirement_readiness_evidence(
        evidence,
        required_scopes=required_scopes,
        capability_requirements=capability_requirements,
    )


def test_retirement_qualification_certificate_is_ready_with_provenance() -> None:
    qualification = proven_critic_retirement_qualification()
    assert qualification.report.readiness is CriticRetirementReadiness.READY
    assert qualification.provenance is (
        CriticRetirementEvidenceProvenance.HISTORICAL_PRE_RETIREMENT_QUALIFICATION
    )
    assert qualification.parity_qualification_commit
    assert qualification.ds_mig_03_hitl_transition_commit
    assert qualification.final_regression_gate_commit
    assert qualification.report.blocking_mismatch_count == 0
    assert qualification.report.shadow_error_count == 0
    assert qualification.report.shadow_unavailable_count == 1
    assert ParityHostScope.GRAPH_FINAL in qualification.qualified_scopes
    assert ParityHostScope.UAEP_STEP in qualification.qualified_scopes
    assert ParityVerificationCapability.SEMANTIC in qualification.qualified_capabilities
    assert ParityVerificationCapability.HUMAN_HITL in qualification.qualified_capabilities


def test_frozen_evidence_covers_default_capability_requirements_by_mode() -> None:
    qualification = proven_critic_retirement_qualification()
    for requirement in DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS:
        if requirement.mode is ParityCapabilityRequirementMode.CROSS_SYSTEM:
            assert requirement.capability in (
                qualification.report.cross_system_capabilities_qualified
            )
        elif requirement.mode is ParityCapabilityRequirementMode.DECISION_SUPERSET:
            assert requirement.capability in (
                qualification.report.decision_superset_capabilities_qualified
            )
        elif requirement.mode is ParityCapabilityRequirementMode.ARCHITECTURAL_MAPPING:
            assert requirement.capability in (
                qualification.report.architectural_mappings_qualified
            )


def test_requirement_drift_fails_closed_when_new_capability_required() -> None:
    synthetic_requirement = ParityCapabilityRequirement(
        ParityVerificationCapability.TRAJECTORY,
        ParityCapabilityRequirementMode.ARCHITECTURAL_MAPPING,
    )
    report = _evaluate_frozen(
        FROZEN_CRITIC_RETIREMENT_EVIDENCE,
        capability_requirements=(
            *DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
            synthetic_requirement,
        ),
    )
    assert report.readiness is not CriticRetirementReadiness.READY


def test_missing_semantic_capability_is_not_ready() -> None:
    evidence = replace(
        FROZEN_CRITIC_RETIREMENT_EVIDENCE,
        cross_system_capabilities_qualified=frozenset({
            ParityVerificationCapability.STRUCTURAL,
            ParityVerificationCapability.DETERMINISTIC_GUARDRAIL,
            ParityVerificationCapability.TRAJECTORY,
        }),
    )
    report = _evaluate_frozen(evidence)
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert ParityVerificationCapability.SEMANTIC in report.missing_capabilities


def test_missing_graph_final_scope_is_not_ready() -> None:
    evidence = replace(
        FROZEN_CRITIC_RETIREMENT_EVIDENCE,
        scopes_exercised=frozenset({ParityHostScope.UAEP_STEP}),
    )
    report = _evaluate_frozen(evidence)
    assert report.readiness is not CriticRetirementReadiness.READY
    assert ParityHostScope.GRAPH_FINAL in report.missing_scopes


def test_missing_uaep_step_scope_is_not_ready() -> None:
    evidence = replace(
        FROZEN_CRITIC_RETIREMENT_EVIDENCE,
        scopes_exercised=frozenset({ParityHostScope.GRAPH_FINAL}),
    )
    report = _evaluate_frozen(evidence)
    assert report.readiness is not CriticRetirementReadiness.READY
    assert ParityHostScope.UAEP_STEP in report.missing_scopes


def test_blocking_mismatch_is_not_ready() -> None:
    evidence = replace(
        FROZEN_CRITIC_RETIREMENT_EVIDENCE,
        blocking_mismatch_count=1,
    )
    report = _evaluate_frozen(evidence)
    assert report.readiness is CriticRetirementReadiness.NOT_READY


def test_shadow_error_is_not_ready() -> None:
    evidence = replace(
        FROZEN_CRITIC_RETIREMENT_EVIDENCE,
        shadow_error_count=1,
    )
    report = _evaluate_frozen(evidence)
    assert report.readiness is not CriticRetirementReadiness.READY


def test_intentional_shadow_unavailable_remains_ready() -> None:
    report = _evaluate_frozen(FROZEN_CRITIC_RETIREMENT_EVIDENCE)
    assert report.shadow_unavailable_count == 1
    assert report.readiness is CriticRetirementReadiness.READY


def test_forbidden_audit_retirement_qualification_module() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for fragment in _FORBIDDEN_FRAGMENTS:
        assert fragment not in source
