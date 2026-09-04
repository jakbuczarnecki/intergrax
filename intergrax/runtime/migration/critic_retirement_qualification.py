# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable Critic retirement qualification certificate (DS-MIG-04).

Evaluates immutable historical parity evidence through the canonical retirement
readiness evaluator. This is not a live shadow comparison — it is the frozen
qualification record that justified deleting ``intergrax/runtime/critic/**``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadinessEvidence,
    CriticRetirementReadinessReport,
    DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    ParityHostScope,
    ParityVerificationCapability,
    evaluate_critic_retirement_readiness_evidence,
)

PARITY_QUALIFICATION_SOURCE_COMMIT = "342661cb872abc12a2b704e027f1d324bf1d79f0"
DS_MIG_03_HITL_TRANSITION_COMMIT = "f7820a3e9abdb65af70d9f09ce439e3272582bbc"
FINAL_PRE_RETIREMENT_REGRESSION_COMMIT = "5444efdde53673f344fb049eb9ec9dfe235b8a32"

_REQUIRED_RETIREMENT_SCOPES = frozenset({
    ParityHostScope.GRAPH_FINAL,
    ParityHostScope.UAEP_STEP,
})

_CROSS_SYSTEM_QUALIFIED = frozenset({
    ParityVerificationCapability.STRUCTURAL,
    ParityVerificationCapability.DETERMINISTIC_GUARDRAIL,
    ParityVerificationCapability.SEMANTIC,
    ParityVerificationCapability.TRAJECTORY,
})
_DECISION_SUPERSET_QUALIFIED = frozenset({
    ParityVerificationCapability.EVIDENCE,
    ParityVerificationCapability.DOMAIN,
})
_ARCHITECTURAL_QUALIFIED = frozenset({ParityVerificationCapability.HUMAN_HITL})

FROZEN_CRITIC_RETIREMENT_EVIDENCE = CriticRetirementReadinessEvidence(
    blocking_mismatch_count=0,
    shadow_error_count=0,
    shadow_unavailable_count=1,
    scopes_exercised=_REQUIRED_RETIREMENT_SCOPES,
    decision_capabilities_exercised=(
        _CROSS_SYSTEM_QUALIFIED | _DECISION_SUPERSET_QUALIFIED | _ARCHITECTURAL_QUALIFIED
    ),
    critic_capabilities_exercised=_CROSS_SYSTEM_QUALIFIED | _ARCHITECTURAL_QUALIFIED,
    cross_system_capabilities_qualified=_CROSS_SYSTEM_QUALIFIED,
    decision_superset_capabilities_qualified=_DECISION_SUPERSET_QUALIFIED,
    architectural_mappings_qualified=_ARCHITECTURAL_QUALIFIED,
)


class CriticRetirementEvidenceProvenance(str, Enum):
    HISTORICAL_PRE_RETIREMENT_QUALIFICATION = "historical_pre_retirement_qualification"


@dataclass(frozen=True, slots=True)
class CriticRetirementQualification:
    """Immutable aggregate proving safe Critic runtime retirement."""

    report: CriticRetirementReadinessReport
    provenance: CriticRetirementEvidenceProvenance
    parity_qualification_commit: str
    ds_mig_03_hitl_transition_commit: str
    final_regression_gate_commit: str
    qualified_capabilities: frozenset[ParityVerificationCapability]
    qualified_scopes: frozenset[ParityHostScope]
    capability_requirements: tuple[
        tuple[ParityVerificationCapability, str],
        ...,
    ]


def proven_critic_retirement_qualification() -> CriticRetirementQualification:
    """Return the immutable retirement certificate evaluated from frozen evidence."""
    report = evaluate_critic_retirement_readiness_evidence(
        FROZEN_CRITIC_RETIREMENT_EVIDENCE,
        required_scopes=_REQUIRED_RETIREMENT_SCOPES,
        capability_requirements=DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    )
    capability_requirements = tuple(
        (requirement.capability, requirement.mode.value)
        for requirement in DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS
    )
    qualified_capabilities = (
        report.cross_system_capabilities_qualified
        | report.decision_superset_capabilities_qualified
        | report.architectural_mappings_qualified
    )
    return CriticRetirementQualification(
        report=report,
        provenance=CriticRetirementEvidenceProvenance.HISTORICAL_PRE_RETIREMENT_QUALIFICATION,
        parity_qualification_commit=PARITY_QUALIFICATION_SOURCE_COMMIT,
        ds_mig_03_hitl_transition_commit=DS_MIG_03_HITL_TRANSITION_COMMIT,
        final_regression_gate_commit=FINAL_PRE_RETIREMENT_REGRESSION_COMMIT,
        qualified_capabilities=qualified_capabilities,
        qualified_scopes=report.scopes_exercised,
        capability_requirements=capability_requirements,
    )
