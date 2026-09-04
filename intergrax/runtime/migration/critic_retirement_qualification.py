# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable Critic retirement qualification certificate (DS-MIG-04).

Encodes pre-retirement parity evidence with explicit provenance. This is not a live
shadow comparison — it is the frozen qualification record that justified deleting
``intergrax/runtime/critic/**``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadiness,
    CriticRetirementReadinessReport,
    DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    ParityHostScope,
    ParityVerificationCapability,
)


PARITY_QUALIFICATION_SOURCE_COMMIT = "a0a1d2ac6a566915ce36007f85feb8242f31703b"
DS_MIG_03_HITL_TRANSITION_COMMIT = "f7820a3e9abdb65af70d9f09ce439e3272582bbc"
FINAL_PRE_RETIREMENT_REGRESSION_COMMIT = "cfa0461f2c44d1a3a0e8171469f9eac3c3d3836f"


class CriticRetirementEvidenceProvenance(str, Enum):
    HISTORICAL_PRE_RETIREMENT_QUALIFICATION = "historical_pre_retirement_qualification"


@dataclass(frozen=True, slots=True)
class CriticRetirementQualification:
    """Immutable aggregate proving safe Critic runtime retirement."""

    readiness: CriticRetirementReadiness
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
    """Return the immutable retirement certificate from qualified pre-cut evidence."""
    cross_system = frozenset({
        ParityVerificationCapability.STRUCTURAL,
        ParityVerificationCapability.DETERMINISTIC_GUARDRAIL,
        ParityVerificationCapability.SEMANTIC,
        ParityVerificationCapability.TRAJECTORY,
    })
    decision_superset = frozenset({
        ParityVerificationCapability.EVIDENCE,
        ParityVerificationCapability.DOMAIN,
    })
    architectural = frozenset({ParityVerificationCapability.HUMAN_HITL})
    scopes = frozenset({
        ParityHostScope.GRAPH_FINAL,
        ParityHostScope.UAEP_STEP,
    })
    report = CriticRetirementReadinessReport(
        readiness=CriticRetirementReadiness.READY,
        blocking_mismatch_count=0,
        shadow_error_count=0,
        shadow_unavailable_count=1,
        scopes_exercised=scopes,
        decision_capabilities_exercised=cross_system | decision_superset | architectural,
        critic_capabilities_exercised=cross_system | architectural,
        cross_system_capabilities_qualified=cross_system,
        decision_superset_capabilities_qualified=decision_superset,
        architectural_mappings_qualified=architectural,
        missing_scopes=frozenset(),
        missing_capabilities=frozenset(),
    )
    capability_requirements = tuple(
        (requirement.capability, requirement.mode.value)
        for requirement in DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS
    )
    return CriticRetirementQualification(
        readiness=report.readiness,
        report=report,
        provenance=CriticRetirementEvidenceProvenance.HISTORICAL_PRE_RETIREMENT_QUALIFICATION,
        parity_qualification_commit=PARITY_QUALIFICATION_SOURCE_COMMIT,
        ds_mig_03_hitl_transition_commit=DS_MIG_03_HITL_TRANSITION_COMMIT,
        final_regression_gate_commit=FINAL_PRE_RETIREMENT_REGRESSION_COMMIT,
        qualified_capabilities=cross_system | decision_superset | architectural,
        qualified_scopes=scopes,
        capability_requirements=capability_requirements,
    )
