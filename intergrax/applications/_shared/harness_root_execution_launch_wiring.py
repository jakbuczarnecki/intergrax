# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness composition for canonical root execution launch (GR-2-R3)."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_admission import RootExecutionAuthorityAdmissionPort
from intergrax.runtime.governance.execution_admission_composition import (
    build_root_execution_authority_admission,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)


def build_harness_root_execution_authority_admission(
    *,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
) -> RootExecutionAuthorityAdmissionPort:
    """Permissive harness policy — production hosts must inject real policy ports."""
    return build_root_execution_authority_admission(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=governance_evidence_recorder,
    )
