# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness/lab governance composition for host task execution (no Nexus materialization)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.applications._shared.harness_admitted_root_governance_identity import (
    admit_harness_root_governance_identity,
)
from intergrax.applications._shared.harness_root_execution_launch_wiring import (
    build_harness_root_execution_authority_admission,
)
from intergrax.applications._shared.profile_resolution.execution_admission import (
    EffectiveProfileExecutionPinningDependencies,
    build_effective_profile_revision_admission,
)
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionPort,
)
from intergrax.runtime.execution.effective_profile_revision_admission import (
    EffectiveProfileRevisionAdmissionPort,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.task.task import Task

__all__ = [
    "HarnessHostTaskExecutionGovernance",
    "build_harness_host_task_execution_governance",
    "resolve_harness_effective_profile_revision_admission",
]


@dataclass(frozen=True, slots=True)
class HarnessHostTaskExecutionGovernance:
    """Harness-specific root authority and governance identity admission."""

    root_authority_admission: RootExecutionAuthorityAdmissionPort
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity]


def build_harness_host_task_execution_governance(
    *,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
) -> HarnessHostTaskExecutionGovernance:
    """Harness composition — explicit harness governance identity admission."""
    return HarnessHostTaskExecutionGovernance(
        root_authority_admission=build_harness_root_execution_authority_admission(
            governance_evidence_recorder=governance_evidence_recorder,
        ),
        admit_root_governance_identity=admit_harness_root_governance_identity,
    )


def resolve_harness_effective_profile_revision_admission(
    pinning_dependencies: EffectiveProfileExecutionPinningDependencies | None,
) -> EffectiveProfileRevisionAdmissionPort | None:
    """Resolve effective-profile revision admission for harness host execution builds."""
    if pinning_dependencies is None:
        return None
    return build_effective_profile_revision_admission(pinning_dependencies)
