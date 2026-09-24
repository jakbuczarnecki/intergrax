# © Artur Czarnecki. All rights reserved.

"""Test/lab helpers — Nexus-backed harness host execution (not production shared API)."""

from __future__ import annotations

from intergrax.agents.persistence.skill_host_wiring import HostSkillCatalogWiring
from intergrax.applications._shared.harness_host_task_execution_wiring import (
    build_harness_host_task_execution_governance,
    resolve_harness_effective_profile_revision_admission,
)
from intergrax.applications._shared.profile_resolution.execution_admission import (
    EffectiveProfileExecutionPinningDependencies,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.runtime.execution.effective_profile_revision_admission import (
    EffectiveProfileRevisionAdmissionPort,
)
from intergrax.runtime.execution.environment_host_task_execution import (
    build_environment_host_task_execution,
)
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.nexus.nexus_loop import NexusLoop

__all__ = [
    "build_harness_environment_host_task_execution",
    "build_harness_host_task_execution",
]


def build_harness_host_task_execution(
    nexus_loop: NexusLoop,
    *,
    orchestration_triggers: frozenset[str],
    pipeline_capability_suffix: str = ".pipeline",
    revision_admission: EffectiveProfileRevisionAdmissionPort | None = None,
    skill_host_wiring: HostSkillCatalogWiring | None = None,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
) -> HostTaskExecution:
    governance = build_harness_host_task_execution_governance(
        governance_evidence_recorder=governance_evidence_recorder,
    )
    return build_host_task_execution(
        nexus_loop,
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_capability_suffix,
        revision_admission=revision_admission,
        root_authority_admission=governance.root_authority_admission,
        admit_root_governance_identity=governance.admit_root_governance_identity,
        skill_host_wiring=skill_host_wiring,
    )


def build_harness_environment_host_task_execution(
    nexus_loop: NexusLoop,
    env: ApplicationEnvironmentProfile,
    *,
    pinning_dependencies: EffectiveProfileExecutionPinningDependencies | None = None,
    skill_host_wiring: HostSkillCatalogWiring | None = None,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
) -> HostTaskExecution:
    governance = build_harness_host_task_execution_governance(
        governance_evidence_recorder=governance_evidence_recorder,
    )
    return build_environment_host_task_execution(
        nexus_loop,
        env,
        revision_admission=resolve_harness_effective_profile_revision_admission(
            pinning_dependencies,
        ),
        skill_host_wiring=skill_host_wiring,
        root_authority_admission=governance.root_authority_admission,
        admit_root_governance_identity=governance.admit_root_governance_identity,
    )
