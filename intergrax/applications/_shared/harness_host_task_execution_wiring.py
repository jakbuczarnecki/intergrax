# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit harness/lab host task execution composition (not generic production wiring)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications._shared.harness_admitted_root_governance_identity import (
    admit_harness_root_governance_identity,
)
from intergrax.applications._shared.harness_root_execution_launch_wiring import (
    build_harness_root_execution_authority_admission,
)
from intergrax.runtime.execution.environment_host_task_execution import (
    build_environment_host_task_execution,
)
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.applications._shared.profile_resolution.execution_admission import (
    EffectiveProfileExecutionPinningDependencies,
    build_effective_profile_revision_admission,
)
from intergrax.agents.persistence.skill_host_wiring import HostSkillCatalogWiring
from intergrax.runtime.execution.host_task import HostTaskExecution
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
    revision_admission: object | None = None,
    skill_host_wiring: HostSkillCatalogWiring | None = None,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
) -> HostTaskExecution:
    """Harness composition root — explicit harness governance identity admission."""
    return build_host_task_execution(
        nexus_loop,
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_capability_suffix,
        revision_admission=revision_admission,
        root_authority_admission=build_harness_root_execution_authority_admission(
            governance_evidence_recorder=governance_evidence_recorder,
        ),
        admit_root_governance_identity=admit_harness_root_governance_identity,
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
    """Build harness host task execution from environment orchestration profile."""
    revision_admission = None
    if pinning_dependencies is not None:
        revision_admission = build_effective_profile_revision_admission(
            pinning_dependencies,
        )
    return build_environment_host_task_execution(
        nexus_loop,
        env,
        revision_admission=revision_admission,
        skill_host_wiring=skill_host_wiring,
        root_authority_admission=build_harness_root_execution_authority_admission(
            governance_evidence_recorder=governance_evidence_recorder,
        ),
        admit_root_governance_identity=admit_harness_root_governance_identity,
    )
