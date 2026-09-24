# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Environment-profile host task execution materialization (runtime-owned Nexus coupling)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.agents.persistence.skill_host_wiring import HostSkillCatalogWiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
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
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration_capabilities import (
    orchestration_capabilities_from_triggers,
)
from intergrax.runtime.task.task import Task

__all__ = ["build_environment_host_task_execution"]


def build_environment_host_task_execution(
    nexus_loop: NexusLoop,
    env: ApplicationEnvironmentProfile,
    *,
    orchestration_triggers: frozenset[str] | None = None,
    pipeline_capability_suffix: str | None = None,
    revision_admission: EffectiveProfileRevisionAdmissionPort | None = None,
    root_authority_admission: RootExecutionAuthorityAdmissionPort,
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity],
    skill_host_wiring: HostSkillCatalogWiring | None = None,
) -> HostTaskExecution:
    """Build canonical host task execution from environment orchestration profile."""
    graph_spec = env.graph_spec
    return build_host_task_execution(
        nexus_loop,
        orchestration_triggers=(
            orchestration_triggers
            if orchestration_triggers is not None
            else orchestration_capabilities_from_triggers(
                graph_spec.trigger_capabilities if graph_spec is not None else None,
            )
        ),
        pipeline_capability_suffix=(
            pipeline_capability_suffix
            if pipeline_capability_suffix is not None
            else (
                graph_spec.pipeline_capability_suffix
                if graph_spec is not None
                else ".pipeline"
            )
        ),
        revision_admission=revision_admission,
        root_authority_admission=root_authority_admission,
        admit_root_governance_identity=admit_root_governance_identity,
        skill_host_wiring=skill_host_wiring,
    )
