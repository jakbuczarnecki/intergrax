# © Artur Czarnecki. All rights reserved.

"""Host task execution from neutral wiring target (EE-only Nexus cast)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.agents.persistence.skill_host_wiring import HostSkillCatalogWiring
from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)
from intergrax.contracts.runtime_execution_admission import RootExecutionAuthorityAdmissionPort
from intergrax.runtime.execution.effective_profile_revision_admission import (
    EffectiveProfileRevisionAdmissionPort,
)
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task


def build_host_task_execution_from_wiring_target(
    host: HostOrchestrationApplicationWiringTarget,
    *,
    orchestration_triggers: frozenset[str],
    root_authority_admission: RootExecutionAuthorityAdmissionPort,
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity],
    pipeline_capability_suffix: str = ".pipeline",
    revision_admission: EffectiveProfileRevisionAdmissionPort | None = None,
    skill_host_wiring: HostSkillCatalogWiring | None = None,
) -> HostTaskExecution:
    if not isinstance(host, NexusLoop):
        raise TypeError("host task execution requires internal orchestration host materialization")
    return build_host_task_execution(
        host,
        orchestration_triggers=orchestration_triggers,
        root_authority_admission=root_authority_admission,
        admit_root_governance_identity=admit_root_governance_identity,
        pipeline_capability_suffix=pipeline_capability_suffix,
        revision_admission=revision_admission,
        skill_host_wiring=skill_host_wiring,
    )


__all__ = ["build_host_task_execution_from_wiring_target"]
