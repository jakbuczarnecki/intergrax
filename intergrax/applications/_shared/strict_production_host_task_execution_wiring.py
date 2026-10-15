# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strict production serving host task execution governance composition."""

from __future__ import annotations

from intergrax.applications._shared.harness_host_task_execution_wiring import (
    HarnessHostTaskExecutionGovernance,
)
from intergrax.applications._shared.harness_root_execution_launch_wiring import (
    build_harness_root_execution_authority_admission,
)
from intergrax.applications._shared.strict_production_serving_execution_tenant_authority import (
    StrictProductionServingExecutionTenantBinding,
    admit_strict_production_serving_root_governance_identity,
)
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.applications._shared.strict_production_serving_http_execution_principal import (
    get_strict_serving_http_caller_principal,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.task.task import Task


def build_strict_production_serving_host_task_execution_governance(
    *,
    binding: StrictProductionServingExecutionTenantBinding,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
) -> HarnessHostTaskExecutionGovernance:
    """Strict GCA serving — admitted identity from per-request HTTP caller principal."""

    def _admit(task: Task) -> AdmittedRootGovernanceIdentity:
        caller_principal = get_strict_serving_http_caller_principal()
        return admit_strict_production_serving_root_governance_identity(
            task,
            binding=binding,
            principal=caller_principal,
        )

    return HarnessHostTaskExecutionGovernance(
        root_authority_admission=build_harness_root_execution_authority_admission(
            governance_evidence_recorder=governance_evidence_recorder,
        ),
        admit_root_governance_identity=_admit,
    )


__all__ = [
    "build_strict_production_serving_host_task_execution_governance",
]
