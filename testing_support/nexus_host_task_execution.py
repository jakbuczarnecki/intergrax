# © Artur Czarnecki. All rights reserved.

"""Explicit certified-internal host task execution fixtures for tests (not production)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.runtime_execution_admission import RootExecutionAuthorityAdmissionPort
from intergrax.runtime.execution.certified_internal_harness_governance_identity import (
    admit_certified_internal_harness_root_governance_identity,
)
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.governance.execution_admission_composition import (
    build_reference_allowing_root_execution_authority_admission,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task

__all__ = [
    "admit_certified_internal_harness_root_governance_identity",
    "build_certified_internal_test_host_task_execution",
    "build_reference_allowing_root_execution_authority_admission",
]


def build_certified_internal_test_host_task_execution(
    nexus_loop: NexusLoop,
    *,
    orchestration_triggers: frozenset[str] | None = None,
    pipeline_capability_suffix: str = ".pipeline",
    root_authority_admission: RootExecutionAuthorityAdmissionPort | None = None,
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity] | None = None,
) -> HostTaskExecution:
    """Test/lab composition — explicit certified-internal harness identity admission."""
    return build_host_task_execution(
        nexus_loop,
        orchestration_triggers=orchestration_triggers or frozenset(),
        pipeline_capability_suffix=pipeline_capability_suffix,
        root_authority_admission=(
            root_authority_admission or build_reference_allowing_root_execution_authority_admission()
        ),
        admit_root_governance_identity=(
            admit_root_governance_identity
            or admit_certified_internal_harness_root_governance_identity
        ),
    )
