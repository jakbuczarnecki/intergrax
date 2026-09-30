# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Governed contractor strict production orchestration topology composition (GR-10-R13-R2)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.orchestration_topology import OrchestrationTopologySubmissionPort
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)
from intergrax.runtime.execution.harness_host_topology_submission_wiring import (
    build_strict_topology_submission_from_materialization,
)


def build_governed_contractor_production_orchestration_topology_submission_port(
    orchestration: EnvironmentOrchestrationMaterialization,
    *,
    provider_invocation_store: ProviderInvocationStore,
    tenant_id: str,
    clock: Callable[[], datetime],
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
) -> OrchestrationTopologySubmissionPort[object, object]:
    """Production topology submission sharing the host ``ProviderInvocationStore`` with GR-7."""
    return build_strict_topology_submission_from_materialization(
        orchestration,
        provider_invocation_store=provider_invocation_store,
        tenant_id=tenant_id,
        clock=clock,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
    )


__all__ = [
    "build_governed_contractor_production_orchestration_topology_submission_port",
]
