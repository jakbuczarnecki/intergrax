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
from intergrax.runtime.execution.orchestration_topology_production_composition import (
    build_strict_production_orchestration_topology_submission_port,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def build_governed_contractor_production_orchestration_topology_submission_port(
    nexus_loop: NexusLoop,
    *,
    provider_invocation_store: ProviderInvocationStore,
    tenant_id: str,
    clock: Callable[[], datetime],
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
) -> OrchestrationTopologySubmissionPort[object, object]:
    """Production topology submission sharing the host ``ProviderInvocationStore`` with GR-7."""
    return build_strict_production_orchestration_topology_submission_port(
        nexus_loop,
        provider_invocation_store=provider_invocation_store,
        tenant_id=tenant_id,
        clock=clock,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
    )


__all__ = [
    "build_governed_contractor_production_orchestration_topology_submission_port",
]
