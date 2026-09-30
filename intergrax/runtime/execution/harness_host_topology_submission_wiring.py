# © Artur Czarnecki. All rights reserved.

"""Strict production topology submission from orchestration materialization (EE only)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.orchestration_topology import OrchestrationTopologySubmissionPort
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)
from intergrax.runtime.execution.orchestration_topology_production_composition import (
    build_strict_production_orchestration_topology_submission_port,
)


def build_strict_topology_submission_from_materialization(
    materialization: EnvironmentOrchestrationMaterialization,
    *,
    provider_invocation_store: ProviderInvocationStore,
    tenant_id: str,
    clock: Callable[[], datetime],
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
) -> OrchestrationTopologySubmissionPort[object, object]:
    backend = orchestration_backend_for_execution_engine(materialization)
    return build_strict_production_orchestration_topology_submission_port(
        backend,
        provider_invocation_store=provider_invocation_store,
        tenant_id=tenant_id,
        clock=clock,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
    )


__all__ = ["build_strict_topology_submission_from_materialization"]
