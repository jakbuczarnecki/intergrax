# © Artur Czarnecki. All rights reserved.

"""Strict production orchestration topology capability on harness host runtime (GR-10-R13-R3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from intergrax.applications._shared.harness_orchestration_topology_production_composition import (
    build_harness_host_production_orchestration_topology_submission_port,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.orchestration_topology import OrchestrationTopologySubmissionPort
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime


class HarnessHostOrchestrationTopologyReliabilityCompositionError(ValueError):
    """Strict production host cannot expose topology Reliability without durable store."""


class StrictOrchestrationTopologySubmissionPortBuilder(Protocol):
    """Product composition root builds strict topology submission from host inputs."""

    def __call__(
        self,
        nexus_loop: NexusLoop,
        *,
        provider_invocation_store: ProviderInvocationStore,
        tenant_id: str,
        clock: Callable[[], datetime],
        meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
    ) -> OrchestrationTopologySubmissionPort[object, object]: ...


@dataclass(frozen=True, slots=True)
class HarnessHostOrchestrationTopologyWiring:
    """Typed runtime capability: canonical strict production topology submission port."""

    submission_port: OrchestrationTopologySubmissionPort[object, object]
    provider_invocation_store: ProviderInvocationStore


def build_harness_host_orchestration_topology_wiring(
    nexus_loop: NexusLoop,
    *,
    provider_invocation_store: ProviderInvocationStore | None,
    tenant_id: str,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
    submission_port_builder: StrictOrchestrationTopologySubmissionPortBuilder | None = None,
    clock: Callable[[], datetime] | None = None,
) -> HarnessHostOrchestrationTopologyWiring:
    """Fail-closed strict production topology Reliability wiring for harness hosts."""
    if provider_invocation_store is None:
        raise HarnessHostOrchestrationTopologyReliabilityCompositionError(
            "strict production harness host orchestration topology requires "
            "ProviderInvocationStore at composition time",
        )
    if not provider_invocation_store.is_durable:
        raise HarnessHostOrchestrationTopologyReliabilityCompositionError(
            "strict production harness host orchestration topology requires "
            "durable ProviderInvocationStore (is_durable must be True)",
        )
    resolved_tenant = tenant_id.strip()
    if not resolved_tenant:
        raise HarnessHostOrchestrationTopologyReliabilityCompositionError(
            "strict production harness host orchestration topology requires tenant_id",
        )
    resolved_clock = clock or (lambda: datetime.now(timezone.utc))
    builder = submission_port_builder or (
        build_harness_host_production_orchestration_topology_submission_port
    )
    submission_port = builder(
        nexus_loop,
        provider_invocation_store=provider_invocation_store,
        tenant_id=resolved_tenant,
        clock=resolved_clock,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
    )
    return HarnessHostOrchestrationTopologyWiring(
        submission_port=submission_port,
        provider_invocation_store=provider_invocation_store,
    )


def resolve_harness_host_orchestration_topology_wiring(
    runtime: HarnessHostRuntime,
) -> HarnessHostOrchestrationTopologyWiring:
    """Resolve wired strict production topology capability (fail-closed when absent)."""
    wiring = runtime.orchestration_topology
    if wiring is None:
        raise HarnessHostOrchestrationTopologyReliabilityCompositionError(
            "harness host runtime has no orchestration topology Reliability capability",
        )
    return wiring


__all__ = [
    "HarnessHostOrchestrationTopologyReliabilityCompositionError",
    "HarnessHostOrchestrationTopologyWiring",
    "StrictOrchestrationTopologySubmissionPortBuilder",
    "build_harness_host_orchestration_topology_wiring",
    "resolve_harness_host_orchestration_topology_wiring",
]
