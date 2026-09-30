# © Artur Czarnecki. All rights reserved.

"""Test-only helper — Nexus materialization must go through Execution Engine in production."""

from __future__ import annotations

from intergrax.applications._shared.host_orchestration_backend_spec_builder import (
    build_host_orchestration_loop_init_spec_from_environment,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    materialize_host_orchestration_backend,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


def build_nexus_loop_from_environment(
    registry: AgentRegistryRead,
    **kwargs: object,
) -> NexusLoop:
    spec = build_host_orchestration_loop_init_spec_from_environment(registry, **kwargs)
    materialization = materialize_host_orchestration_backend(registry, spec)
    return materialization.orchestration_backend_for_host_wiring()


__all__ = ["build_nexus_loop_from_environment"]
