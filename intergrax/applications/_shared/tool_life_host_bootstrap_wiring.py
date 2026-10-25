# © Artur Czarnecki. All rights reserved.

"""G-A07: canonical host bootstrap wiring for TOOL-LIFE durable composition (default-off)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.tool_life_execution_recovery_wiring import (
    TOOL_LIFE_DURABLE_DB_NAME,
    ApplicationToolLifeExecutionRecoveryWiring,
    wire_tool_life_durable_single_host_execution_recovery,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.persistence_topology import PersistenceTopology

_TOOL_LIFE_DURABLE_WIRING_ENABLED_ENV = "INTERGRAX_TOOL_LIFE_DURABLE_WIRING_ENABLED"


@dataclass(frozen=True, slots=True)
class ToolLifeHostBootstrapWiringResolution:
    """Distinguishes configuration from effective materialized composition."""

    configuration_enabled: bool
    topology_eligible: bool
    wiring: ApplicationToolLifeExecutionRecoveryWiring | None


def is_tool_life_durable_wiring_configuration_enabled() -> bool:
    raw = os.environ.get(_TOOL_LIFE_DURABLE_WIRING_ENABLED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_tool_life_durable_db_path_for_host(
    *,
    idempotency_db_path: Path | None,
    explicit_db_path: Path | None = None,
) -> Path | None:
    if explicit_db_path is not None:
        return explicit_db_path
    if idempotency_db_path is not None:
        return idempotency_db_path.parent / TOOL_LIFE_DURABLE_DB_NAME
    return None


def resolve_tool_life_host_bootstrap_wiring(
    environment: ApplicationEnvironmentProfile,
    *,
    idempotency_db_path: Path | None = None,
    tool_life_durable_db_path: Path | None = None,
) -> ToolLifeHostBootstrapWiringResolution:
    """
    Materialize certified owner-loss composition only when explicitly enabled and
    topology is DURABLE_SINGLE_HOST. Does not activate recovery or mint issuers.
    """
    configuration_enabled = is_tool_life_durable_wiring_configuration_enabled()
    topology = environment.meta.required_persistence_topology
    topology_eligible = topology is PersistenceTopology.DURABLE_SINGLE_HOST
    if not configuration_enabled or not topology_eligible:
        return ToolLifeHostBootstrapWiringResolution(
            configuration_enabled=configuration_enabled,
            topology_eligible=topology_eligible,
            wiring=None,
        )
    db_path = resolve_tool_life_durable_db_path_for_host(
        idempotency_db_path=idempotency_db_path,
        explicit_db_path=tool_life_durable_db_path,
    )
    if db_path is None:
        return ToolLifeHostBootstrapWiringResolution(
            configuration_enabled=configuration_enabled,
            topology_eligible=topology_eligible,
            wiring=None,
        )
    wiring = wire_tool_life_durable_single_host_execution_recovery(
        required_persistence_topology=topology,
        db_path=db_path,
    )
    return ToolLifeHostBootstrapWiringResolution(
        configuration_enabled=configuration_enabled,
        topology_eligible=topology_eligible,
        wiring=wiring,
    )


__all__ = [
    "ToolLifeHostBootstrapWiringResolution",
    "is_tool_life_durable_wiring_configuration_enabled",
    "resolve_tool_life_durable_db_path_for_host",
    "resolve_tool_life_host_bootstrap_wiring",
]
