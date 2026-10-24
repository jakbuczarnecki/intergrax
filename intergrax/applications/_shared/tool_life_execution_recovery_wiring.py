# © Artur Czarnecki. All rights reserved.

"""Tier-3 wiring for TOOL-LIFE-X durable single-host execution/recovery (L3-R10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.runtime.external_operations.tool_life_durable_single_host_composition import (
    ToolLifeDurableSingleHostComposition,
    ToolLifeDurableTopologyNotSupportedError,
    compose_tool_life_durable_single_host_certified_owner_loss,
)

TOOL_LIFE_DURABLE_DB_NAME = "intergrax_tool_life_durable.db"


@dataclass(frozen=True, slots=True)
class ApplicationToolLifeExecutionRecoveryWiring:
    """Resolved durable TOOL-LIFE-X execution/recovery composition for one host."""

    composition: ToolLifeDurableSingleHostComposition
    db_path: Path


def resolve_tool_life_durable_db_path(
    data_home: Path,
    *,
    explicit_db_path: Path | None = None,
) -> Path:
    if explicit_db_path is not None:
        return explicit_db_path
    return data_home / "data" / "sqlite" / TOOL_LIFE_DURABLE_DB_NAME


def wire_tool_life_durable_single_host_execution_recovery(
    *,
    required_persistence_topology: PersistenceTopology,
    db_path: Path,
) -> ApplicationToolLifeExecutionRecoveryWiring | None:
    """Materialize sanctioned composition when host topology is DURABLE_SINGLE_HOST."""
    if required_persistence_topology is not PersistenceTopology.DURABLE_SINGLE_HOST:
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        composition = compose_tool_life_durable_single_host_certified_owner_loss(
            db_path=db_path,
            persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        )
    except ToolLifeDurableTopologyNotSupportedError:
        return None
    return ApplicationToolLifeExecutionRecoveryWiring(
        composition=composition,
        db_path=db_path,
    )


__all__ = [
    "ApplicationToolLifeExecutionRecoveryWiring",
    "TOOL_LIFE_DURABLE_DB_NAME",
    "resolve_tool_life_durable_db_path",
    "wire_tool_life_durable_single_host_execution_recovery",
]
