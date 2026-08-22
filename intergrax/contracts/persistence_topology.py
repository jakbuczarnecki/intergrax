# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical persistence deployment topology classification (PCM R3-A)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.contracts.idempotency_store import IdempotencyStore


class PersistenceTopology(str, Enum):
    """
    Storage capability classes for stateful runtime mechanisms.

    PROCESS_LOCAL — state exists only inside one process; lost on restart.
    DURABLE_SINGLE_HOST — survives process restart on one host (e.g. local SQLite).
    SHARED_MULTI_HOST — multiple workers observe one authoritative shared state.

    Orthogonal to ExecutionMode: STRICT does not imply SHARED_MULTI_HOST.
    """

    PROCESS_LOCAL = "process_local"
    DURABLE_SINGLE_HOST = "durable_single_host"
    SHARED_MULTI_HOST = "shared_multi_host"


class DeploymentTopology(str, Enum):
    """
    Host/operator physical deployment classification.

    PROCESS_LOCAL — single process; no durable host-local requirement.
    SINGLE_HOST — one host; durable local storage may satisfy persistence.
    MULTI_HOST — multiple hosts/workers; shared storage is required.

    Orthogonal to ExecutionMode: STRICT does not imply MULTI_HOST.
    """

    PROCESS_LOCAL = "process_local"
    SINGLE_HOST = "single_host"
    MULTI_HOST = "multi_host"


_DEPLOYMENT_TO_REQUIRED: dict[DeploymentTopology, PersistenceTopology] = {
    DeploymentTopology.PROCESS_LOCAL: PersistenceTopology.PROCESS_LOCAL,
    DeploymentTopology.SINGLE_HOST: PersistenceTopology.DURABLE_SINGLE_HOST,
    DeploymentTopology.MULTI_HOST: PersistenceTopology.SHARED_MULTI_HOST,
}


def required_persistence_for_deployment(
    deployment: DeploymentTopology | str,
) -> PersistenceTopology:
    """Derive the persistence capability required by a host deployment topology."""
    if not isinstance(deployment, DeploymentTopology):
        deployment = DeploymentTopology(deployment)
    return _DEPLOYMENT_TO_REQUIRED[deployment]


_TOPOLOGY_RANK: dict[PersistenceTopology, int] = {
    PersistenceTopology.PROCESS_LOCAL: 0,
    PersistenceTopology.DURABLE_SINGLE_HOST: 1,
    PersistenceTopology.SHARED_MULTI_HOST: 2,
}


def topology_satisfies(
    required: PersistenceTopology,
    provided: PersistenceTopology,
) -> bool:
    """Return True when ``provided`` meets or exceeds ``required`` capability."""
    return _TOPOLOGY_RANK[provided] >= _TOPOLOGY_RANK[required]


def resolve_idempotency_store_topology(
    store: IdempotencyStore | None,
) -> PersistenceTopology | None:
    """
    Resolve declared idempotency-store topology.

    Returns None when the store omits canonical capability — fail-closed input.
    """
    if store is None:
        return None
    try:
        topology = store.persistence_topology
    except (AttributeError, NotImplementedError):
        return None
    if isinstance(topology, PersistenceTopology):
        return topology
    return None


def format_topology_mismatch_error(
    *,
    mechanism: str,
    required: PersistenceTopology,
    provided: PersistenceTopology | None,
) -> str:
    """Diagnostic error without connection secrets."""
    provided_label = provided.value if provided is not None else "unknown"
    return (
        f"{mechanism} persistence topology mismatch: "
        f"required={required.value} provided={provided_label}"
    )
