# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production composition for SQLite DURABLE_SINGLE_HOST TOOL-LIFE-X stores (L3-R9)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.autonomous_work.repository import WorkerInstanceRepository
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.runtime.external_operations.execution_recovery_external_operation_worker_revocation import (
    ExecutionRecoveryAdmittedCommitAuthorityLossPolicy,
    ExecutionRecoveryExternalOperationWorkerRevocation,
)
from intergrax.runtime.external_operations.execution_recovery_worker_execution_revocation_supervision import (
    ExecutionRecoveryWorkerExecutionRevocationSupervision,
    WorkerExecutionCommitAuthorityLossPolicy,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    configure_worker_execution_certified_owner_loss,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation_trust import (
    WorkerExecutionOwnershipRevocationTrustMaterial,
    configure_worker_execution_ownership_revocation_trust,
    resolve_worker_execution_revocation_trust_from_environment,
)


class ToolLifeDurableTopologyNotSupportedError(RuntimeError):
    """Requested persistence topology cannot host sanctioned revocation supervision."""


@dataclass(frozen=True, slots=True)
class ToolLifeDurableSingleHostComposition:
    bundle: SqliteToolLifeDurableStoresBundle
    worker_revocation: ExecutionRecoveryExternalOperationWorkerRevocation


def compose_tool_life_durable_single_host_certified_owner_loss(
    *,
    db_path: Path,
    persistence_topology: PersistenceTopology,
    worker_instance_repository: WorkerInstanceRepository | None = None,
    loss_policy: WorkerExecutionCommitAuthorityLossPolicy | None = None,
    trust_material: WorkerExecutionOwnershipRevocationTrustMaterial | None = None,
) -> ToolLifeDurableSingleHostComposition:
    if persistence_topology is not PersistenceTopology.DURABLE_SINGLE_HOST:
        raise ToolLifeDurableTopologyNotSupportedError(
            "worker execution revocation supervision requires DURABLE_SINGLE_HOST",
        )
    trust = trust_material or resolve_worker_execution_revocation_trust_from_environment()
    configure_worker_execution_ownership_revocation_trust(trust)
    bundle = SqliteToolLifeDurableStoresBundle(db_path=db_path)
    policy = loss_policy or ExecutionRecoveryAdmittedCommitAuthorityLossPolicy()
    supervision = ExecutionRecoveryWorkerExecutionRevocationSupervision(
        loss_policy=policy,
        trust=trust,
    )
    configure_worker_execution_certified_owner_loss(
        worker_execution_ownership_revocation=bundle.worker_execution_ownership_revocation,
        worker_instance_repository=worker_instance_repository,
        authoritative_state_store=bundle.external_operation_state,
        worker_execution_revocation_supervision=supervision,
    )
    from intergrax.runtime.external_operations.certified_owner_loss_composition import (
        require_certified_owner_loss_composition,
    )

    composition = require_certified_owner_loss_composition()
    recorder = composition.worker_execution_revocation_recorder
    if recorder is None:
        raise ToolLifeDurableTopologyNotSupportedError(
            "revocation recorder was not registered after composition",
        )
    return ToolLifeDurableSingleHostComposition(
        bundle=bundle,
        worker_revocation=ExecutionRecoveryExternalOperationWorkerRevocation(
            recorder=recorder,
        ),
    )


__all__ = [
    "ToolLifeDurableSingleHostComposition",
    "ToolLifeDurableTopologyNotSupportedError",
    "compose_tool_life_durable_single_host_certified_owner_loss",
]
