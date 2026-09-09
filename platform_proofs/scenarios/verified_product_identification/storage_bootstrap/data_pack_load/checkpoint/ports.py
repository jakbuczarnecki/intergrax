"""Provider-neutral checkpoint store port for storage bootstrap resume."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
    BootstrapCheckpointState,
    BootstrapRunIdentity,
)


class BootstrapCheckpointStorePort(Protocol):
    """Durable logical-batch commit checkpoint — single-host filesystem safe by default."""

    def load(self, run_identity: BootstrapRunIdentity) -> BootstrapCheckpointState | None: ...

    def initialize(self, run_identity: BootstrapRunIdentity, state: BootstrapCheckpointState) -> BootstrapCheckpointState: ...

    def commit_batch(
        self,
        *,
        run_identity: BootstrapRunIdentity,
        expected_revision: int,
        checkpoint: BootstrapCheckpointState,
    ) -> BootstrapCheckpointState: ...
