# © Artur Czarnecki. All rights reserved.

"""Process-scoped GCA host store bundle for strict external-work composition (F01)."""

from __future__ import annotations

from dataclasses import dataclass

from governed_contractor_application.host.stores import (
    ContinuationStateStore,
    GovernedExecutionStore,
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
    PolicyBundleArtifactStore,
    ProofReceiptStore,
)


@dataclass(slots=True)
class GovernedExternalWorkHostStoreBundle:
    """Exactly-one process-scoped instances for orchestrator host persistence."""

    execution_store: GovernedExecutionStore
    receipt_store: ProofReceiptStore
    bundle_store: PolicyBundleArtifactStore
    continuation_store: ContinuationStateStore


def build_process_scoped_governed_external_work_host_stores() -> GovernedExternalWorkHostStoreBundle:
    """Mint one in-memory bundle per strict external-work session (shared across requests)."""
    return GovernedExternalWorkHostStoreBundle(
        execution_store=InMemoryGovernedExecutionStore(),
        receipt_store=InMemoryProofReceiptStore(),
        bundle_store=InMemoryPolicyBundleArtifactStore(),
        continuation_store=InMemoryContinuationStateStore(),
    )


__all__ = [
    "GovernedExternalWorkHostStoreBundle",
    "build_process_scoped_governed_external_work_host_stores",
]
