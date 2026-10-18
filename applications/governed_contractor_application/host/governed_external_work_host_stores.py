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

    @property
    def is_durable(self) -> bool:
        return (
            getattr(self.execution_store, "is_durable", False)
            and getattr(self.receipt_store, "is_durable", False)
            and getattr(self.bundle_store, "is_durable", False)
            and getattr(self.continuation_store, "is_durable", False)
        )


def validate_production_governed_external_work_host_store_bundle(
    bundle: GovernedExternalWorkHostStoreBundle,
) -> None:
    """Fail closed when any GCA host store is not production-durable."""
    if not bundle.is_durable:
        raise ValueError(
            "production external work requires durable governed external work host stores "
            "(all execution/receipt/bundle/continuation stores must report is_durable=True); "
            "in-memory and lab/demo backends are forbidden on strict composition",
        )


def build_process_scoped_governed_external_work_host_stores() -> GovernedExternalWorkHostStoreBundle:
    """Mint in-memory bundle for lab/tests/offline_demo only — not strict production."""
    return GovernedExternalWorkHostStoreBundle(
        execution_store=InMemoryGovernedExecutionStore(),
        receipt_store=InMemoryProofReceiptStore(),
        bundle_store=InMemoryPolicyBundleArtifactStore(),
        continuation_store=InMemoryContinuationStateStore(),
    )


__all__ = [
    "GovernedExternalWorkHostStoreBundle",
    "build_process_scoped_governed_external_work_host_stores",
    "validate_production_governed_external_work_host_store_bundle",
]
