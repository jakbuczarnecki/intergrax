# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition-only wiring for worker capability fulfillment (UCA-6C-R6-R5.8-R2-H1-R1)."""

from __future__ import annotations

from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_capability_fulfillment_ports import (
    CapabilityRealizationCoordinatorPort,
    WorkerCapabilityDirectReuseFulfillmentPort,
    WorkerCapabilityRecoveryPort,
    WorkerQualifiedCapabilityResumePort,
)
from intergrax.contracts.tools.qualified_capability_execution_intent_preparation import (
    QualifiedCapabilityExecutionIntentPreparationPort,
)


def build_worker_capability_fulfillment_coordinator(
    *,
    recovery: WorkerCapabilityRecoveryPort,
    resume: WorkerQualifiedCapabilityResumePort,
    direct_reuse: WorkerCapabilityDirectReuseFulfillmentPort,
    realization: CapabilityRealizationCoordinatorPort | None = None,
    intent_preparation: QualifiedCapabilityExecutionIntentPreparationPort | None = None,
) -> WorkerCapabilityFulfillmentCoordinator:
    """Construct fulfillment coordinator with explicit port dependencies only."""
    return WorkerCapabilityFulfillmentCoordinator(
        recovery=recovery,
        resume=resume,
        direct_reuse=direct_reuse,
        realization=realization,
        intent_preparation=intent_preparation,
    )


__all__ = ["build_worker_capability_fulfillment_coordinator"]
