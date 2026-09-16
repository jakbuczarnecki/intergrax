# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Integration Boundary namespace (MP-4B SPI retired — canonical contracts are top-level modules)."""

from __future__ import annotations

from intergrax.contracts.decision.integration import (
    DecisionAdapterMetadata,
    DecisionIntegrationAdapterProvider,
    DecisionIntegrationAuditProvider,
    DecisionIntegrationAuditRecord,
    DecisionIntegrationResult,
    DecisionIntegrationStatus,
    DecisionLifecycleIntegrationAdapter,
    DecisionSystemIntegrationAdapter,
    DecisionSystemIntegrationEngine,
    PlatformDecisionLifecycleReference,
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
)

__all__ = (
    "DecisionAdapterMetadata",
    "DecisionIntegrationAdapterProvider",
    "DecisionIntegrationAuditProvider",
    "DecisionIntegrationAuditRecord",
    "DecisionIntegrationResult",
    "DecisionIntegrationStatus",
    "DecisionLifecycleIntegrationAdapter",
    "DecisionSystemIntegrationAdapter",
    "DecisionSystemIntegrationEngine",
    "PlatformDecisionLifecycleReference",
    "REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE",
    "ReferenceDecisionLifecycleReference",
    "ReferenceEnterpriseLifecycleState",
)
