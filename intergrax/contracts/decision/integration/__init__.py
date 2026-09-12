# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision System integration boundary (DS-E2E-15J-DECISION-SYSTEM-INTEGRATION-BOUNDARY)."""

from intergrax.contracts.decision.integration.audit import (
    DecisionIntegrationAuditProvider,
    DecisionIntegrationAuditRecord,
    DefaultDecisionIntegrationAuditProvider,
)
from intergrax.contracts.decision.integration.composition import (
    ConfiguredDecisionIntegrationCompositionProvider,
    DecisionIntegrationCompositionProvider,
    DecisionIntegrationCompositionSpec,
)
from intergrax.contracts.decision.integration.engine import (
    DecisionSystemIntegrationEngine,
)
from intergrax.contracts.decision.integration.factory import (
    DecisionSystemIntegrationFactory,
)
from intergrax.contracts.decision.integration.lifecycle import (
    DefaultDecisionLifecycleIntegrationAdapter,
    DefaultLifecycleAdapterProvider,
    SingleLifecycleAdapterProvider,
)
from intergrax.contracts.decision.integration.protocol import (
    DecisionIntegrationAdapterProvider,
    DecisionLifecycleIntegrationAdapter,
    DecisionSystemIntegrationAdapter,
)
from intergrax.contracts.decision.integration.references import (
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    PlatformDecisionLifecycleReference,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
)
from intergrax.contracts.decision.integration.result import (
    DecisionAdapterMetadata,
    DecisionIntegrationResult,
    DecisionIntegrationStatus,
)

__all__ = [
    "ConfiguredDecisionIntegrationCompositionProvider",
    "DecisionAdapterMetadata",
    "DecisionIntegrationAdapterProvider",
    "DecisionIntegrationAuditProvider",
    "DecisionIntegrationAuditRecord",
    "DecisionIntegrationCompositionProvider",
    "DecisionIntegrationCompositionSpec",
    "DecisionIntegrationResult",
    "DecisionIntegrationStatus",
    "DecisionLifecycleIntegrationAdapter",
    "DecisionSystemIntegrationAdapter",
    "DecisionSystemIntegrationEngine",
    "DecisionSystemIntegrationFactory",
    "DefaultDecisionIntegrationAuditProvider",
    "DefaultDecisionLifecycleIntegrationAdapter",
    "DefaultLifecycleAdapterProvider",
    "PlatformDecisionLifecycleReference",
    "REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE",
    "ReferenceDecisionLifecycleReference",
    "ReferenceEnterpriseLifecycleState",
    "SingleLifecycleAdapterProvider",
]
