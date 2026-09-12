# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision System integration boundary (DS-E2E-15J-DECISION-SYSTEM-INTEGRATION-BOUNDARY)."""

from intergrax.contracts.decision.integration.admission import (
    DecisionIntegrationPluginIdentifiable,
    DecisionPluginAdmissionProvider,
    DefaultDecisionPluginAdmissionProvider,
    PluginAdmissionDecision,
    filter_admitted_adapter_providers,
    resolve_integration_plugin_descriptor,
)
from intergrax.contracts.decision.integration.audit import (
    DecisionIntegrationAuditProvider,
    DecisionIntegrationAuditRecord,
    DefaultDecisionIntegrationAuditProvider,
    RecordingDecisionIntegrationAuditProvider,
)
from intergrax.contracts.decision.integration.audit_sink import (
    DecisionAuditSink,
    DecisionIntegrationAuditEnvelope,
    InMemoryDecisionAuditSink,
)
from intergrax.contracts.decision.integration.metadata import (
    DecisionIntegrationPluginDescriptor,
    IntegrationAuditProviderMetadata,
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
    "DecisionAuditSink",
    "DecisionIntegrationAdapterProvider",
    "DecisionIntegrationAuditEnvelope",
    "DecisionIntegrationAuditProvider",
    "DecisionIntegrationAuditRecord",
    "DecisionIntegrationCompositionProvider",
    "DecisionIntegrationCompositionSpec",
    "DecisionIntegrationPluginDescriptor",
    "DecisionIntegrationPluginIdentifiable",
    "DecisionIntegrationResult",
    "DecisionIntegrationStatus",
    "DecisionLifecycleIntegrationAdapter",
    "DecisionPluginAdmissionProvider",
    "DecisionSystemIntegrationAdapter",
    "DecisionSystemIntegrationEngine",
    "DecisionSystemIntegrationFactory",
    "DefaultDecisionIntegrationAuditProvider",
    "DefaultDecisionLifecycleIntegrationAdapter",
    "DefaultDecisionPluginAdmissionProvider",
    "DefaultLifecycleAdapterProvider",
    "InMemoryDecisionAuditSink",
    "IntegrationAuditProviderMetadata",
    "PlatformDecisionLifecycleReference",
    "PluginAdmissionDecision",
    "REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE",
    "RecordingDecisionIntegrationAuditProvider",
    "ReferenceDecisionLifecycleReference",
    "ReferenceEnterpriseLifecycleState",
    "SingleLifecycleAdapterProvider",
    "filter_admitted_adapter_providers",
    "resolve_integration_plugin_descriptor",
]
