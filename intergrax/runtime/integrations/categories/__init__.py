# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider category integration contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from intergrax.runtime.integrations.categories.ai import (
    DOCUMENT_PARSER_INTEGRATION_CONTRACT_SCHEMA,
    LLM_GUARDRAIL_INTEGRATION_CONTRACT_SCHEMA,
    ML_INFERENCE_HOST_INTEGRATION_CONTRACT_SCHEMA,
    MODEL_SERVING_RUNTIME_INTEGRATION_CONTRACT_SCHEMA,
    SPEECH_PROVIDER_INTEGRATION_CONTRACT_SCHEMA,
    VISION_SERVING_INTEGRATION_CONTRACT_SCHEMA,
    DocumentParserIntegrationContract,
    LlmGuardrailIntegrationContract,
    MlInferenceHostIntegrationContract,
    ModelServingRuntimeIntegrationContract,
    SpeechProviderIntegrationContract,
    VisionServingIntegrationContract,
)
from intergrax.runtime.integrations.categories.automation import (
    BILLING_METER_INTEGRATION_CONTRACT_SCHEMA,
    BROWSER_AUTOMATION_INTEGRATION_CONTRACT_SCHEMA,
    CRM_INTEGRATION_CONTRACT_SCHEMA,
    BillingMeterIntegrationContract,
    BrowserAutomationIntegrationContract,
    CrmIntegrationContract,
)
from intergrax.runtime.integrations.categories.collaboration import (
    COLLABORATION_SUITE_INTEGRATION_CONTRACT_SCHEMA,
    ISSUE_TRACKER_INTEGRATION_CONTRACT_SCHEMA,
    WIKI_KNOWLEDGE_INTEGRATION_CONTRACT_SCHEMA,
    CollaborationSuiteIntegrationContract,
    IssueTrackerIntegrationContract,
    WikiKnowledgeIntegrationContract,
)
from intergrax.runtime.integrations.categories.data import (
    GRAPH_STORE_INTEGRATION_CONTRACT_SCHEMA,
    KEY_VALUE_CACHE_INTEGRATION_CONTRACT_SCHEMA,
    RELATIONAL_STORE_INTEGRATION_CONTRACT_SCHEMA,
    GraphStoreIntegrationContract,
    KeyValueCacheIntegrationContract,
    RelationalStoreIntegrationContract,
)
from intergrax.runtime.integrations.categories.devops import (
    CI_CD_INTEGRATION_CONTRACT_SCHEMA,
    CLOUD_PLATFORM_INTEGRATION_CONTRACT_SCHEMA,
    SANDBOX_HOST_INTEGRATION_CONTRACT_SCHEMA,
    SECURITY_SCANNER_INTEGRATION_CONTRACT_SCHEMA,
    WORKFLOW_ORCHESTRATOR_INTEGRATION_CONTRACT_SCHEMA,
    CiCdIntegrationContract,
    CloudPlatformIntegrationContract,
    SandboxHostIntegrationContract,
    SecurityScannerIntegrationContract,
    WorkflowOrchestratorIntegrationContract,
)
from intergrax.runtime.integrations.categories.managed_retrieval import (
    MANAGED_RETRIEVAL_INTEGRATION_CONTRACT_SCHEMA,
    ManagedRetrievalIntegrationContract,
)
from intergrax.runtime.integrations.categories.messaging import (
    CONVERSATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA,
    MESSAGE_BUS_INTEGRATION_CONTRACT_SCHEMA,
    NOTIFICATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA,
    ConversationChannelIntegrationContract,
    MessageBusIntegrationContract,
    NotificationChannelIntegrationContract,
)
from intergrax.runtime.integrations.categories.search import (
    RERANK_PROVIDER_INTEGRATION_CONTRACT_SCHEMA,
    SEARCH_PROVIDER_INTEGRATION_CONTRACT_SCHEMA,
    RerankProviderIntegrationContract,
    SearchProviderIntegrationContract,
)
from intergrax.runtime.integrations.categories.security import (
    FEATURE_FLAG_INTEGRATION_CONTRACT_SCHEMA,
    IDENTITY_PROVIDER_INTEGRATION_CONTRACT_SCHEMA,
    SECRETS_STORE_INTEGRATION_CONTRACT_SCHEMA,
    FeatureFlagIntegrationContract,
    IdentityProviderIntegrationContract,
    SecretsStoreIntegrationContract,
)
from intergrax.runtime.integrations.categories.storage import (
    OBJECT_STORAGE_INTEGRATION_CONTRACT_SCHEMA,
    VECTOR_STORE_INTEGRATION_CONTRACT_SCHEMA,
    ObjectStorageIntegrationContract,
    VectorStoreIntegrationContract,
)
from intergrax.runtime.integrations.contracts import PlatformIntegrationContract, PlatformIntegrationKind
from intergrax.runtime.integrations.document_store import DocumentStoreVendorIntegrationContract
from intergrax.runtime.integrations.observability import ObservabilityVendorIntegrationContract

# Provider folder category for observability backends (layout.py).
OBSERVABILITY_BACKEND_CATEGORY = PlatformIntegrationKind.OBSERVABILITY_BACKEND.value

# integration_kind used by ObservabilityVendorIntegrationContract (INTEGRATIONS-1B).
OBSERVABILITY_VENDOR_INTEGRATION_KIND = PlatformIntegrationKind.OBSERVABILITY_VENDOR.value

# Maps layout.py SLUG_CATEGORY folder names to category contract classes.
# observability_backend aliases to ObservabilityVendorIntegrationContract — no duplicate contract.
# document_store aliases to DocumentStoreVendorIntegrationContract — no duplicate contract.
PROVIDER_CATEGORY_CONTRACT_REGISTRY: dict[str, type[PlatformIntegrationContract]] = {
    "relational_store": RelationalStoreIntegrationContract,
    "document_store": DocumentStoreVendorIntegrationContract,
    "key_value_cache": KeyValueCacheIntegrationContract,
    "message_bus": MessageBusIntegrationContract,
    "object_storage": ObjectStorageIntegrationContract,
    "vector_store": VectorStoreIntegrationContract,
    "search_provider": SearchProviderIntegrationContract,
    "notification_channel": NotificationChannelIntegrationContract,
    "conversation_channel": ConversationChannelIntegrationContract,
    "collaboration_suite": CollaborationSuiteIntegrationContract,
    "issue_tracker": IssueTrackerIntegrationContract,
    "wiki_knowledge": WikiKnowledgeIntegrationContract,
    OBSERVABILITY_BACKEND_CATEGORY: ObservabilityVendorIntegrationContract,
    "browser_automation": BrowserAutomationIntegrationContract,
    "cloud_platform": CloudPlatformIntegrationContract,
    "secrets_store": SecretsStoreIntegrationContract,
    "graph_store": GraphStoreIntegrationContract,
    "document_parser": DocumentParserIntegrationContract,
    "rerank_provider": RerankProviderIntegrationContract,
    "feature_flag": FeatureFlagIntegrationContract,
    "ci_cd": CiCdIntegrationContract,
    "security_scanner": SecurityScannerIntegrationContract,
    "sandbox_host": SandboxHostIntegrationContract,
    "identity_provider": IdentityProviderIntegrationContract,
    "speech_provider": SpeechProviderIntegrationContract,
    "workflow_orchestrator": WorkflowOrchestratorIntegrationContract,
    "billing_meter": BillingMeterIntegrationContract,
    "crm": CrmIntegrationContract,
    "vision_serving": VisionServingIntegrationContract,
    "ml_inference_host": MlInferenceHostIntegrationContract,
    "model_serving_runtime": ModelServingRuntimeIntegrationContract,
    "llm_guardrail": LlmGuardrailIntegrationContract,
    "managed_retrieval": ManagedRetrievalIntegrationContract,
}

__all__ = [
    "BILLING_METER_INTEGRATION_CONTRACT_SCHEMA",
    "BROWSER_AUTOMATION_INTEGRATION_CONTRACT_SCHEMA",
    "CI_CD_INTEGRATION_CONTRACT_SCHEMA",
    "CLOUD_PLATFORM_INTEGRATION_CONTRACT_SCHEMA",
    "COLLABORATION_SUITE_INTEGRATION_CONTRACT_SCHEMA",
    "CONVERSATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA",
    "CRM_INTEGRATION_CONTRACT_SCHEMA",
    "DOCUMENT_PARSER_INTEGRATION_CONTRACT_SCHEMA",
    "FEATURE_FLAG_INTEGRATION_CONTRACT_SCHEMA",
    "GRAPH_STORE_INTEGRATION_CONTRACT_SCHEMA",
    "IDENTITY_PROVIDER_INTEGRATION_CONTRACT_SCHEMA",
    "ISSUE_TRACKER_INTEGRATION_CONTRACT_SCHEMA",
    "KEY_VALUE_CACHE_INTEGRATION_CONTRACT_SCHEMA",
    "LLM_GUARDRAIL_INTEGRATION_CONTRACT_SCHEMA",
    "MANAGED_RETRIEVAL_INTEGRATION_CONTRACT_SCHEMA",
    "MESSAGE_BUS_INTEGRATION_CONTRACT_SCHEMA",
    "ML_INFERENCE_HOST_INTEGRATION_CONTRACT_SCHEMA",
    "MODEL_SERVING_RUNTIME_INTEGRATION_CONTRACT_SCHEMA",
    "NOTIFICATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA",
    "OBJECT_STORAGE_INTEGRATION_CONTRACT_SCHEMA",
    "OBSERVABILITY_BACKEND_CATEGORY",
    "OBSERVABILITY_VENDOR_INTEGRATION_KIND",
    "PROVIDER_CATEGORY_CONTRACT_REGISTRY",
    "RELATIONAL_STORE_INTEGRATION_CONTRACT_SCHEMA",
    "RERANK_PROVIDER_INTEGRATION_CONTRACT_SCHEMA",
    "SANDBOX_HOST_INTEGRATION_CONTRACT_SCHEMA",
    "SEARCH_PROVIDER_INTEGRATION_CONTRACT_SCHEMA",
    "SECRETS_STORE_INTEGRATION_CONTRACT_SCHEMA",
    "SECURITY_SCANNER_INTEGRATION_CONTRACT_SCHEMA",
    "SPEECH_PROVIDER_INTEGRATION_CONTRACT_SCHEMA",
    "VECTOR_STORE_INTEGRATION_CONTRACT_SCHEMA",
    "VISION_SERVING_INTEGRATION_CONTRACT_SCHEMA",
    "WIKI_KNOWLEDGE_INTEGRATION_CONTRACT_SCHEMA",
    "WORKFLOW_ORCHESTRATOR_INTEGRATION_CONTRACT_SCHEMA",
    "BillingMeterIntegrationContract",
    "BrowserAutomationIntegrationContract",
    "CiCdIntegrationContract",
    "CloudPlatformIntegrationContract",
    "CollaborationSuiteIntegrationContract",
    "ConversationChannelIntegrationContract",
    "CrmIntegrationContract",
    "DocumentParserIntegrationContract",
    "FeatureFlagIntegrationContract",
    "GraphStoreIntegrationContract",
    "IdentityProviderIntegrationContract",
    "IssueTrackerIntegrationContract",
    "KeyValueCacheIntegrationContract",
    "LlmGuardrailIntegrationContract",
    "ManagedRetrievalIntegrationContract",
    "MessageBusIntegrationContract",
    "MlInferenceHostIntegrationContract",
    "ModelServingRuntimeIntegrationContract",
    "NotificationChannelIntegrationContract",
    "ObjectStorageIntegrationContract",
    "RelationalStoreIntegrationContract",
    "RerankProviderIntegrationContract",
    "SandboxHostIntegrationContract",
    "SearchProviderIntegrationContract",
    "SecretsStoreIntegrationContract",
    "SecurityScannerIntegrationContract",
    "SpeechProviderIntegrationContract",
    "VectorStoreIntegrationContract",
    "VisionServingIntegrationContract",
    "WikiKnowledgeIntegrationContract",
    "WorkflowOrchestratorIntegrationContract",
]
