# © Artur Czarnecki. All rights reserved.

"""CONFIG-X configurable concern inventory SSOT."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from intergrax.integrations.contracts.base import IntegrationCategory

from tests.qualification.config_x._config_x_types import ConfigClassification

_EVIDENCE = "test_config_x_qualification_gates.py::test_cx_q01_concern_inventory_closed_world"


@dataclass(frozen=True, slots=True)
class ConfigConcernRecord:
    concern_id: str
    domain: str
    classification: ConfigClassification
    configuration_contract: str
    composition_owner: str
    effective_resolution_owner: str
    provider_surface: str


def _integration_row(category: IntegrationCategory) -> ConfigConcernRecord:
    cat = category.value
    return ConfigConcernRecord(
        concern_id=f"integration.{cat}",
        domain="INTEGRATIONS",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="IntegrationProfile / IntegrationBinding",
        composition_owner="intergrax/integrations/registry/factory.py (resolve_slug / resolve_from_profile)",
        effective_resolution_owner="Integration Catalog + IntegrationProfile slug_for_category",
        provider_surface=f"intergrax/integrations/providers/{cat}/*",
    )


_PLATFORM_CONCERNS: Final[tuple[ConfigConcernRecord, ...]] = (
    ConfigConcernRecord(
        concern_id="llm.chat_completion",
        domain="LLM_AI",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="LLMProfile / intergrax/llm_adapters/contracts/llm_profile.py",
        composition_owner="intergrax/llm_adapters/registry/profile.py",
        effective_resolution_owner="LLMAdapterRegistry.create(explicit provider)",
        provider_surface="intergrax/llm_adapters/providers/*",
    ),
    ConfigConcernRecord(
        concern_id="llm.embedding",
        domain="LLM_AI",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="IntegrationProfile.embedding_provider + RAG embedding profile",
        composition_owner="intergrax/rag/embedding/registry/profile.py",
        effective_resolution_owner="resolve_from_profile(EMBEDDING_PROVIDER)",
        provider_surface="intergrax/rag/embedding/providers/*",
    ),
    ConfigConcernRecord(
        concern_id="llm.rerank",
        domain="LLM_AI",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="IntegrationProfile.rerank_provider",
        composition_owner="intergrax/rag/rerankers/registry/reranker_registry.py",
        effective_resolution_owner="resolve_from_profile(RERANK_PROVIDER)",
        provider_surface="intergrax/integrations/providers/rerank_provider/*",
    ),
    ConfigConcernRecord(
        concern_id="llm.fallback_policy",
        domain="LLM_AI",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="LLMProfile.fallback_profiles (explicit typed policy)",
        composition_owner="intergrax/llm_adapters/contracts/llm_profile.py",
        effective_resolution_owner="Caller policy — no runtime silent vendor fallback in registry",
        provider_surface="intergrax/llm_adapters/contracts/llm_profile.py",
    ),
    ConfigConcernRecord(
        concern_id="llm.model_routing_qualifier",
        domain="LLM_AI",
        classification=ConfigClassification.G_COMPATIBILITY_ADAPTER,
        configuration_contract="agents/model_routing_qualifier/routing_profile.py",
        composition_owner="agents/model_routing_qualifier/model_routing.py",
        effective_resolution_owner="Explicit routing profile — qualification agent scope",
        provider_surface="agents/model_routing_qualifier/*",
    ),
    ConfigConcernRecord(
        concern_id="persistence.relational_sqlite_lab",
        domain="PERSISTENCE",
        classification=ConfigClassification.D_SANCTIONED_EXPLICIT_DEFAULT,
        configuration_contract="IntegrationProfile.relational_store + lab presets",
        composition_owner="intergrax/integrations/registry/presets.py",
        effective_resolution_owner="resolve_from_profile(RELATIONAL_STORE)",
        provider_surface="intergrax/integrations/providers/relational_store/sqlite/*",
    ),
    ConfigConcernRecord(
        concern_id="int_config.configure_existing",
        domain="INTEGRATIONS",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="ExistingCapabilityConfigurationRealizationRequest",
        composition_owner="intergrax/integrations/existing_capability_configuration_facade.py",
        effective_resolution_owner="ExistingCapabilityConfigurationRealizationService + strategy SPI",
        provider_surface="intergrax/integrations/providers/relational_store/sqlite/configuration_realization.py",
    ),
    ConfigConcernRecord(
        concern_id="execution.integration_bound_resolution",
        domain="PLATFORM_ACTIVATION",
        classification=ConfigClassification.B_CANONICAL_COMPOSITION_OWNER,
        configuration_contract="ConfiguredCapabilityBinding / ExecutionIntegrationConfigurationAdoption",
        composition_owner="intergrax/integrations/execution_bound_integration_resolution.py",
        effective_resolution_owner="ExecutionBoundIntegrationResolution (single class — TRACE-X-P5-R2)",
        provider_surface="intergrax/integrations/execution_bound_integration_resolution.py",
    ),
    ConfigConcernRecord(
        concern_id="observability.otlp_transport",
        domain="OBSERVABILITY",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="OtlpTransportPort / ObservabilityExportPayload",
        composition_owner="intergrax/applications/_shared/runtime_event_delivery_wiring.py",
        effective_resolution_owner="Runtime event delivery wiring — explicit OTLP transport injection",
        provider_surface="intergrax/runtime/observability/exporters/*",
    ),
    ConfigConcernRecord(
        concern_id="plugin.integration_catalog",
        domain="PLATFORM_ACTIVATION",
        classification=ConfigClassification.B_CANONICAL_COMPOSITION_OWNER,
        configuration_contract="IntegrationEntry / plugin_register contract_specs",
        composition_owner="intergrax/integrations/registry/catalog.py",
        effective_resolution_owner="get_entry(slug) — registration ≠ activation",
        provider_surface="intergrax/integrations/contracts/catalog_factory.py",
    ),
    ConfigConcernRecord(
        concern_id="plugin.llm_adapter_registry",
        domain="PLATFORM_ACTIVATION",
        classification=ConfigClassification.B_CANONICAL_COMPOSITION_OWNER,
        configuration_contract="LLMAdapterRegistrationSpec",
        composition_owner="intergrax/llm_adapters/llm_provider_registry.py",
        effective_resolution_owner="LLMAdapterRegistry.create(explicit provider id)",
        provider_surface="intergrax/llm_adapters/providers/registrations/*",
    ),
    ConfigConcernRecord(
        concern_id="tool.marketplace_activation",
        domain="PLATFORM_ACTIVATION",
        classification=ConfigClassification.B_CANONICAL_COMPOSITION_OWNER,
        configuration_contract="QualifiedMarketplaceToolActivationResolver",
        composition_owner="intergrax/tools/qualified_marketplace_tool_activation_resolver.py",
        effective_resolution_owner="Governed qualification + explicit binding",
        provider_surface="intergrax/tools/qualified_marketplace_tool_activation_resolver.py",
    ),
    ConfigConcernRecord(
        concern_id="agent.effective_profile",
        domain="PLATFORM_ACTIVATION",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="EffectiveProfileExecutionPinningStore (TRACE-X-P5-R1)",
        composition_owner="intergrax/applications/_shared/profile_resolution/wiring.py",
        effective_resolution_owner="Profile revision admission + pinning store",
        provider_surface="intergrax/applications/_shared/profile_resolution/*",
    ),
    ConfigConcernRecord(
        concern_id="speech.provider",
        domain="INTEGRATIONS",
        classification=ConfigClassification.G_COMPATIBILITY_ADAPTER,
        configuration_contract="IntegrationProfile.speech_provider",
        composition_owner="intergrax/speech_adapters/registry/resolver.py",
        effective_resolution_owner="resolve_from_profile(SPEECH_PROVIDER)",
        provider_surface="intergrax/speech_adapters/*",
    ),
    ConfigConcernRecord(
        concern_id="env.integration_slug",
        domain="PLATFORM_ACTIVATION",
        classification=ConfigClassification.F_ENVIRONMENT_DEPLOYMENT_CONSTANT,
        configuration_contract="INTERGRAX_INTEGRATION_<CATEGORY> env + read_integration_slug_from_env",
        composition_owner="intergrax/integrations/_shared/config.py",
        effective_resolution_owner="build_profile_from_env → merge into IntegrationProfile",
        provider_surface="intergrax/integrations/registry/factory.py",
    ),
    ConfigConcernRecord(
        concern_id="application.composition_root",
        domain="PLATFORM_ACTIVATION",
        classification=ConfigClassification.G_COMPATIBILITY_ADAPTER,
        configuration_contract="Application EnvironmentProfile / integration_wiring",
        composition_owner="applications/*/host/integration_wiring.py",
        effective_resolution_owner="Sanctioned Tier-3 composition — must not become second catalog",
        provider_surface="applications/*/host/*_wiring.py",
    ),
    ConfigConcernRecord(
        concern_id="lab.integration_preset",
        domain="PLATFORM_ACTIVATION",
        classification=ConfigClassification.H_REFERENCE_LAB_TEST_ONLY,
        configuration_contract="IntegrationProfile.lab_harness_preset",
        composition_owner="intergrax/integrations/contracts/integration_profile.py",
        effective_resolution_owner="Explicit lab/product preset constructors only",
        provider_surface="applications/lab_application/host/integration_wiring.py",
    ),
    ConfigConcernRecord(
        concern_id="tokenizer.selection",
        domain="LLM_AI",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract="Tokenizer / TokenizerRegistry.default_tokenizer_id",
        composition_owner="intergrax/tokenizers/registry/tokenizer_registry.py",
        effective_resolution_owner=(
            "Explicit tokenizer id or configured default_tokenizer_id "
            "(CONFIG-X-BLK-TOK-01 remediated — not active)"
        ),
        provider_surface="intergrax/tokenizers/registry/tokenizer_registry.py",
    ),
    ConfigConcernRecord(
        concern_id="tool.observability_backend",
        domain="OBSERVABILITY",
        classification=ConfigClassification.A_CANONICAL_CONFIGURATION_CONTRACT,
        configuration_contract=(
            "IntegrationProfile.observability_roles (ObservabilityRoleBindings)"
        ),
        composition_owner=(
            "intergrax/tools/registry/wiring.py "
            "(ToolWiringContext.from_integration_profile → observability_role_backends)"
        ),
        effective_resolution_owner=(
            "resolve_observability_backend — role → observability_role_backends; "
            "default → observability_backend only (CONFIG-X-BLK-OBS-TOOL-01 remediated)"
        ),
        provider_surface="intergrax/tools/providers/observability/resolve.py",
    ),
)


@lru_cache(maxsize=1)
def config_x_concern_inventory() -> tuple[ConfigConcernRecord, ...]:
    integration_rows = tuple(_integration_row(cat) for cat in IntegrationCategory)
    return integration_rows + _PLATFORM_CONCERNS


CONFIG_X_CONCERN_INVENTORY: Final[tuple[ConfigConcernRecord, ...]] = config_x_concern_inventory()
