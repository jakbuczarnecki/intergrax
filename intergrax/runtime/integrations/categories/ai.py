# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AI, ML, and document parsing provider category contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_READ_HEALTH,
    _READ_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

DOCUMENT_PARSER_INTEGRATION_CONTRACT_SCHEMA = "document_parser_integration_contract.v1"
VISION_SERVING_INTEGRATION_CONTRACT_SCHEMA = "vision_serving_integration_contract.v1"
ML_INFERENCE_HOST_INTEGRATION_CONTRACT_SCHEMA = "ml_inference_host_integration_contract.v1"
LLM_GUARDRAIL_INTEGRATION_CONTRACT_SCHEMA = "llm_guardrail_integration_contract.v1"
SPEECH_PROVIDER_INTEGRATION_CONTRACT_SCHEMA = "speech_provider_integration_contract.v1"


class DocumentParserIntegrationContract(PlatformIntegrationContract):
    """Category contract for document_parser providers (docling, pymupdf, …)."""

    schema_id: Literal["document_parser_integration_contract.v1"] = (
        DOCUMENT_PARSER_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.DOCUMENT_PARSER.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(default_factory=lambda: _READ_HEALTH)
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> DocumentParserIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.DOCUMENT_PARSER.value,
            default_capabilities=_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class VisionServingIntegrationContract(PlatformIntegrationContract):
    """Category contract for vision_serving providers (triton, …)."""

    schema_id: Literal["vision_serving_integration_contract.v1"] = (
        VISION_SERVING_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.VISION_SERVING.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> VisionServingIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.VISION_SERVING.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class MlInferenceHostIntegrationContract(PlatformIntegrationContract):
    """Category contract for ml_inference_host providers (replicate, …)."""

    schema_id: Literal["ml_inference_host_integration_contract.v1"] = (
        ML_INFERENCE_HOST_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.ML_INFERENCE_HOST.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> MlInferenceHostIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.ML_INFERENCE_HOST.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class LlmGuardrailIntegrationContract(PlatformIntegrationContract):
    """Category contract for llm_guardrail providers (llm_guard, presidio, …)."""

    schema_id: Literal["llm_guardrail_integration_contract.v1"] = LLM_GUARDRAIL_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.LLM_GUARDRAIL.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> LlmGuardrailIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.LLM_GUARDRAIL.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class SpeechProviderIntegrationContract(PlatformIntegrationContract):
    """Category contract for speech_provider slugs (elevenlabs, deepgram, …)."""

    schema_id: Literal["speech_provider_integration_contract.v1"] = (
        SPEECH_PROVIDER_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.SPEECH_PROVIDER.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> SpeechProviderIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.SPEECH_PROVIDER.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )
