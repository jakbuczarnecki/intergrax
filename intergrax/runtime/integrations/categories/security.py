# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Security, identity, and feature-flag provider category contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_READ_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

SECRETS_STORE_INTEGRATION_CONTRACT_SCHEMA = "secrets_store_integration_contract.v1"
IDENTITY_PROVIDER_INTEGRATION_CONTRACT_SCHEMA = "identity_provider_integration_contract.v1"
FEATURE_FLAG_INTEGRATION_CONTRACT_SCHEMA = "feature_flag_integration_contract.v1"


class SecretsStoreIntegrationContract(PlatformIntegrationContract):
    """Category contract for secrets_store providers (vault, aws_secrets_manager, …)."""

    schema_id: Literal["secrets_store_integration_contract.v1"] = SECRETS_STORE_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.SECRETS_STORE.value
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
    ) -> SecretsStoreIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.SECRETS_STORE.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class IdentityProviderIntegrationContract(PlatformIntegrationContract):
    """Category contract for identity_provider slugs (auth0, okta, …)."""

    schema_id: Literal["identity_provider_integration_contract.v1"] = (
        IDENTITY_PROVIDER_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.IDENTITY_PROVIDER.value
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
    ) -> IdentityProviderIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.IDENTITY_PROVIDER.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class FeatureFlagIntegrationContract(PlatformIntegrationContract):
    """Category contract for feature_flag providers (unleash, launchdarkly, …)."""

    schema_id: Literal["feature_flag_integration_contract.v1"] = FEATURE_FLAG_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.FEATURE_FLAG.value
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
    ) -> FeatureFlagIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.FEATURE_FLAG.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )
