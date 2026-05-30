# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Azure credential openers — internal to the azure integration package.

Only this module may construct ``azure.identity`` credentials for the Azure cloud facade.
All composition roots use ``bundle.create_azure_*`` or ``profile.resolve(CLOUD_PLATFORM)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.providers.azure.adapter import AzureCloudPlatform
from intergrax.integrations.providers.azure.config import AzureIntegrationConfig


def _import_azure_identity() -> tuple[Any, Any]:
    try:
        from azure.identity import ClientSecretCredential, DefaultAzureCredential
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Azure integration requires azure-identity. "
            "Install with: uv sync --extra dev  (includes azure-identity)"
        ) from exc
    return ClientSecretCredential, DefaultAzureCredential


def open_azure_credential(
    config: AzureIntegrationConfig,
    *,
    credential_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    if credential_factory is not None:
        return credential_factory()
    ClientSecretCredential, DefaultAzureCredential = _import_azure_identity()
    if config.uses_service_principal:
        return ClientSecretCredential(
            tenant_id=config.tenant_id,
            client_id=config.client_id,
            client_secret=config.client_secret,
        )
    return DefaultAzureCredential()


def open_azure_cloud_platform(
    config: AzureIntegrationConfig,
    *,
    implementation: Optional[CloudPlatform] = None,
    credential: Optional[Any] = None,
    credential_factory: Optional[Callable[[], Any]] = None,
) -> CloudPlatform:
    if implementation is not None:
        return implementation
    token_credential = credential or open_azure_credential(
        config,
        credential_factory=credential_factory,
    )
    return AzureCloudPlatform(config, token_credential)
