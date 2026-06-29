# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_doppler_secrets_store as _legacy_create_doppler_secrets_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.secrets_store.doppler.integration import (
    DOPPLER_SECRETS_STORE_PROVIDER_ID,
    DopplerSecretsStoreIntegration,
    DopplerSecretsStoreIntegrationConfig,
    DopplerSecretsStoreClient,
)

__all__ = [
    "create_doppler_secrets_store",
    "create_doppler_secrets_store_integration",
]


def create_doppler_secrets_store_integration(
    *,
    client: DopplerSecretsStoreClient | None = None,
    enabled: bool = False,
) -> DopplerSecretsStoreIntegration:
    """
    Build a contract-based Doppler secrets store integration.

    The legacy facade (create_doppler_secrets_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Doppler secrets store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DopplerSecretsStoreIntegration.from_client(client, enabled=enabled)
    return DopplerSecretsStoreIntegration.for_provider(
        provider_id=DOPPLER_SECRETS_STORE_PROVIDER_ID,
        display_name="Doppler",
        config=DopplerSecretsStoreIntegrationConfig(enabled=enabled),
    )


def create_doppler_secrets_store(**kwargs: object) -> DopplerSecretsStoreIntegration:
    """Compatibility shim — constructs DopplerSecretsStoreIntegration from legacy runtime."""
    runtime = _legacy_create_doppler_secrets_store(**kwargs)
    if isinstance(runtime, DopplerSecretsStoreIntegration):
        return runtime
    return DopplerSecretsStoreIntegration.from_runtime(runtime)
