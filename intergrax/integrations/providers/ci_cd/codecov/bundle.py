# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p6.factories import create_codecov_ci_cd as _legacy_create_codecov_ci_cd

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ci_cd.codecov.integration import (
    CODECOV_CI_CD_PROVIDER_ID,
    CodecovCiCdIntegration,
    CodecovCiCdIntegrationConfig,
    CodecovCiCdClient,
)

__all__ = [
    "create_codecov_ci_cd",
    "create_codecov_ci_cd_integration",
]


def create_codecov_ci_cd_integration(
    *,
    client: CodecovCiCdClient | None = None,
    enabled: bool = False,
) -> CodecovCiCdIntegration:
    """
    Build a contract-based Codecov ci cd integration.

    The legacy facade (create_codecov_ci_cd) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Codecov ci cd integration requires an injected client when enabled=True",
        )
    if client is not None:
        return CodecovCiCdIntegration.from_client(client, enabled=enabled)
    return CodecovCiCdIntegration.for_provider(
        provider_id=CODECOV_CI_CD_PROVIDER_ID,
        display_name="Codecov",
        config=CodecovCiCdIntegrationConfig(enabled=enabled),
    )


def create_codecov_ci_cd(**kwargs: object) -> CodecovCiCdIntegration:
    """Compatibility shim — constructs CodecovCiCdIntegration from legacy runtime."""
    runtime = _legacy_create_codecov_ci_cd(**kwargs)
    if isinstance(runtime, CodecovCiCdIntegration):
        return runtime
    return CodecovCiCdIntegration.from_runtime(runtime)
