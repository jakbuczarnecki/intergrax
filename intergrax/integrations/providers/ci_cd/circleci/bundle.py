# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p6.factories import create_circleci_ci_cd as _legacy_create_circleci_ci_cd

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ci_cd.circleci.integration import (
    CIRCLECI_CI_CD_PROVIDER_ID,
    CircleciCiCdIntegration,
    CircleciCiCdIntegrationConfig,
    CircleciCiCdClient,
)

__all__ = [
    "create_circleci_ci_cd",
    "create_circleci_ci_cd_integration",
]


def create_circleci_ci_cd_integration(
    *,
    client: CircleciCiCdClient | None = None,
    enabled: bool = False,
) -> CircleciCiCdIntegration:
    """
    Build a contract-based Circleci ci cd integration.

    The legacy facade (create_circleci_ci_cd) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Circleci ci cd integration requires an injected client when enabled=True",
        )
    if client is not None:
        return CircleciCiCdIntegration.from_client(client, enabled=enabled)
    return CircleciCiCdIntegration.for_provider(
        provider_id=CIRCLECI_CI_CD_PROVIDER_ID,
        display_name="Circleci",
        config=CircleciCiCdIntegrationConfig(enabled=enabled),
    )


def create_circleci_ci_cd(**kwargs: object) -> CircleciCiCdIntegration:
    """Compatibility shim — constructs CircleciCiCdIntegration from legacy runtime."""
    runtime = _legacy_create_circleci_ci_cd(**kwargs)
    if isinstance(runtime, CircleciCiCdIntegration):
        return runtime
    return CircleciCiCdIntegration.from_client(runtime)
