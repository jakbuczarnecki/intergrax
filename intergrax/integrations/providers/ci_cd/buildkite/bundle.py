# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_buildkite_ci_cd as _legacy_create_buildkite_ci_cd

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ci_cd.buildkite.integration import (
    BUILDKITE_CI_CD_PROVIDER_ID,
    BuildkiteCiCdIntegration,
    BuildkiteCiCdIntegrationConfig,
    BuildkiteCiCdClient,
)

__all__ = [
    "create_buildkite_ci_cd",
    "create_buildkite_ci_cd_integration",
]


def create_buildkite_ci_cd_integration(
    *,
    client: BuildkiteCiCdClient | None = None,
    enabled: bool = False,
) -> BuildkiteCiCdIntegration:
    """
    Build a contract-based Buildkite ci cd integration.

    The legacy facade (create_buildkite_ci_cd) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Buildkite ci cd integration requires an injected client when enabled=True",
        )
    if client is not None:
        return BuildkiteCiCdIntegration.from_client(client, enabled=enabled)
    return BuildkiteCiCdIntegration.for_provider(
        provider_id=BUILDKITE_CI_CD_PROVIDER_ID,
        display_name="Buildkite",
        config=BuildkiteCiCdIntegrationConfig(enabled=enabled),
    )


def create_buildkite_ci_cd(**kwargs: object) -> BuildkiteCiCdIntegration:
    """Compatibility shim — constructs BuildkiteCiCdIntegration from legacy runtime."""
    runtime = _legacy_create_buildkite_ci_cd(**kwargs)
    if isinstance(runtime, BuildkiteCiCdIntegration):
        return runtime
    return BuildkiteCiCdIntegration.from_runtime(runtime)
