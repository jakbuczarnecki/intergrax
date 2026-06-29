# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p6.factories import create_gitlab_ci_ci_cd as _legacy_create_gitlab_ci_ci_cd

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ci_cd.gitlab_ci.integration import (
    GITLAB_CI_CI_CD_PROVIDER_ID,
    GitlabCiCiCdIntegration,
    GitlabCiCiCdIntegrationConfig,
    GitlabCiCiCdClient,
)

__all__ = [
    "create_gitlab_ci_ci_cd",
    "create_gitlab_ci_ci_cd_integration",
]


def create_gitlab_ci_ci_cd_integration(
    *,
    client: GitlabCiCiCdClient | None = None,
    enabled: bool = False,
) -> GitlabCiCiCdIntegration:
    """
    Build a contract-based Gitlab Ci ci cd integration.

    The legacy facade (create_gitlab_ci_ci_cd) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Gitlab Ci ci cd integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GitlabCiCiCdIntegration.from_client(client, enabled=enabled)
    return GitlabCiCiCdIntegration.for_provider(
        provider_id=GITLAB_CI_CI_CD_PROVIDER_ID,
        display_name="Gitlab Ci",
        config=GitlabCiCiCdIntegrationConfig(enabled=enabled),
    )


def create_gitlab_ci_ci_cd(**kwargs: object) -> GitlabCiCiCdIntegration:
    """Compatibility shim — constructs GitlabCiCiCdIntegration from legacy runtime."""
    runtime = _legacy_create_gitlab_ci_ci_cd(**kwargs)
    if isinstance(runtime, GitlabCiCiCdIntegration):
        return runtime
    return GitlabCiCiCdIntegration.from_runtime(runtime)
