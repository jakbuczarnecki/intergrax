# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_github_actions_ci_cd

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ci_cd.github_actions.integration import (
    GITHUB_ACTIONS_CI_CD_PROVIDER_ID,
    GithubActionsCiCdIntegration,
    GithubActionsCiCdIntegrationConfig,
    GithubActionsCiCdClient,
)

__all__ = [
    "create_github_actions_ci_cd",
    "create_github_actions_ci_cd_integration",
]


def create_github_actions_ci_cd_integration(
    *,
    client: GithubActionsCiCdClient | None = None,
    enabled: bool = False,
) -> GithubActionsCiCdIntegration:
    """
    Build a contract-based Github Actions ci cd integration.

    The legacy facade (create_github_actions_ci_cd) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Github Actions ci cd integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GithubActionsCiCdIntegration.from_client(client, enabled=enabled)
    return GithubActionsCiCdIntegration.for_provider(
        provider_id=GITHUB_ACTIONS_CI_CD_PROVIDER_ID,
        display_name="Github Actions",
        config=GithubActionsCiCdIntegrationConfig(enabled=enabled),
    )
