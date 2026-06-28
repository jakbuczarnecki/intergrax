# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_argocd_ci_cd

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ci_cd.argocd.integration import (
    ARGOCD_CI_CD_PROVIDER_ID,
    ArgocdCiCdIntegration,
    ArgocdCiCdIntegrationConfig,
    ArgocdCiCdClient,
)

__all__ = [
    "create_argocd_ci_cd",
    "create_argocd_ci_cd_integration",
]


def create_argocd_ci_cd_integration(
    *,
    client: ArgocdCiCdClient | None = None,
    enabled: bool = False,
) -> ArgocdCiCdIntegration:
    """
    Build a contract-based Argocd ci cd integration.

    The legacy facade (create_argocd_ci_cd) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Argocd ci cd integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ArgocdCiCdIntegration.from_client(client, enabled=enabled)
    return ArgocdCiCdIntegration.for_provider(
        provider_id=ARGOCD_CI_CD_PROVIDER_ID,
        display_name="Argocd",
        config=ArgocdCiCdIntegrationConfig(enabled=enabled),
    )
