# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_jenkins_ci_cd as _legacy_create_jenkins_ci_cd

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ci_cd.jenkins.integration import (
    JENKINS_CI_CD_PROVIDER_ID,
    JenkinsCiCdIntegration,
    JenkinsCiCdIntegrationConfig,
    JenkinsCiCdClient,
)

__all__ = [
    "create_jenkins_ci_cd",
    "create_jenkins_ci_cd_integration",
]


def create_jenkins_ci_cd_integration(
    *,
    client: JenkinsCiCdClient | None = None,
    enabled: bool = False,
) -> JenkinsCiCdIntegration:
    """
    Build a contract-based Jenkins ci cd integration.

    The legacy facade (create_jenkins_ci_cd) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Jenkins ci cd integration requires an injected client when enabled=True",
        )
    if client is not None:
        return JenkinsCiCdIntegration.from_client(client, enabled=enabled)
    return JenkinsCiCdIntegration.for_provider(
        provider_id=JENKINS_CI_CD_PROVIDER_ID,
        display_name="Jenkins",
        config=JenkinsCiCdIntegrationConfig(enabled=enabled),
    )


def create_jenkins_ci_cd(**kwargs: object) -> JenkinsCiCdIntegration:
    """Compatibility shim — constructs JenkinsCiCdIntegration from legacy runtime."""
    runtime = _legacy_create_jenkins_ci_cd(**kwargs)
    if isinstance(runtime, JenkinsCiCdIntegration):
        return runtime
    return JenkinsCiCdIntegration.from_client(runtime)
