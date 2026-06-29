# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p6.factories import create_azure_pipelines_ci_cd as _legacy_create_azure_pipelines_ci_cd

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ci_cd.azure_pipelines.integration import (
    AZURE_PIPELINES_CI_CD_PROVIDER_ID,
    AzurePipelinesCiCdIntegration,
    AzurePipelinesCiCdIntegrationConfig,
    AzurePipelinesCiCdClient,
)

__all__ = [
    "create_azure_pipelines_ci_cd",
    "create_azure_pipelines_ci_cd_integration",
]


def create_azure_pipelines_ci_cd_integration(
    *,
    client: AzurePipelinesCiCdClient | None = None,
    enabled: bool = False,
) -> AzurePipelinesCiCdIntegration:
    """
    Build a contract-based Azure Pipelines ci cd integration.

    The legacy facade (create_azure_pipelines_ci_cd) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Azure Pipelines ci cd integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AzurePipelinesCiCdIntegration.from_client(client, enabled=enabled)
    return AzurePipelinesCiCdIntegration.for_provider(
        provider_id=AZURE_PIPELINES_CI_CD_PROVIDER_ID,
        display_name="Azure Pipelines",
        config=AzurePipelinesCiCdIntegrationConfig(enabled=enabled),
    )


def create_azure_pipelines_ci_cd(**kwargs: object) -> AzurePipelinesCiCdIntegration:
    """Compatibility shim — constructs AzurePipelinesCiCdIntegration from legacy runtime."""
    runtime = _legacy_create_azure_pipelines_ci_cd(**kwargs)
    if isinstance(runtime, AzurePipelinesCiCdIntegration):
        return runtime
    return AzurePipelinesCiCdIntegration.from_runtime(runtime)
