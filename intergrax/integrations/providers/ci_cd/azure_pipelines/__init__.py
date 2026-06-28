# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "AZURE_PIPELINES_CI_CD_PROVIDER_ID",
    "AzurePipelinesCiCdIntegration",
    "AzurePipelinesCiCdIntegrationConfig",
    "AzurePipelinesCiCdClient",
    "create_azure_pipelines_ci_cd",
    "create_azure_pipelines_ci_cd_integration",
    "register_azure_pipelines_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_azure_pipelines_ci_cd",
        "create_azure_pipelines_ci_cd_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_PIPELINES_CI_CD_PROVIDER_ID",
        "AzurePipelinesCiCdIntegration",
        "AzurePipelinesCiCdIntegrationConfig",
        "AzurePipelinesCiCdClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_PIPELINES_CI_CD_PROVIDER_ID",
        "AzurePipelinesCiCdIntegration",
        "AzurePipelinesCiCdIntegrationConfig",
        "AzurePipelinesCiCdClient",
    }
)

def __getattr__(name: str):
    if name == "register_azure_pipelines_integration":
        from intergrax.integrations.providers.ci_cd.azure_pipelines.register import register_azure_pipelines_integration

        return register_azure_pipelines_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.ci_cd.azure_pipelines import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ci_cd.azure_pipelines import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ci_cd.azure_pipelines import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
