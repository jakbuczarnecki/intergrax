# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GITLAB_CI_CI_CD_PROVIDER_ID",
    "GitlabCiCiCdIntegration",
    "GitlabCiCiCdIntegrationConfig",
    "GitlabCiCiCdClient",
    "create_gitlab_ci_ci_cd",
    "create_gitlab_ci_ci_cd_integration",
    "register_gitlab_ci_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_gitlab_ci_ci_cd",
        "create_gitlab_ci_ci_cd_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "GITLAB_CI_CI_CD_PROVIDER_ID",
        "GitlabCiCiCdIntegration",
        "GitlabCiCiCdIntegrationConfig",
        "GitlabCiCiCdClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "GITLAB_CI_CI_CD_PROVIDER_ID",
        "GitlabCiCiCdIntegration",
        "GitlabCiCiCdIntegrationConfig",
        "GitlabCiCiCdClient",
    }
)

def __getattr__(name: str):
    if name == "register_gitlab_ci_integration":
        from intergrax.integrations.providers.ci_cd.gitlab_ci.register import register_gitlab_ci_integration

        return register_gitlab_ci_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.ci_cd.gitlab_ci import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ci_cd.gitlab_ci import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ci_cd.gitlab_ci import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
