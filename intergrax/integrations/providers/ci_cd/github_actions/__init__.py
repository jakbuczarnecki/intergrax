# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GITHUB_ACTIONS_CI_CD_PROVIDER_ID",
    "GithubActionsCiCdIntegration",
    "GithubActionsCiCdIntegrationConfig",
    "GithubActionsCiCdClient",
    "create_github_actions_ci_cd",
    "create_github_actions_ci_cd_integration",
    "register_github_actions_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_github_actions_ci_cd",
        "create_github_actions_ci_cd_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "GITHUB_ACTIONS_CI_CD_PROVIDER_ID",
        "GithubActionsCiCdIntegration",
        "GithubActionsCiCdIntegrationConfig",
        "GithubActionsCiCdClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "GITHUB_ACTIONS_CI_CD_PROVIDER_ID",
        "GithubActionsCiCdIntegration",
        "GithubActionsCiCdIntegrationConfig",
        "GithubActionsCiCdClient",
    }
)

def __getattr__(name: str):
    if name == "register_github_actions_integration":
        from intergrax.integrations.providers.ci_cd.github_actions.register import register_github_actions_integration

        return register_github_actions_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.ci_cd.github_actions import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ci_cd.github_actions import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ci_cd.github_actions import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
