# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "CIRCLECI_CI_CD_PROVIDER_ID",
    "CircleciCiCdIntegration",
    "CircleciCiCdIntegrationConfig",
    "CircleciCiCdClient",
    "create_circleci_ci_cd",
    "create_circleci_ci_cd_integration",
    "register_circleci_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_circleci_ci_cd",
        "create_circleci_ci_cd_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "CIRCLECI_CI_CD_PROVIDER_ID",
        "CircleciCiCdIntegration",
        "CircleciCiCdIntegrationConfig",
        "CircleciCiCdClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "CIRCLECI_CI_CD_PROVIDER_ID",
        "CircleciCiCdIntegration",
        "CircleciCiCdIntegrationConfig",
        "CircleciCiCdClient",
    }
)

def __getattr__(name: str):
    if name == "register_circleci_integration":
        from intergrax.integrations.providers.ci_cd.circleci.register import register_circleci_integration

        return register_circleci_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.ci_cd.circleci import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ci_cd.circleci import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.ci_cd.circleci import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
