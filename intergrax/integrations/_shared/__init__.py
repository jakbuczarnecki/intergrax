# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared integration helpers — lightweight package root (config eager; health lazy)."""

from intergrax.integrations._shared.config import (
    ENV_INTEGRATION_PREFIX,
    BaseIntegrationConfig,
    ProviderConfig,
    env_key_for_category,
    merge_config,
    read_integration_slug_from_env,
)

__all__ = [
    "ENV_INTEGRATION_PREFIX",
    "BaseIntegrationConfig",
    "ProviderConfig",
    "env_key_for_category",
    "health_check",
    "health_check_all",
    "health_check_entry",
    "merge_config",
    "read_integration_slug_from_env",
]

_HEALTH_EXPORTS = frozenset(
    {
        "health_check",
        "health_check_all",
        "health_check_entry",
    }
)


def __getattr__(name: str) -> object:
    if name in _HEALTH_EXPORTS:
        from intergrax.integrations._shared.health import (
            health_check,
            health_check_all,
            health_check_entry,
        )

        return {
            "health_check": health_check,
            "health_check_all": health_check_all,
            "health_check_entry": health_check_entry,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
