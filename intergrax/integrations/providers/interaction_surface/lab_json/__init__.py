# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Lab JSON integration — single public entry for laboratory interaction intake.

Implementation lives under ``runtime.interactions.adapters.lab_json_adapter``;
compose only through this package.
"""

from intergrax.integrations.providers.interaction_surface.lab_json.config import (
    DEFAULT_SOURCE,
    ENV_LAB_JSON_DEFAULT_SOURCE,
    LabJsonIntegrationConfig,
)

__all__ = [
    "DEFAULT_SOURCE",
    "ENV_LAB_JSON_DEFAULT_SOURCE",
    "LabJsonIntegrationBundle",
    "LabJsonIntegrationAdapter",
    "LabJsonIntegrationConfig",
    "create_lab_json_integration",
    "create_lab_json_interaction_surface",
    "register_lab_json_integration",
    "resolve_lab_json_config",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "LabJsonIntegrationBundle",
        "create_lab_json_integration",
        "create_lab_json_interaction_surface",
        "resolve_lab_json_config",
    }
)


def __getattr__(name: str):
    if name == "register_lab_json_integration":
        from intergrax.integrations.providers.interaction_surface.lab_json.register import register_lab_json_integration

        return register_lab_json_integration
    if name == "LabJsonIntegrationAdapter":
        from intergrax.integrations.providers.interaction_surface.lab_json.adapter import LabJsonIntegrationAdapter

        return LabJsonIntegrationAdapter
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.interaction_surface.lab_json import bundle as _bundle

        return getattr(_bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
