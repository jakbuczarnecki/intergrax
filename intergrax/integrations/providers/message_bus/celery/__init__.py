# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Celery integration — single public entry for all Celery-backed Tier-0 facades.

Implementation lives under ``intergrax.queueing.providers.celery`` and
``intergrax.queueing.worker_bootstrap``; compose only through this package.
"""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.message_bus.celery.config import (
    DEFAULT_APP_NAME,
    DEFAULT_BACKEND_URL,
    DEFAULT_BROKER_URL,
    ENV_CELERY_APP_NAME,
    ENV_CELERY_BACKEND_URL,
    ENV_CELERY_BROKER_URL,
    CeleryIntegrationConfig,
)

__all__ = [
    "DEFAULT_APP_NAME",
    "DEFAULT_BACKEND_URL",
    "DEFAULT_BROKER_URL",
    "ENV_CELERY_APP_NAME",
    "ENV_CELERY_BACKEND_URL",
    "ENV_CELERY_BROKER_URL",
    "CeleryIntegrationBundle",
    "CeleryIntegrationConfig",
    "create_celery_integration",
    "create_celery_message_bus",
    "create_celery_worker_app",
    "create_nexus_celery_worker_app",
    "register_celery_integration",
    "resolve_celery_config",
    "create_celery_message_bus_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "CeleryIntegrationBundle",
        "create_celery_integration",
        "create_celery_message_bus",
        "create_celery_worker_app",
        "create_nexus_celery_worker_app",
        "resolve_celery_config",
        "create_celery_message_bus_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "CELERY_MESSAGE_BUS_PROVIDER_ID",
        "CeleryMessageBusIntegration",
        "CeleryMessageBusIntegrationConfig",
        "CeleryMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_celery_integration":
        from intergrax.integrations.providers.message_bus.celery.register import register_celery_integration

        return register_celery_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.celery import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.celery import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
