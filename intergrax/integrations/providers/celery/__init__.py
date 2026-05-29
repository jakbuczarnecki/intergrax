# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Celery integration — single public entry for all Celery-backed Tier-0 facades.

Implementation lives under ``intergrax.queueing.providers.celery`` and
``intergrax.queueing.worker_bootstrap``; compose only through this package.
"""

from intergrax.integrations.providers.celery.config import (
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
]

_BUNDLE_EXPORTS = frozenset(
    {
        "CeleryIntegrationBundle",
        "create_celery_integration",
        "create_celery_message_bus",
        "create_celery_worker_app",
        "create_nexus_celery_worker_app",
        "resolve_celery_config",
    }
)


def __getattr__(name: str):
    if name == "register_celery_integration":
        from intergrax.integrations.providers.celery.register import register_celery_integration

        return register_celery_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.celery import bundle as _bundle

        return getattr(_bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
