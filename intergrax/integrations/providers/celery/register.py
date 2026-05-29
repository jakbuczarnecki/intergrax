# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Celery in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.celery.bundle import create_celery_message_bus
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_celery_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.CELERY.value,
            categories=(IntegrationCategory.MESSAGE_BUS,),
            factory=create_celery_message_bus,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_CELERY",
            description=(
                "Celery message bus — task queue + worker app "
                "(via create_celery_integration / create_celery_worker_app)"
            ),
        ),
        override=override,
    )
