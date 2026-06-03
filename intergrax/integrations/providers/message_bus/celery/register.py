# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register celery in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.celery.bundle import create_celery_message_bus
from intergrax.integrations.providers.message_bus.celery.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_celery_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_celery_message_bus, override=override)
