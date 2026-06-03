# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register slack in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.slack.bundle import create_slack_catalog_factory
from intergrax.integrations.providers.notification_channel.slack.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_slack_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_slack_catalog_factory, override=override)
