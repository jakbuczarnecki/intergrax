# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register pagerduty in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.pagerduty.bundle import create_pagerduty_notification_channel
from intergrax.integrations.providers.notification_channel.pagerduty.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.notification_channel.pagerduty.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_pagerduty_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_pagerduty_notification_channel,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )