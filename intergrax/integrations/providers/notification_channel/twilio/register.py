# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register twilio in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.twilio.bundle import create_twilio_notification_channel
from intergrax.integrations.providers.notification_channel.twilio.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.notification_channel.twilio.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_twilio_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_twilio_notification_channel,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )