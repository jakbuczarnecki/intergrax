# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register email_smtp in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.email_smtp.bundle import create_email_smtp_notification_channel
from intergrax.integrations.providers.notification_channel.email_smtp.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.notification_channel.email_smtp.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_email_smtp_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_email_smtp_notification_channel,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )