# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register grafana_oncall in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.grafana_oncall.bundle import create_grafana_oncall_notification_channel
from intergrax.integrations.providers.notification_channel.grafana_oncall.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.notification_channel.grafana_oncall.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_grafana_oncall_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_grafana_oncall_notification_channel,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )