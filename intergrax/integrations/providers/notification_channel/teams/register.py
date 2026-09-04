# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register teams in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.teams.bundle import (
    create_teams_notification_channel,
)
from intergrax.integrations.providers.notification_channel.teams.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.notification_channel.teams.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_teams_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_teams_notification_channel,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )