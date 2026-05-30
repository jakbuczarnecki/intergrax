# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for log notification integration (Phase M.8)."""

from __future__ import annotations

import pytest

from intergrax.integrations._shared.conformance import assert_notification_channel
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.log.adapter import LogNotificationAdapter
from intergrax.integrations.providers.log.bundle import create_log_notification_channel
from intergrax.integrations.providers.log.register import register_log_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def test_create_log_notification_channel() -> None:
    channel = create_log_notification_channel()
    assert isinstance(channel, LogNotificationAdapter)
    assert_notification_channel(channel)


def test_register_and_resolve_log_via_profile() -> None:
    register_log_integration()
    profile = IntegrationProfile.lab()

    channel = resolve(
        IntegrationCategory.NOTIFICATION_CHANNEL,
        profile=profile,
    )

    assert isinstance(channel, LogNotificationAdapter)


def test_register_default_integrations_includes_log() -> None:
    register_default_integrations()
    profile = IntegrationProfile.lab()

    channel = resolve(
        IntegrationCategory.NOTIFICATION_CHANNEL,
        profile=profile,
    )

    assert isinstance(channel, LogNotificationAdapter)
