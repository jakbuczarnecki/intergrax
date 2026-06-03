# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Celery integration provider (Phase M.4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.integrations._shared.conformance import assert_message_bus
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.message_bus.celery.bundle import (
    CeleryIntegrationBundle,
    create_celery_integration,
    create_celery_message_bus,
)
from intergrax.integrations.providers.message_bus.celery.register import register_celery_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


@pytest.fixture
def mock_app() -> MagicMock:
    return MagicMock(name="celery_app")


def test_create_celery_integration_bundle(mock_app: MagicMock) -> None:
    bundle = create_celery_integration(
        app=mock_app,
        broker_url="redis://test:6379/1",
        backend_url="redis://test:6379/2",
    )

    assert isinstance(bundle, CeleryIntegrationBundle)
    assert isinstance(bundle.message_bus, CeleryTaskQueue)
    assert bundle.app is mock_app
    assert bundle.config.broker_url == "redis://test:6379/1"
    assert bundle.config.backend_url == "redis://test:6379/2"


def test_create_celery_message_bus_uses_injected_app(mock_app: MagicMock) -> None:
    bus = create_celery_message_bus(app=mock_app)

    assert isinstance(bus, CeleryTaskQueue)
    assert bus._app is mock_app  # noqa: SLF001 — unit test wiring check


def test_register_and_resolve_via_profile(mock_app: MagicMock) -> None:
    register_celery_integration()
    profile = IntegrationProfile(message_bus="celery")

    bus = resolve(
        IntegrationCategory.MESSAGE_BUS,
        profile=profile,
        config={"app": mock_app},
    )

    assert_message_bus(bus)
    assert isinstance(bus, CeleryTaskQueue)


def test_register_default_integrations_includes_celery(mock_app: MagicMock) -> None:
    register_default_integrations()
    profile = IntegrationProfile(message_bus="celery")

    bus = resolve(
        IntegrationCategory.MESSAGE_BUS,
        profile=profile,
        config={"app": mock_app},
    )

    assert isinstance(bus, CeleryTaskQueue)
