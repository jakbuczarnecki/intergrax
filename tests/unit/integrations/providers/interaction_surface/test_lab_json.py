# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for lab JSON integration provider (Phase M.4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.integrations._shared.conformance import assert_interaction_surface
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.interaction_surface.lab_json.bundle import (
    LabJsonIntegrationBundle,
    create_lab_json_integration,
    create_lab_json_interaction_surface,
)
from intergrax.integrations.providers.interaction_surface.lab_json.integration import LabJsonInteractionSurfaceIntegration
from intergrax.integrations.providers.interaction_surface.lab_json.register import register_lab_json_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.interactions.factory import create_interaction_adapter, resolve_interaction_settings
from intergrax.runtime.interactions.metadata_keys import INTERACTION_CHANNEL_KEY

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


@pytest.fixture
def mock_interaction() -> MagicMock:
    adapter = MagicMock()
    adapter.channel = "lab"
    return adapter


def test_create_lab_json_integration_bundle(mock_interaction: MagicMock) -> None:
    bundle = create_lab_json_integration(interaction_adapter=mock_interaction)

    assert isinstance(bundle, LabJsonIntegrationBundle)
    assert isinstance(bundle.interaction_surface, LabJsonInteractionSurfaceIntegration)
    assert bundle.interaction_surface._runtime is mock_interaction


def test_create_lab_json_interaction_surface_default() -> None:
    surface = create_lab_json_interaction_surface()

    assert isinstance(surface, LabJsonInteractionSurfaceIntegration)
    assert surface.channel == "lab"
    assert surface.can_handle({"message": "hello", "capability": "echo.basic"})


def test_register_and_resolve_via_profile(mock_interaction: MagicMock) -> None:
    register_lab_json_integration()
    profile = IntegrationProfile(interaction_surface="lab_json")

    surface = resolve(
        IntegrationCategory.INTERACTION_SURFACE,
        profile=profile,
        config={"interaction_adapter": mock_interaction},
    )

    assert isinstance(surface, LabJsonInteractionSurfaceIntegration)
    assert surface._runtime is mock_interaction


def test_register_and_resolve_conformance() -> None:
    register_lab_json_integration()
    profile = IntegrationProfile(interaction_surface="lab_json")

    surface = resolve(IntegrationCategory.INTERACTION_SURFACE, profile=profile)

    assert_interaction_surface(surface)
    assert isinstance(surface, LabJsonInteractionSurfaceIntegration)


def test_register_default_integrations_includes_lab_json() -> None:
    register_default_integrations()
    profile = IntegrationProfile(interaction_surface="lab_json")

    surface = resolve(IntegrationCategory.INTERACTION_SURFACE, profile=profile)

    assert isinstance(surface, LabJsonInteractionSurfaceIntegration)


def test_runtime_interaction_factory_delegates_lab_surface() -> None:
    adapter = create_interaction_adapter(resolve_interaction_settings(surface="lab"))
    assert isinstance(adapter, LabJsonInteractionSurfaceIntegration)


def test_runtime_interaction_factory_delegates_lab_json_surface() -> None:
    adapter = create_interaction_adapter(resolve_interaction_settings(surface="lab_json"))
    assert isinstance(adapter, LabJsonInteractionSurfaceIntegration)


def test_lab_json_adapter_to_task_roundtrip() -> None:
    surface = create_lab_json_interaction_surface()
    task = surface.to_task(
        {
            "tenant_id": "t1",
            "user_id": "u1",
            "message": "run smoke test",
            "capability": "echo.basic",
        },
        tenant_id="fallback",
    )
    assert task.tenant_id == "t1"
    assert task.context.capability == "echo.basic"
    assert task.metadata[INTERACTION_CHANNEL_KEY] == "lab"
