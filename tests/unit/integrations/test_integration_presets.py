# © Artur Czarnecki. All rights reserved.

"""Integration profile presets resolve offline (Phase DX-4.4)."""

from __future__ import annotations

import pytest

from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry import presets

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(scope="module", autouse=True)
def _bootstrap() -> None:
    bootstrap_catalogs(register_shipped=True, integration_preset="core")


@pytest.mark.parametrize(
    "factory",
    [
        presets.lab_stack,
        presets.legal_stack,
        presets.research_stack,
        presets.data_stack,
        presets.observability_stack,
    ],
)
def test_preset_resolves_relational(factory) -> None:
    profile = factory()
    store = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
    assert store is not None


def test_profile_class_methods() -> None:
    assert IntegrationProfile.lab_stack() is not None
    assert IntegrationProfile.legal_stack() is not None
