# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

import pytest

from intergrax.integrations import IntegrationProfile, register_default_integrations
from intergrax.integrations.registry.catalog import clear_catalog as clear_integration_catalog
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _integrations_catalog() -> None:
    clear_integration_catalog()
    register_default_integrations()
    yield
    clear_integration_catalog()


def test_from_integration_profile_lab_resolves_notification() -> None:
    profile = IntegrationProfile.lab()
    ctx = ToolWiringContext.from_integration_profile(profile)

    assert ctx.issue_tracker is None
    assert ctx.search_provider is None
    assert ctx.notification_channel is not None
