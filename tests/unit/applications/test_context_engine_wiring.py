# © Artur Czarnecki. All rights reserved.

"""CE-3.3: resolve_context_engine_from_environment."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.context_wiring import resolve_context_engine_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
)
from intergrax.context.bootstrap import reset_context_catalog_bootstrap_for_tests
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    yield
    reset_context_catalog_bootstrap_for_tests()


def test_resolve_context_engine_default_preset() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ce.engine")
    engine = resolve_context_engine_from_environment(env)
    assert isinstance(engine, DefaultNexusContextEngine)
    assert engine.engine_id == "default"
    assert len(engine.registry.list_providers()) >= 10


def test_resolve_custom_engine_ref_raises() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ce.engine")
    env.context_profile = ContextProfile(engine_preset="custom", engine_ref="pkg.Engine")
    with pytest.raises(ValueError, match="not wired"):
        resolve_context_engine_from_environment(env)
