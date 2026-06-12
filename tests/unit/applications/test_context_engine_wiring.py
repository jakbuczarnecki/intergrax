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
from intergrax.runtime.nexus.context.codebase_engine import CodebaseContextEngine
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.preset_engines import (
    ExploreChildContextEngine,
    RegulatedMinimalContextEngine,
)

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


def test_resolve_preset_engines() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ce.engine")
    env.context_profile = ContextProfile(engine_preset="regulated_minimal")
    assert isinstance(resolve_context_engine_from_environment(env), RegulatedMinimalContextEngine)
    env.context_profile = ContextProfile(engine_preset="explore_child")
    assert isinstance(resolve_context_engine_from_environment(env), ExploreChildContextEngine)


def test_resolve_custom_engine_ref() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ce.engine")
    env.context_profile = ContextProfile(
        engine_preset="custom",
        engine_ref="intergrax.runtime.nexus.context.codebase_engine.CodebaseContextEngine",
    )
    engine = resolve_context_engine_from_environment(env)
    assert isinstance(engine, CodebaseContextEngine)
