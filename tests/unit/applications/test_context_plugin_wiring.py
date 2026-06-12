# © Artur Czarnecki. All rights reserved.

"""CE-2.4: Context plugin id validation on environment wiring."""

from __future__ import annotations

import logging

import pytest

from intergrax.applications._shared.context_wiring import (
    resolve_context_plugin_registry_from_environment,
    validate_context_plugin_ids,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.context.bootstrap import reset_context_catalog_bootstrap_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    yield
    reset_context_catalog_bootstrap_for_tests()


def test_validate_unknown_plugin_fails_in_lab_mode() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ctx.plugins")
    env.context_profile = ContextProfile(context_plugin_ids=["does.not.exist"])

    with pytest.raises(ValueError, match="Unknown context plugin"):
        validate_context_plugin_ids(env, production_mode=False)


def test_validate_unknown_plugin_warns_in_production(caplog: pytest.LogCaptureFixture) -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ctx.plugins")
    env = env.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    env.context_profile = ContextProfile(context_plugin_ids=["missing.plugin"])

    with caplog.at_level(logging.WARNING):
        unknown = validate_context_plugin_ids(env, production_mode=True)

    assert unknown == ["missing.plugin"]
    assert "Unknown context plugin" in caplog.text


def test_resolve_registry_from_environment_uses_builtin_by_default() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ctx.plugins")
    registry = resolve_context_plugin_registry_from_environment(env)
    assert len(registry.list_providers()) >= 10
