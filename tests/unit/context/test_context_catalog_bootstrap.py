# © Artur Czarnecki. All rights reserved.

"""CE-2.2–CE-2.3: Context catalog bootstrap and builtin providers."""

from __future__ import annotations

import pytest

from intergrax.context.bootstrap import (
    bootstrap_context_catalog,
    materialize_context_plugin_registry,
    reset_context_catalog_bootstrap_for_tests,
)
from intergrax.context.providers.builtin import BuiltinContextPlugin
from intergrax.context.registry import list_context_plugin_ids

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    yield
    reset_context_catalog_bootstrap_for_tests()


def test_bootstrap_registers_builtin_plugin() -> None:
    result = bootstrap_context_catalog()
    assert "intergrax.builtin" in result.catalog_plugin_ids
    assert result.context_plugins >= 0


def test_builtin_plugin_registers_at_least_ten_providers() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = registry.list_providers()
    assert len(providers) >= 10
    assert {provider.provider_id for provider in providers} >= set(
        BuiltinContextPlugin.builtin_provider_ids()
    )


def test_materialize_respects_enabled_plugin_ids() -> None:
    bootstrap_context_catalog()
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    assert len(registry.list_providers()) == len(BuiltinContextPlugin.builtin_provider_ids())


def test_bootstrap_is_idempotent() -> None:
    bootstrap_context_catalog()
    first = list_context_plugin_ids()
    bootstrap_context_catalog()
    assert list_context_plugin_ids() == first
