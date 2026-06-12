# © Artur Czarnecki. All rights reserved.

"""CE-1.4: Context plugin registry and registration."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.plugin import ContextPlugin, register_context_plugin
from intergrax.context.registry import (
    ContextPluginRegistry,
    UnknownContextPluginError,
    clear_context_plugin_catalog,
    get_context_plugin,
    list_context_plugin_ids,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _StubProvider:
    provider_id = "acme.stub"
    supported_sources = frozenset({ContextFragmentSource.CUSTOM})

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        return [
            ContextFragment(
                fragment_id="stub-1",
                source=ContextFragmentSource.CUSTOM,
                source_id="acme",
                content="stub",
                token_estimate=1,
                relevance_score=0.5,
                freshness_score=0.5,
                confidence_score=0.5,
                mandatory=False,
            )
        ]


class _AcmeContextPlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "acme.context"

    @classmethod
    def plugin_version(cls) -> str:
        return "0.1.0"

    @classmethod
    def plugin_description(cls) -> str:
        return "test plugin"

    @classmethod
    def register(cls, registry: ContextPluginRegistry) -> None:
        registry.add_provider(_StubProvider())


@pytest.fixture(autouse=True)
def _clear_catalog() -> None:
    clear_context_plugin_catalog()
    yield
    clear_context_plugin_catalog()


def test_registry_add_list_unregister_provider() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(_StubProvider())

    providers = registry.list_providers()
    assert len(providers) == 1
    assert providers[0].provider_id == "acme.stub"

    registry.remove_provider("acme.stub")
    assert registry.list_providers() == ()


def test_registry_rejects_duplicate_provider() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(_StubProvider())
    with pytest.raises(ValueError, match="already registered"):
        registry.add_provider(_StubProvider())


def test_register_context_plugin_catalog() -> None:
    register_context_plugin(_AcmeContextPlugin)
    assert list_context_plugin_ids() == ["acme.context"]

    entry = get_context_plugin("acme.context")
    registry = ContextPluginRegistry()
    entry.register_into(registry)

    assert len(registry.list_providers()) == 1


def test_unknown_plugin_raises() -> None:
    with pytest.raises(UnknownContextPluginError):
        get_context_plugin("missing.plugin")


def test_context_plugin_protocol_runtime_checkable() -> None:
    assert isinstance(_AcmeContextPlugin, ContextPlugin)
