# © Artur Czarnecki. All rights reserved.

"""CE-2.2–CE-2.3: Context catalog bootstrap and builtin providers."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.context.bootstrap import (
    ContextCatalogBootstrapResult,
    bootstrap_context_catalog,
    materialize_context_plugin_registry,
    reset_context_catalog_bootstrap_for_tests,
)
from intergrax.context.plugin import ContextPlugin
from intergrax.context.providers.builtin import BuiltinContextPlugin
from intergrax.context.provider_descriptor import build_provider_descriptor
from intergrax.context.registry import ContextPluginRegistry, clear_context_plugin_catalog, list_context_plugin_ids
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import EP_CONTEXT, reset_entry_point_spec_cache_for_tests
from intergrax.core.plugins.errors import PluginLoadError

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _StubProvider:
    provider_id = "acme.stub"

    @property
    def supported_sources(self):
        return frozenset({ContextFragmentSource.CUSTOM})

    @property
    def descriptor(self):
        return build_provider_descriptor(
            self.provider_id,
            provider_version="0.1.0",
            supported_sources=self.supported_sources,
            origin="plugin:acme.context",
        )

    async def collect(self, request: object, ctx: object) -> list[object]:
        return []


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


class _UnsupportedContextTarget:
    value = "not-a-plugin"


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_context_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()


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


def test_discovery_disabled_load_report_is_empty() -> None:
    result = bootstrap_context_catalog(
        register_shipped=False,
        discover_entry_points=False,
    )

    assert result.load_report.group == EP_CONTEXT
    assert result.load_report.accepted == ()
    assert result.load_report.rejected == ()
    assert result.load_report.failed == ()
    assert result.load_report.registered_count == 0


def test_healthy_entry_point_accepted_and_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "acme_context",
                f"{__name__}:_AcmeContextPlugin",
                EP_CONTEXT,
            )
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = bootstrap_context_catalog(
        register_shipped=False,
        discover_entry_points=True,
    )

    assert "acme.context" in result.catalog_plugin_ids
    assert len(result.load_report.accepted) == 1
    assert result.load_report.accepted[0].name == "acme_context"
    assert result.load_report.registered_count == 1
    assert result.context_plugins == 1


def test_broken_entry_point_fail_fast_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("broken_ep", "not-a-valid-target", EP_CONTEXT)]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    with pytest.raises(PluginLoadError):
        bootstrap_context_catalog(
            register_shipped=False,
            discover_entry_points=True,
        )


def test_explicit_plugin_registers_without_ep_accepted_row() -> None:
    result = bootstrap_context_catalog(
        register_shipped=False,
        context_plugins=[_AcmeContextPlugin],
        discover_entry_points=False,
    )

    assert "acme.context" in result.catalog_plugin_ids
    assert result.context_plugins == 1
    assert result.load_report.accepted == ()
    assert result.load_report.registered_count == 0


def test_shipped_builtin_not_in_accepted_external_ep_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "acme_context",
                f"{__name__}:_AcmeContextPlugin",
                EP_CONTEXT,
            )
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = bootstrap_context_catalog(discover_entry_points=True)

    assert "intergrax.builtin" in result.catalog_plugin_ids
    accepted_names = {spec.name for spec in result.load_report.accepted}
    assert "intergrax.builtin" not in accepted_names
    assert accepted_names == {"acme_context"}


def test_canonical_context_plugin_protocol_check() -> None:
    assert issubclass(_AcmeContextPlugin, ContextPlugin)
    assert not issubclass(_UnsupportedContextTarget, ContextPlugin)


def test_unsupported_target_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "unsupported_ep",
                f"{__name__}:_UnsupportedContextTarget",
                EP_CONTEXT,
            )
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = bootstrap_context_catalog(
        register_shipped=False,
        discover_entry_points=True,
    )

    assert result.load_report.rejected
    assert (
        result.load_report.rejected[0].reason_code
        is PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    )
    assert result.load_report.registered_count == 0


def test_skip_conflict_records_structured_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "acme_context",
                f"{__name__}:_AcmeContextPlugin",
                EP_CONTEXT,
            ),
            _EntryPoint(
                "acme_duplicate",
                f"{__name__}:_AcmeContextPlugin",
                EP_CONTEXT,
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = bootstrap_context_catalog(
        register_shipped=False,
        discover_entry_points=True,
        on_conflict="skip",
    )

    assert "acme.context" in result.catalog_plugin_ids
    assert len(result.load_report.accepted) == 1
    assert len(result.load_report.rejected) == 1
    assert (
        result.load_report.rejected[0].reason_code
        is PluginAdmissionReasonCode.PLUGIN_ID_SKIPPED
    )


def test_bootstrap_result_is_frozen() -> None:
    result = bootstrap_context_catalog(register_shipped=False, discover_entry_points=False)
    assert isinstance(result, ContextCatalogBootstrapResult)
    with pytest.raises(AttributeError):
        result.context_plugins = 1  # type: ignore[misc]
