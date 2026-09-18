# © Artur Czarnecki. All rights reserved.

"""PLUG-02 — integration entry-point admission report and isolation."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import EP_INTEGRATIONS, reset_entry_point_spec_cache_for_tests
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("catalog_fixture_installed")]


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_catalog()
    reset_default_integrations_state()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    yield
    clear_catalog()
    reset_default_integrations_state()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()


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


class _BrokenIntegration:
    pass


def test_integration_report_captures_invalid_sibling_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "broken_ep",
                f"{__name__}:_BrokenIntegration",
                EP_INTEGRATIONS,
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    result = bootstrap_catalogs(
        register_shipped=False,
        discover_entry_points=True,
        integration_plugins=(CustomMemoryKvPlugin,),
    )

    assert result.integration_plugins == 1
    report = result.integration_plugin_load_report
    assert report.group == EP_INTEGRATIONS
    assert report.registered_count == 0
    assert len(report.rejected) == 1
    assert report.rejected[0].reason_code == PluginAdmissionReasonCode.INVALID_TARGET_TYPE


def test_integration_report_ordering_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("z_bad", f"{__name__}:_BrokenIntegration", EP_INTEGRATIONS),
            _EntryPoint("a_bad", f"{__name__}:_BrokenIntegration", EP_INTEGRATIONS),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    first = bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    clear_catalog()
    second = bootstrap_catalogs(register_shipped=False, discover_entry_points=True)

    assert first.integration_plugin_load_report.rejected == second.integration_plugin_load_report.rejected
    assert first.integration_plugin_load_report.failed == second.integration_plugin_load_report.failed
