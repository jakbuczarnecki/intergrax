# © Artur Czarnecki. All rights reserved.

"""Total-outcome contract for ``register_plugins_with_report``."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.plugins.admission import (
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    EntryPointSpec,
    register_plugins_with_report,
    reset_entry_point_spec_cache_for_tests,
)

pytestmark = pytest.mark.unit

_GROUP = "intergrax.stage10.totality"


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


class _Plugin:
    pass


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [_EntryPoint("plugin", f"{__name__}:_Plugin", _GROUP)],
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)


def test_register_plugins_with_report_valid_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch)

    def _register(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        return True, None

    report = register_plugins_with_report(
        _GROUP,
        _register,
        discover_entry_points=True,
    )

    assert report.registered_count == 1
    assert len(report.accepted) == 1
    assert report.rejected == ()
    assert report.failed == ()


def test_register_plugins_with_report_valid_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch)

    def _register(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        return False, PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
            reason="not admitted",
            fail_closed=True,
        )

    report = register_plugins_with_report(
        _GROUP,
        _register,
        discover_entry_points=True,
    )

    assert report.registered_count == 0
    assert report.accepted == ()
    assert len(report.rejected) == 1
    assert report.failed == ()


def test_register_plugins_with_report_invalid_missing_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch)

    def _register(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        return False, None

    with pytest.raises(ValueError, match="registration callback must return exactly one outcome"):
        register_plugins_with_report(
            _GROUP,
            _register,
            discover_entry_points=True,
        )


def test_register_plugins_with_report_invalid_contradictory_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch)

    def _register(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        return True, PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
            reason="contradictory",
            fail_closed=True,
        )

    with pytest.raises(ValueError, match="registration callback must return exactly one outcome"):
        register_plugins_with_report(
            _GROUP,
            _register,
            discover_entry_points=True,
        )
