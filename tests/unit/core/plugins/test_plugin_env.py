# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.plugin_env import discover_plugins_enabled

pytestmark = pytest.mark.unit


def test_discover_plugins_enabled_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_DISCOVER_PLUGINS", raising=False)
    assert discover_plugins_enabled() is False


def test_discover_plugins_enabled_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    assert discover_plugins_enabled() is True
