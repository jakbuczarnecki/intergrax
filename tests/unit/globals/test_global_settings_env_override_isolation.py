# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_env_override_preserves_poland_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_DEFAULT_LOCALE", "pl-PL")
    monkeypatch.setenv("INTERGRAX_DEFAULT_LANGUAGE", "pl")
    monkeypatch.setenv("INTERGRAX_DEFAULT_REGION", "pl-PL")
    monkeypatch.setenv("INTERGRAX_DEFAULT_TIMEZONE", "Europe/Warsaw")
    import intergrax.globals.settings as mod

    importlib.reload(mod)
    gs = mod.GlobalSettings()
    assert gs.default_locale == "pl-PL"
    assert gs.default_timezone == "Europe/Warsaw"


def test_sequential_env_patches_do_not_bleed(monkeypatch: pytest.MonkeyPatch) -> None:
    import intergrax.globals.settings as mod

    monkeypatch.setenv("INTERGRAX_DEFAULT_LOCALE", "de-DE")
    importlib.reload(mod)
    assert mod.GlobalSettings().default_locale == "de-DE"

    monkeypatch.setenv("INTERGRAX_DEFAULT_LOCALE", "fr-FR")
    importlib.reload(mod)
    assert mod.GlobalSettings().default_locale == "fr-FR"
