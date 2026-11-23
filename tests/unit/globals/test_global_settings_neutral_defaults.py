# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_global_settings_neutral_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "INTERGRAX_DEFAULT_LANGUAGE",
        "INTERGRAX_DEFAULT_LOCALE",
        "INTERGRAX_DEFAULT_REGION",
        "INTERGRAX_DEFAULT_TIMEZONE",
    ):
        monkeypatch.delenv(key, raising=False)
    import intergrax.globals.settings as mod

    importlib.reload(mod)
    gs = mod.GlobalSettings()
    assert gs.default_language == "en"
    assert gs.default_locale == "en-US"
    assert gs.default_region == "en-US"
    assert gs.default_timezone == "UTC"
