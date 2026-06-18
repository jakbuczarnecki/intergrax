# © Artur Czarnecki. All rights reserved.

"""CE-MAINT-03 preset regression baseline fixtures."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.context_presets import (
    codebase_context_profile,
    regulated_minimal_context_profile,
)
from intergrax.runtime.nexus.context.codebase_engine import CodebaseContextEngine
from intergrax.runtime.nexus.context.preset_engines import RegulatedMinimalContextEngine

pytestmark = pytest.mark.gate

_BASELINE_PRESETS: dict[str, str] = {
    "codebase": "codebase",
    "regulated_minimal": "regulated_minimal",
}


@pytest.mark.parametrize("preset_name,engine_preset", list(_BASELINE_PRESETS.items()))
def test_context_preset_engine_id_matches_baseline(preset_name: str, engine_preset: str) -> None:
    profiles = {
        "codebase": codebase_context_profile(),
        "regulated_minimal": regulated_minimal_context_profile(),
    }
    engine_factories = {
        "codebase": CodebaseContextEngine,
        "regulated_minimal": RegulatedMinimalContextEngine,
    }
    profile = profiles[preset_name]
    assert profile.engine_preset == engine_preset
    engine = engine_factories[preset_name]()
    assert engine.engine_id == engine_preset
