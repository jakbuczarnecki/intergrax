# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSITION = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "decision_exposure_selection_composition.py"
)
_PLUGIN_COMPOSITION = _REPO_ROOT / "intergrax" / "runtime" / "decision_plugin_composition.py"
_REGISTRY = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "decision_exposure_selection_registry.py"
)


def test_registry_does_not_import_entry_point_targets() -> None:
    source = _REGISTRY.read_text(encoding="utf-8")
    assert "load_entry_point_value" not in source
    assert "load_decision_exposure_selection_strategy" not in source


def test_production_composition_uses_admitted_load_path() -> None:
    source = _PLUGIN_COMPOSITION.read_text(encoding="utf-8")
    assert "plan_decision_plugin_admission" in source
    assert "load_entry_point_targets_for_specs" in source
    assert "load_decision_exposure_selection_strategy(" not in source


def test_composition_module_does_not_call_direct_strategy_id_loader() -> None:
    source = _COMPOSITION.read_text(encoding="utf-8")
    assert "load_decision_exposure_selection_strategy(" not in source
