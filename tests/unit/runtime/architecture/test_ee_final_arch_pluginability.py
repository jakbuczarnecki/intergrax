# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — plugin capability contracts without execution authority injection."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import ARCH_MODEL, _REPO_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PLUGIN_GATES = (
    "tests/unit/runtime/architecture/test_hardening_5_plugin_architecture_gate.py",
    "tests/unit/runtime/architecture/test_ds_plugin_architecture_gates.py",
    "tests/unit/runtime/architecture/test_plugin_ep_scanner_consolidation_gate.py",
)


def test_ee_final_arch_pluginability_documented() -> None:
    text = ARCH_MODEL.read_text(encoding="utf-8")
    assert "## 8. Pluginability model" in text
    assert "manifest" in text.lower() or "contract" in text.lower()
    assert "arbitrary" in text.lower() or "registration alone" in text.lower()


def test_ee_final_arch_frozen_plugin_gates_exist() -> None:
    missing = [rel for rel in _PLUGIN_GATES if not (_REPO_ROOT / rel).is_file()]
    assert missing == []
