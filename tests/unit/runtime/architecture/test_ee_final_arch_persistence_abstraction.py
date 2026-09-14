# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — persistence port abstraction (no vendor store in execution core)."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import (
    ARCH_MODEL,
    EXECUTION_ROOT,
    _REPO_ROOT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EVENT_STORE_PREFIX = "intergrax.runtime.events.stores."
_EXECUTION_CORE_NAMES = (
    "runtime.py",
    "boundary.py",
    "host_task.py",
)


def test_ee_final_arch_persistence_model_documented() -> None:
    text = ARCH_MODEL.read_text(encoding="utf-8")
    assert "## 7. Persistence architecture" in text
    assert "port" in text.lower()


def test_ee_final_arch_execution_core_avoids_direct_event_store_imports() -> None:
    violations: list[str] = []
    for name in _EXECUTION_CORE_NAMES:
        path = EXECUTION_ROOT / name
        source = path.read_text(encoding="utf-8")
        if _EVENT_STORE_PREFIX in source:
            violations.append(name)
    assert violations == []


def test_ee_final_arch_eec1_cross_plane_gate_present() -> None:
    path = (
        _REPO_ROOT
        / "tests/unit/runtime/architecture/test_eec1_execution_engine_cross_plane_certification.py"
    )
    assert path.is_file()
