# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R4 — no lazy Nexus module resolution under ``intergrax/agents/**``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.harness_01.nexus_boundary_detector import (
    file_has_lazy_nexus_module_resolution,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_ROOT = _REPO_ROOT / "intergrax" / "agents"

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_harness_01_intergrax_agents_production_tree_has_zero_lazy_nexus_exports() -> None:
    offenders: list[str] = []
    for path in sorted(_AGENTS_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if file_has_lazy_nexus_module_resolution(source):
            offenders.append(str(path.relative_to(_REPO_ROOT)).replace("\\", "/"))
    assert offenders == [], (
        "intergrax/agents must not lazily resolve intergrax.runtime.nexus modules:\n"
        + "\n".join(offenders)
    )


def test_lazy_nexus_detector_negative_synthetic_map() -> None:
    source = """
_LAZY = {
    "X": ("intergrax.runtime.nexus.foo", "X"),
}

def __getattr__(name: str) -> object:
    module_path, attr = _LAZY[name]
    return export_from_import_path(module_path, attr)
"""
    assert file_has_lazy_nexus_module_resolution(source) is True


def test_lazy_nexus_detector_positive_prose_comment() -> None:
    source = 'DOC = "Agents must not depend on intergrax.runtime.nexus.*"\n'
    assert file_has_lazy_nexus_module_resolution(source) is False
