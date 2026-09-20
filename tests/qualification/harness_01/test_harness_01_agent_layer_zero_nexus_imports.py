# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R2 — ``intergrax/agents/**`` must not AST-import Nexus."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.harness_01.nexus_boundary_detector import file_imports_nexus_module

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_ROOT = _REPO_ROOT / "intergrax" / "agents"

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_harness_01_intergrax_agents_production_tree_has_zero_nexus_imports() -> None:
    offenders: list[str] = []
    for path in sorted(_AGENTS_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if file_imports_nexus_module(source):
            offenders.append(str(path.relative_to(_REPO_ROOT)).replace("\\", "/"))
    assert offenders == [], "intergrax/agents must not import intergrax.runtime.nexus:\n" + "\n".join(
        offenders
    )
