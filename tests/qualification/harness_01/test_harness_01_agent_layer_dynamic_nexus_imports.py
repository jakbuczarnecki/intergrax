# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R3 — no dynamic Nexus resolution under ``intergrax/agents/**``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.harness_01.nexus_boundary_detector import (
    file_has_dynamic_nexus_import,
    file_imports_nexus_module,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_ROOT = _REPO_ROOT / "intergrax" / "agents"

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_harness_01_intergrax_agents_production_tree_has_zero_dynamic_nexus_imports() -> None:
    offenders: list[str] = []
    for path in sorted(_AGENTS_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if file_has_dynamic_nexus_import(source):
            offenders.append(str(path.relative_to(_REPO_ROOT)).replace("\\", "/"))
    assert offenders == [], (
        "intergrax/agents must not dynamically import intergrax.runtime.nexus:\n"
        + "\n".join(offenders)
    )


def test_nexus_boundary_detector_negative_static_import() -> None:
    source = "from intergrax.runtime.nexus.foo import Bar\n"
    assert file_imports_nexus_module(source) is True
    assert file_has_dynamic_nexus_import(source) is False


def test_nexus_boundary_detector_negative_dynamic_importlib() -> None:
    source = (
        "import importlib\n"
        'importlib.import_module("intergrax.runtime.nexus.foo")\n'
    )
    assert file_has_dynamic_nexus_import(source) is True


def test_nexus_boundary_detector_negative_dunder_import() -> None:
    source = '__import__("intergrax.runtime.nexus.foo")\n'
    assert file_has_dynamic_nexus_import(source) is True


def test_nexus_boundary_detector_positive_prose_comment() -> None:
    source = '# Agents must not import intergrax.runtime.nexus.*\n'
    assert file_imports_nexus_module(source) is False
    assert file_has_dynamic_nexus_import(source) is False


def test_nexus_boundary_detector_positive_contract_import() -> None:
    source = "from intergrax.contracts.agent_run import AgentRunRequest\n"
    assert file_imports_nexus_module(source) is False
    assert file_has_dynamic_nexus_import(source) is False
