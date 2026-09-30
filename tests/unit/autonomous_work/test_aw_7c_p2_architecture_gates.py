# © Artur Czarnecki. All rights reserved.

"""AW-7C-P2 architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_AW_IMPORTS = (
    "intergrax.runtime.tool_runtime",
    "intergrax.nexus",
)


def _sources_under(rel: str) -> list[Path]:
    root = Path(__file__).resolve().parents[3] / rel
    return sorted(root.rglob("*.py"))


def test_aw_scoped_adaptive_integration_no_tool_runtime() -> None:
    paths = [
        Path(__file__).resolve().parents[3]
        / "intergrax/autonomous_work/scoped_adaptive_integration.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in _FORBIDDEN_AW_IMPORTS:
                    assert not node.module.startswith(prefix), path


def test_no_second_qualification_service_symbol() -> None:
    aw_root = Path(__file__).resolve().parents[3] / "intergrax/autonomous_work"
    p4_only = aw_root / "scoped_adaptive_integration_execution.py"
    for path in aw_root.rglob("*.py"):
        if path == p4_only:
            continue
        text = path.read_text(encoding="utf-8")
        assert "A2CapabilityQualificationService" not in text
        assert "CapabilityQualificationService" not in text
