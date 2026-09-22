# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]


def test_execution_suspended_contracts_do_not_import_nexus() -> None:
    root = REPO / "intergrax" / "contracts" / "execution" / "suspended_operation"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "runtime.nexus" not in text
        assert "nexus" not in text.lower() or "suspended_operation" in path.name


def test_suspended_payload_contracts_avoid_semantic_any() -> None:
    path = REPO / "intergrax" / "contracts" / "execution" / "suspended_operation"
    banned = {"Any", "dict[str, Any]", "Mapping[str, Any]"}
    for file in path.glob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for token in banned:
            assert token not in names, f"{file.name} contains banned {token}"


def test_aw_does_not_import_nexus() -> None:
    aw_root = REPO / "intergrax" / "autonomous_work"
    for path in aw_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "runtime.nexus" not in text
