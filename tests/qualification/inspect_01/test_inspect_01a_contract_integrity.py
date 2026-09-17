# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-A contract and architecture static gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.runtime_inspection import federation as federation_module

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_CONTRACT_DIR = _REPO_ROOT / "intergrax" / "contracts" / "runtime_inspection"
_FORBIDDEN_ABI = ("Any", "dict[str, Any]", "binding: object")


def test_a_q12_canonical_contracts_no_opaque_abi() -> None:
    for path in sorted(_CANONICAL_CONTRACT_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_ABI:
            assert token not in text, f"{path.name} contains forbidden ABI token {token!r}"


def test_a_q13_federation_static_gates() -> None:
    source = Path(federation_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "getattr" not in names
    assert "hasattr" not in names
    assert "GLOBAL_READERS" not in source
    assert "get_runtime_inspection" not in source
