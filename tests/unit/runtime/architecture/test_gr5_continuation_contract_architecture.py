# © Artur Czarnecki. All rights reserved.

"""GR-5-R1 — continuation contract must stay contracts-only (no Nexus/runtime)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import ExecutionContinuationPort

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACT_MODULE = _REPO_ROOT / "intergrax" / "contracts" / "execution_continuation.py"


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_gr5_continuation_contract_no_runtime_or_nexus_imports() -> None:
    modules = _collect_imports(_CONTRACT_MODULE)
    forbidden = [
        m
        for m in modules
        if m.startswith("intergrax.runtime")
        or m.startswith("intergrax.applications")
        or "nexus" in m.lower()
    ]
    assert forbidden == []


def test_gr5_continuation_port_exposes_semantic_operations() -> None:
    required = {"request_pause", "get_pending", "apply_resolution", "resume"}
    names = {name for name, _ in inspect.getmembers(ExecutionContinuationPort)}
    assert required <= names
