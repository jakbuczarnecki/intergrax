# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-14 architecture guards (scale/resilience constraints)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_MEMORY_ROOT = _REPO / "intergrax" / "memory"
_MEM_ENT14_TESTS = _REPO / "tests" / "unit" / "memory"
_RESILIENCE_ROOT = _MEM_ENT14_TESTS / "resilience"


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _assigns_global_asyncio_lock(tree: ast.Module) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                if (
                    isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "asyncio"
                    and node.value.func.attr == "Lock"
                ):
                    violations.append(target.id)
    return violations


def test_memory_core_has_no_module_level_asyncio_lock() -> None:
    for path in _iter_py_files(_MEMORY_ROOT):
        tree = _parse(path)
        locks = _assigns_global_asyncio_lock(tree)
        assert not locks, f"{path} assigns module-level asyncio.Lock: {locks}"


def test_mem_ent14_resilience_helpers_no_baseexception_contract() -> None:
    for path in _iter_py_files(_RESILIENCE_ROOT):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "BaseException":
                pytest.fail(f"{path} references BaseException in resilience harness")
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                if isinstance(node.type, ast.Name) and node.type.id == "BaseException":
                    pytest.fail(f"{path} catches BaseException")


def test_mem_ent14_tests_avoid_sleep_based_races() -> None:
    patterns = ("test_mem_ent14_",)
    files = [
        path
        for path in _iter_py_files(_MEM_ENT14_TESTS)
        if any(part in path.name for part in patterns)
    ]
    files.extend(_iter_py_files(_RESILIENCE_ROOT))
    for path in files:
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "asyncio"
                    and node.func.attr == "sleep"
                ):
                    pytest.fail(f"{path} uses asyncio.sleep for concurrency proof")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "time"
                    and node.func.attr == "sleep"
                ):
                    pytest.fail(f"{path} uses time.sleep for concurrency proof")


def test_mem_ent14_resilience_helpers_no_unbounded_retry_loops() -> None:
    for path in _iter_py_files(_RESILIENCE_ROOT):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    pytest.fail(f"{path} contains while True without bounded exit")


def test_no_memory_owned_unbounded_queue_identified() -> None:
    """Documented guard: Memory core must not introduce unbounded queues in MEM-ENT-14 scope."""
    forbidden_queue_imports: list[str] = []
    for path in _iter_py_files(_MEMORY_ROOT):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in {"queue", "asyncio"}:
                    for alias in node.names:
                        if alias.name in {"Queue", "SimpleQueue"}:
                            forbidden_queue_imports.append(f"{path}:{alias.name}")
    assert not forbidden_queue_imports, forbidden_queue_imports
