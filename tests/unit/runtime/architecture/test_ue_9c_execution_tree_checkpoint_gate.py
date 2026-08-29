# © Artur Czarnecki. All rights reserved.

"""UE-9C — canonical Execution Tree checkpoint architecture gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.long_running.execution_tree_checkpoint import ExecutionCheckpointEntry
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

_FORBIDDEN_SYMBOLS = frozenset(
    {
        "task_run_id_placeholder",
        "RUNTIME_CHECKPOINT_KEY",
        "runtime_checkpoint_from_metadata",
        "runtime_checkpoint_from_execution_structured",
        "attach_runtime_checkpoint_to_metadata",
        "RuntimeCheckpointExecutionState",
    }
)

_PRODUCTION_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "long_running",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py",
    _REPO_ROOT / "intergrax" / "agents" / "uaep.py",
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "unified_task_runner.py",
)


def _collect_forbidden_symbols(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return [f"{rel} contains {symbol}" for symbol in _FORBIDDEN_SYMBOLS if symbol in text]


def test_runtime_checkpoint_has_execution_tree_field() -> None:
    assert "execution_tree" in RuntimeCheckpoint.model_fields


def test_execution_checkpoint_entry_has_parent_execution_id() -> None:
    assert "parent_execution_id" in ExecutionCheckpointEntry.model_fields
    assert "execution_id" in ExecutionCheckpointEntry.model_fields


def test_production_code_has_no_legacy_checkpoint_identity_symbols() -> None:
    violations: list[str] = []
    for base in _PRODUCTION_PATHS:
        if base.is_dir():
            for path in base.rglob("*.py"):
                violations.extend(_collect_forbidden_symbols(path))
        else:
            violations.extend(_collect_forbidden_symbols(base))
    assert violations == []


def test_production_code_has_no_second_checkpoint_persistence_framework() -> None:
    long_running = _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "persistence_contract.py"
    text = long_running.read_text(encoding="utf-8")
    assert "class TaskCheckpointPersistence" in text
    tree = ast.parse(text, filename=str(long_running))
    class_names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert "ExecutionTreeCheckpointPersistence" not in class_names
