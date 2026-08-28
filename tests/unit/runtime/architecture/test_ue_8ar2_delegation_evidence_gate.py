# © Artur Czarnecki. All rights reserved.

"""UE-8AR2 — no mutable holder output channel for delegation evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CHILD_RUNNER_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"
_GRAPH_EXECUTOR_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py"
)
_FORBIDDEN = "effective_delegation_holder"


def _assert_no_forbidden_symbol(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    rel = path.relative_to(_REPO_ROOT).as_posix()
    assert _FORBIDDEN not in source, (
        f"{rel} must not use {_FORBIDDEN!r} mutable delegation evidence carrier"
    )


def test_child_execution_runner_has_no_effective_delegation_holder() -> None:
    _assert_no_forbidden_symbol(_CHILD_RUNNER_PATH)


def test_graph_executor_has_no_effective_delegation_holder() -> None:
    _assert_no_forbidden_symbol(_GRAPH_EXECUTOR_PATH)
