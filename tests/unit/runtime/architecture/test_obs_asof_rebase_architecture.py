# © Artur Czarnecki. All rights reserved.

"""OBS-ASOF-REBASE — canonical historical execution query architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_HISTORICAL_CLUSTER = (
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "historical_reconstruction.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "asof_projection.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "unified_run_journal.py",
)

_FORBIDDEN_CONTROL = frozenset(
    {
        "TaskRunner",
        "NexusLoop",
        "RetryEngine",
        "ExecutionRuntime",
        "HistoricalRuntime",
        "ReplayRuntime",
    }
)

_TIMESTAMP_BOUNDARY_PATTERNS = (
    re.compile(r"AsOfBoundary\s*\([^)]*timestamp", re.IGNORECASE),
    re.compile(r"execution_as_of\s*=\s*datetime", re.IGNORECASE),
    re.compile(r"load_events_before_timestamp"),
    re.compile(r"filter_events_until\s*\([^)]*timestamp"),
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_obs_asof_rebase_historical_service_composes_canonical_path() -> None:
    path = _HISTORICAL_CLUSTER[0]
    text = path.read_text(encoding="utf-8")
    assert "reconstruct_run_execution_as_of" in text
    assert "ExecutionReconstructor" in text
    imports = _module_imports(path)
    assert "intergrax.runtime.observability.reconstruction" in imports
    assert "intergrax.runtime.diagnostics" not in imports


def test_obs_asof_rebase_no_second_reconstructor_in_cluster() -> None:
    for path in _HISTORICAL_CLUSTER[:2]:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        class_names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ExecutionReconstructor"
        ]
        assert class_names == []


def test_obs_asof_rebase_cluster_no_diagnostics_imports() -> None:
    for path in _HISTORICAL_CLUSTER:
        text = path.read_text(encoding="utf-8")
        assert "intergrax.runtime.diagnostics" not in text, path.relative_to(_REPO_ROOT)


def test_obs_asof_rebase_cluster_no_control_plane_surfaces() -> None:
    for path in _HISTORICAL_CLUSTER:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert names.isdisjoint(_FORBIDDEN_CONTROL), path.relative_to(_REPO_ROOT)


def test_obs_asof_rebase_no_timestamp_execution_boundary_patterns() -> None:
    for path in _HISTORICAL_CLUSTER:
        text = path.read_text(encoding="utf-8")
        for pattern in _TIMESTAMP_BOUNDARY_PATTERNS:
            assert pattern.search(text) is None, (
                f"INVALID_EXECUTION_BOUNDARY pattern {pattern.pattern!r} in {path.relative_to(_REPO_ROOT)}"
            )


def test_obs_asof_rebase_canonical_prefix_authority() -> None:
    journal = _HISTORICAL_CLUSTER[2].read_text(encoding="utf-8")
    assert "load_positioned_run_journal_through" in journal
    asof = _HISTORICAL_CLUSTER[1].read_text(encoding="utf-8")
    assert "load_positioned_run_journal_through" in asof
