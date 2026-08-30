# © Artur Czarnecki. All rights reserved.

"""UE-9BR1 — single canonical RuntimeEvent schema architecture gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.events.runtime_event import RuntimeEvent

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNTIME_ROOT = _REPO_ROOT / "intergrax" / "runtime"

_RUNTIME_EVENT_PRODUCTION_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "signals.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "spine_consolidation.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "context_skill_recording.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "trace_bridge.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "planner_events.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "ingestion_events.py",
    _REPO_ROOT / "intergrax" / "runtime" / "kernel" / "step_kernel.py",
    _REPO_ROOT / "intergrax" / "runtime" / "middleware" / "trace_middleware.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "agent_router.py",
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "trace_scope.py",
    _REPO_ROOT / "intergrax" / "agents" / "uaep.py",
    _REPO_ROOT / "intergrax" / "contracts" / "runtime_execution_context.py",
)

_EVENT_OBS_MINT_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "events",
    _REPO_ROOT / "intergrax" / "runtime" / "observability",
)

_EXCLUDED_PARTS = frozenset({"__pycache__", "tests"})
_CONFORMANCE_EXCLUDED = frozenset(
    {
        "persistence_conformance.py",
    }
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_forbidden_calls(path: Path, forbidden: frozenset[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in forbidden:
            violations.append(f"{rel}:{node.lineno} calls {name}")
    return violations


def _runtime_python_files() -> list[Path]:
    files: list[Path] = []
    for path in _RUNTIME_ROOT.rglob("*.py"):
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def _event_obs_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _EVENT_OBS_MINT_PATHS:
        for path in root.rglob("*.py"):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            if path.name in _CONFORMANCE_EXCLUDED:
                continue
            files.append(path)
    return files


def _collect_runtime_event_v1_literals() -> list[str]:
    violations: list[str] = []
    for path in _runtime_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "runtime_event.v1" in line:
                violations.append(f"{rel}:{lineno}")
    return violations


def _has_execution_id_source(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "execution_id":
            return True
        if keyword.arg is None and isinstance(keyword.value, ast.Call):
            if _call_name(keyword.value.func) == "runtime_event_identity_kwargs":
                return True
    return False


def _collect_runtime_event_constructions_without_execution_id() -> list[str]:
    violations: list[str] = []
    for path in _runtime_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name != "RuntimeEvent":
                continue
            if not _has_execution_id_source(node):
                violations.append(f"{rel}:{node.lineno} RuntimeEvent(...) missing execution_id")
    return violations


def test_runtime_event_execution_id_field_is_required() -> None:
    field = RuntimeEvent.model_fields["execution_id"]
    assert field.is_required()


def test_production_runtime_has_no_runtime_event_v1_literals() -> None:
    assert _collect_runtime_event_v1_literals() == []


def test_production_runtime_event_constructions_include_execution_id() -> None:
    assert _collect_runtime_event_constructions_without_execution_id() == []


def test_production_runtime_event_emitters_do_not_mint_execution_id() -> None:
    violations: list[str] = []
    for path in _RUNTIME_EVENT_PRODUCTION_PATHS:
        violations.extend(
            _collect_forbidden_calls(path, frozenset({"mint_execution_id"}))
        )
    assert violations == []


def test_event_and_obs_production_code_do_not_mint_execution_id() -> None:
    violations: list[str] = []
    for path in _event_obs_python_files():
        violations.extend(
            _collect_forbidden_calls(path, frozenset({"mint_execution_id"}))
        )
    assert violations == []
