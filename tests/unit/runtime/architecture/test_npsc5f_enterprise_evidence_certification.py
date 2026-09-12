# © Artur Czarnecki. All rights reserved.

"""NPSC-5F — Enterprise Evidence Plane certification (architecture freeze gates)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from testing_support.npsc5f_final_evidence_plane_ownership import (
    collect_forbidden_execution_control_calls,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXECUTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution"
_EVENT_BUS_PATH = _REPO_ROOT / "intergrax" / "runtime" / "events" / "event_bus.py"
_RECONSTRUCTION_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "execution_reconstruction.py",
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "historical_reconstruction.py",
)
_EVENTS_TOP_LEVEL = _REPO_ROOT / "intergrax" / "runtime" / "events"
_PERSISTENCE_PORT_PATH = (
    _REPO_ROOT / "intergrax" / "contracts" / "execution_evidence" / "persistence_port.py"
)

_FORBIDDEN_EXECUTION_ENGINE_STORE_MODULES = frozenset(
    {
        "intergrax.runtime.events.stores.sqlite_runtime_event_store",
        "intergrax.runtime.events.stores.document_backed_runtime_event_store",
        "intergrax.runtime.events.stores.memory_runtime_event_store",
    },
)

_FORBIDDEN_RECONSTRUCTION_MODULES = frozenset(
    {
        "intergrax.runtime.execution",
        "intergrax.runtime.replay",
        "intergrax.runtime.nexus",
        "agents",
        "applications",
    },
)


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _module_prefix_violations(modules: set[str], forbidden_prefixes: frozenset[str]) -> list[str]:
    violations: list[str] = []
    for name in sorted(modules):
        for prefix in forbidden_prefixes:
            if name == prefix or name.startswith(f"{prefix}."):
                violations.append(name)
                break
    return violations


def test_npsc5f_cert_execution_engine_has_no_concrete_event_store_imports() -> None:
    violations: list[str] = []
    for path in _EXECUTION_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for module in _imported_modules(tree):
            if module in _FORBIDDEN_EXECUTION_ENGINE_STORE_MODULES or module.startswith(
                "intergrax.runtime.events.stores."
            ):
                violations.append(f"{rel}:{module}")
    assert violations == []


def test_npsc5f_cert_event_bus_depends_on_evidence_persistence_port_only() -> None:
    source = _EVENT_BUS_PATH.read_text(encoding="utf-8")
    assert "RuntimeEventPersistence" not in source
    assert "EvidencePersistencePort" in source
    assert "as_evidence_persistence_port" in source


def test_npsc5f_cert_evidence_persistence_port_contract_is_stable() -> None:
    source = _PERSISTENCE_PORT_PATH.read_text(encoding="utf-8")
    for method in (
        "def append",
        "def list_positioned_for_run",
        "def list_for_task",
        "def get_by_event_id",
        "def list_positioned_through",
    ):
        assert method in source
    assert "class EvidencePersistencePort" in source


def test_npsc5f_cert_reconstruction_isolated_from_execution_control() -> None:
    violations: list[str] = []
    for path in _RECONSTRUCTION_PATHS:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _module_prefix_violations(
            _imported_modules(tree),
            _FORBIDDEN_RECONSTRUCTION_MODULES,
        ):
            violations.append(f"{rel}:{module}")
    assert violations == []


def test_npsc5f_cert_reconstruction_reads_through_evidence_port_type() -> None:
    source = _RECONSTRUCTION_PATHS[0].read_text(encoding="utf-8")
    assert "EvidencePersistencePort" in source


def test_npsc5f_cert_persistence_roots_have_no_execution_control_calls() -> None:
    assert collect_forbidden_execution_control_calls(_REPO_ROOT) == []


def test_npsc5f_cert_single_durable_runtime_event_append_entry_in_events_layer() -> None:
    """Durable ``RuntimeEvent`` commit flows: bus → port → adapter/store — not parallel writers."""
    producers: list[str] = []
    for path in _EVENTS_TOP_LEVEL.glob("*.py"):
        if path.name in {"evidence_persistence_adapter.py", "persistence_contract.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "append":
                continue
            if not isinstance(node.func.value, ast.Attribute):
                continue
            if node.func.value.attr != "_persistence":
                continue
            producers.append(path.name)
    assert producers == ["event_bus.py"]
