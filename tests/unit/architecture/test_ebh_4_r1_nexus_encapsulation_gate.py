# © Artur Czarnecki. All rights reserved.

"""EBH-4-R1 — Execution Engine exclusive Nexus ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_INTERGRAX = _REPO / "intergrax"
_EE_ZONES = (
    _INTERGRAX / "runtime" / "execution",
    _INTERGRAX / "runtime" / "nexus",
)
_PRODUCTION_ROOTS = (
    _INTERGRAX,
    _REPO / "applications",
    _REPO / "agents",
)
_SKIP_DIR_PARTS = frozenset(
    {"tests", "testing_support", "__pycache__", "build", ".tmp"},
)


def _iter_production_py_files() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP_DIR_PARTS for part in path.parts):
                continue
            if path.parts[-2:] == ("tests", path.name) or "/tests/" in path.as_posix():
                continue
            if "docker" in path.parts and "runtime-context" in path.parts:
                continue
            files.append(path)
    return sorted(files)


def _in_ee_zone(path: Path) -> bool:
    try:
        rel = path.relative_to(_INTERGRAX)
    except ValueError:
        return False
    posix = rel.as_posix()
    return posix.startswith("runtime/execution/") or posix.startswith("runtime/nexus/")


def _imports_nexus_module(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "intergrax.runtime.nexus" or node.module.startswith(
                "intergrax.runtime.nexus."
            ):
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "intergrax.runtime.nexus" or alias.name.startswith(
                    "intergrax.runtime.nexus."
                ):
                    return True
    return False


def _imports_nexus_loop(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.nexus.nexus_loop":
            for alias in node.names:
                if alias.name == "NexusLoop":
                    return True
    return False


def _nexus_loop_construction_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "NexusLoop":
            count += 1
    return count


def test_ebh_4_r1_application_nexus_factory_removed() -> None:
    assert not (_INTERGRAX / "applications" / "_shared" / "nexus_factory.py").exists()


def test_ebh_4_r1_runtime_task_does_not_import_nexus_loop() -> None:
    worker = _INTERGRAX / "runtime" / "task" / "nexus_worker_execution.py"
    tree = ast.parse(worker.read_text(encoding="utf-8-sig"), filename=str(worker))
    assert not _imports_nexus_loop(tree)
    assert _nexus_loop_construction_count(tree) == 0


def test_ebh_4_r1_production_nexus_imports_outside_ee_are_zero() -> None:
    violations: list[str] = []
    for path in _iter_production_py_files():
        if _in_ee_zone(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except OSError:
            continue
        if _imports_nexus_module(tree):
            violations.append(path.relative_to(_REPO).as_posix())
    assert violations == [], f"runtime.nexus imports outside EE: {violations}"


def test_ebh_4_r1_import_gate_detects_fixture_violation(tmp_path: Path) -> None:
    fixture = tmp_path / "fake_production_leak.py"
    fixture.write_text(
        "from intergrax.runtime.nexus.nexus_loop import NexusLoop\n",
        encoding="utf-8",
    )
    tree = ast.parse(fixture.read_text(encoding="utf-8"), filename=str(fixture))
    assert _imports_nexus_module(tree)


def test_ebh_4_r1_scenario_runtime_public_surface_has_no_orchestration_materialization() -> None:
    scenario = _INTERGRAX / "applications" / "_shared" / "scenario_runtime_baseline.py"
    tree = ast.parse(scenario.read_text(encoding="utf-8-sig"), filename=str(scenario))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ScenarioRuntimeComposition":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == "orchestration":
                        pytest.fail(
                            "ScenarioRuntimeComposition must not expose orchestration materialization",
                        )


def test_ebh_4_r1_nexus_loop_construction_outside_ee_is_zero() -> None:
    violations: list[str] = []
    for path in _iter_production_py_files():
        if _in_ee_zone(path):
            continue
        if path.name in {"app.py"} and "debug" in path.parts:
            continue
        if path.name == "workflow.py" and "experiments" in path.parts:
            continue
        if path.name == "organization_worker.py" and "lab" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except OSError:
            continue
        count = _nexus_loop_construction_count(tree)
        if count:
            rel = path.relative_to(_REPO).as_posix()
            violations.append(f"{rel}: NexusLoop()={count}")
    assert violations == []


def test_ebh_4_r1_worker_no_reference_allowing_admission() -> None:
    worker = _INTERGRAX / "runtime" / "task" / "nexus_worker_execution.py"
    source = worker.read_text(encoding="utf-8-sig")
    assert "build_reference_allowing_root_execution_authority_admission" not in source
