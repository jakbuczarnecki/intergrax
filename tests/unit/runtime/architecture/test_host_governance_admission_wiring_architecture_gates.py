# © Artur Czarnecki. All rights reserved.

"""Architecture gates — explicit host governance identity admission wiring (P2C-R0A-R1-R1)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NEXUS_HOST = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "nexus_host_execution.py"
_SHARED_HOST_WIRING = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "host_task_execution_wiring.py"
)
_HARNESS_HOST_WIRING = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "harness_host_task_execution_wiring.py"
)
_RUNTIME_ROOT = _REPO_ROOT / "intergrax" / "runtime"
_SHARED_APPS_ROOT = _REPO_ROOT / "intergrax" / "applications" / "_shared"

_FORBIDDEN_HARNESS_IDENTITY_IMPORTS = frozenset(
    {
        "admit_harness_root_governance_identity",
        "admit_certified_internal_harness_root_governance_identity",
    },
)

_FORBIDDEN_HARNESS_IDENTITY_MODULES = frozenset(
    {
        "intergrax.applications._shared.harness_admitted_root_governance_identity",
        "intergrax.runtime.execution.certified_internal_harness_governance_identity",
    },
)

_IMPORT_ALLOWLIST_REL = frozenset(
    {
        "intergrax/applications/_shared/harness_admitted_root_governance_identity.py",
        "intergrax/applications/_shared/harness_host_task_execution_wiring.py",
        "intergrax/runtime/execution/certified_internal_harness_governance_identity.py",
    },
)

_NEXUS_WORKER_EXECUTION = (
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "nexus_worker_execution.py"
)


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _iter_py_modules(root: Path) -> list[Path]:
    modules: list[Path] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        modules.append(path)
    return sorted(modules)


def _imported_harness_identity_symbols(tree: ast.AST) -> set[str]:
    hits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_HARNESS_IDENTITY_MODULES:
            for alias in node.names:
                hits.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                if alias.name in _FORBIDDEN_HARNESS_IDENTITY_IMPORTS:
                    hits.add(alias.name)
    return hits


def test_nexus_host_execution_requires_explicit_admit_root_governance_identity() -> None:
    import inspect

    from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution

    source = _NEXUS_HOST.read_text(encoding="utf-8-sig")
    assert "admit_certified_internal_harness_root_governance_identity" not in source
    assert " or admit_" not in source.replace(" ", "")
    signature = inspect.signature(build_host_task_execution)
    assert signature.parameters["admit_root_governance_identity"].default is inspect.Parameter.empty
    assert signature.parameters["root_authority_admission"].default is inspect.Parameter.empty


def test_shared_host_wiring_does_not_import_harness_identity_admission() -> None:
    source = _SHARED_HOST_WIRING.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_SHARED_HOST_WIRING))
    assert _imported_harness_identity_symbols(tree) == set()
    assert "admit_harness_root_governance_identity" not in source
    assert "build_harness_root_execution_authority_admission" not in source


def test_harness_host_wiring_injects_explicit_harness_identity_admission() -> None:
    source = _HARNESS_HOST_WIRING.read_text(encoding="utf-8-sig")
    assert "admit_harness_root_governance_identity" in source
    assert "build_harness_root_execution_authority_admission" in source


def test_nexus_worker_execution_does_not_import_harness_identity_admission() -> None:
    source = _NEXUS_WORKER_EXECUTION.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_NEXUS_WORKER_EXECUTION))
    assert _imported_harness_identity_symbols(tree) == set()
    assert "admit_certified_internal_harness_root_governance_identity" not in source


def test_nexus_worker_from_registry_requires_explicit_governance_admission() -> None:
    import inspect

    from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime

    signature = inspect.signature(NexusWorkerRuntime.from_registry)
    param = signature.parameters["admit_root_governance_identity"]
    assert param.default is inspect.Parameter.empty


def test_runtime_and_shared_apps_do_not_import_harness_identity_admission() -> None:
    violations: list[str] = []
    for root in (_RUNTIME_ROOT, _SHARED_APPS_ROOT):
        for path in _iter_py_modules(root):
            rel = _rel(path)
            if rel in _IMPORT_ALLOWLIST_REL:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            symbols = _imported_harness_identity_symbols(tree)
            if symbols:
                violations.append(f"{rel}: {sorted(symbols)}")
    assert violations == []
