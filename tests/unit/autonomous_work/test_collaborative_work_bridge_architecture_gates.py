# © Artur Czarnecki. All rights reserved.

"""AW-4C — collaborative work bridge architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_TOKENS = (
    "class WorkItem",
    "class Assignment",
    "WorkerWorkItem",
    "AutonomousWorkItem",
    "WorkerAssignment",
    "AutonomousAssignment",
    "WorkItemRepository",
    "AssignmentRepository",
    "openai",
    "anthropic",
    "langchain",
    "runtime.task",
    "runtime.events",
    "agents.",
    "APScheduler",
    "celery",
)

_FORBIDDEN_IMPORT_ROOTS = (
    "sqlalchemy",
    "redis",
    "boto3",
    "psycopg",
    "asyncpg",
    "pymongo",
)


def _bridge_module_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    return [
        base / "worker_collaborative_work_bridge.py",
        base / "collaborative_work_intake.py",
    ]


def _contract_path() -> Path:
    module = importlib.import_module(
        "intergrax.contracts.autonomous_work.collaborative_work_bridge"
    )
    assert module.__file__ is not None
    return Path(module.__file__)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_bridge_modules_have_no_workitem_or_execution_imports() -> None:
    for path in _bridge_module_paths():
        joined = "\n".join(_imported_modules(path)).lower()
        for token in (
            "runtime.task",
            "runtime.events",
            "runtime.nexus",
            "agents.",
            "execution",
            "openai",
            "anthropic",
        ):
            assert token not in joined, f"{path.name} imports {token}"


def test_bridge_modules_have_no_forbidden_tokens() -> None:
    for path in _bridge_module_paths():
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        assert "while true" not in lowered, path.name
        for token in _FORBIDDEN_TOKENS:
            if token.lower() in {"apscheduler", "celery"}:
                assert token.lower() not in lowered, f"{path.name} contains {token}"


def test_bridge_modules_have_no_provider_dependencies() -> None:
    for path in _bridge_module_paths():
        for module_name in _imported_modules(path):
            root = module_name.split(".", 1)[0]
            assert root not in _FORBIDDEN_IMPORT_ROOTS, (
                f"{path.name} imports provider {module_name}"
            )


def test_contract_does_not_define_workitem_or_assignment() -> None:
    source = _contract_path().read_text(encoding="utf-8")
    for token in (
        "class WorkItem",
        "class Assignment",
        "WorkItemRepository",
        "AssignmentRepository",
    ):
        assert token not in source


def test_autonomous_work_package_does_not_define_canonical_workitem() -> None:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    for path in base.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "class WorkItem" not in source, path.name
        assert "class Assignment" not in source, path.name


def test_bridge_service_does_not_import_collaborative_work_domain() -> None:
    module = importlib.import_module(
        "intergrax.autonomous_work.worker_collaborative_work_bridge"
    )
    assert module.__file__ is not None
    joined = "\n".join(_imported_modules(Path(module.__file__)))
    assert "intergrax.collaborative_work" not in joined
