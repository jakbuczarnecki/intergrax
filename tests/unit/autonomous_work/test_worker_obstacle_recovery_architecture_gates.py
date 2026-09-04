# © Artur Czarnecki. All rights reserved.

"""AW-6A — obstacle recovery architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_AW6A_MODULE_PATHS = (
    "worker_obstacle_classifier.py",
    "worker_recovery_decision_service.py",
    "obstacle_recovery_ports.py",
)

_FORBIDDEN_IMPORTS = (
    "intergrax.runtime.execution",
    "ExecutionRuntime",
    "CanonicalExecutionIntakePort",
    "ParentExecutionAuthority",
    "RetryEngine",
    "RetryCoordinator",
    "codecraft",
    "CodeCraft",
    "sqlalchemy",
    "psycopg",
    "openai",
    "anthropic",
)

_FORBIDDEN_TOKENS = (
    "while True",
    "execute(",
)


def _aw6a_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    return [base / name for name in _AW6A_MODULE_PATHS]


def test_aw6a_modules_forbidden_imports() -> None:
    for path in _aw6a_paths():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        joined = "\n".join(imported)
        for token in _FORBIDDEN_IMPORTS:
            assert token.lower() not in joined.lower(), f"{path.name} imports {token}"


def test_aw6a_modules_forbidden_tokens() -> None:
    for path in _aw6a_paths():
        source = path.read_text(encoding="utf-8").lower()
        for token in _FORBIDDEN_TOKENS:
            if token == "while True":
                assert "while true" not in source, path.name
            else:
                assert token.lower() not in source, f"{path.name} contains {token}"


def test_aw6a_contracts_no_execution_runtime_import() -> None:
    module = importlib.import_module("intergrax.contracts.autonomous_work.obstacle_recovery")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "ExecutionRuntime" not in source
    assert "RetryEngine" not in source


def test_aw6a_service_has_no_retry_loop() -> None:
    module = importlib.import_module(
        "intergrax.autonomous_work.worker_recovery_decision_service",
    )
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            raise AssertionError("AW-6A must not implement retry loops")
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            func = node.iter.func
            if isinstance(func, ast.Name) and func.id == "range":
                raise AssertionError("AW-6A must not implement retry loops")


def test_aw6a_public_import_smoke() -> None:
    from intergrax.autonomous_work.worker_recovery_decision_service import (
        WorkerRecoveryDecisionService,
    )
    from intergrax.contracts.autonomous_work.obstacle_recovery import (
        RecoveryStrategy,
        WorkerObstacleKind,
    )

    assert WorkerRecoveryDecisionService is not None
    assert WorkerObstacleKind.POLICY_DENIED.value == "POLICY_DENIED"
    assert RecoveryStrategy.STOP.value == "STOP"
