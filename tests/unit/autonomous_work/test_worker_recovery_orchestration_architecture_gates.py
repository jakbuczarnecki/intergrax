# © Artur Czarnecki. All rights reserved.

"""AW-6B — recovery orchestration architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_AW6B_MODULE_PATHS = (
    "worker_recovery_orchestration_service.py",
    "recovery_orchestration_ports.py",
    "recovery_episode_claim.py",
    "in_memory_recovery_episode_repository.py",
    "recovery_episode_serialization.py",
    "postgresql_recovery_episode_repository.py",
)

_FORBIDDEN_IMPORTS = (
    "ExecutionRuntime",
    "RetryEngine",
    "RetryCoordinator",
    "ParentExecutionAuthority",
    "codecraft",
    "CodeCraft",
    "sqlalchemy",
    "psycopg",
    "time.sleep",
    "asyncio.sleep",
)

_FORBIDDEN_NAMES = (
    "RecoveryRuntime",
    "RecoveryExecutionEngine",
    "WorkerRecoveryExecutorLoop",
    "RecoveryManager",
    "RecoveryEngine",
    "RecoveryCoordinatorFactory",
)


def _aw6b_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    return [base / name for name in _AW6B_MODULE_PATHS]


def test_aw6b_modules_forbidden_imports() -> None:
    for path in _aw6b_paths():
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


def test_aw6b_service_forbidden_names_and_sleep() -> None:
    path = importlib.import_module(
        "intergrax.autonomous_work.worker_recovery_orchestration_service",
    ).__file__
    assert path is not None
    source = Path(path).read_text(encoding="utf-8")
    for name in _FORBIDDEN_NAMES:
        assert name not in source
    assert "sleep(" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            raise AssertionError("AW-6B must not implement retry loops")


def test_aw6b_contracts_no_execution_runtime_import() -> None:
    module = importlib.import_module(
        "intergrax.contracts.autonomous_work.recovery_orchestration",
    )
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "ExecutionRuntime" not in source
    assert "RetryEngine" not in source


def test_aw6b_public_import_smoke() -> None:
    from intergrax.autonomous_work.worker_recovery_orchestration_service import (
        WorkerRecoveryOrchestrationService,
    )
    from intergrax.contracts.autonomous_work.recovery_orchestration import (
        RecoveryEpisodeStatus,
        WorkerRecoveryOrchestrationDisposition,
    )

    assert WorkerRecoveryOrchestrationService is not None
    assert RecoveryEpisodeStatus.PENDING.value == "PENDING"
    assert WorkerRecoveryOrchestrationDisposition.RESUMED.value == "RESUMED"
