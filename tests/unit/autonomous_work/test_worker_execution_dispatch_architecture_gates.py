# © Artur Czarnecki. All rights reserved.

"""AW-5A — worker execution dispatch architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_AW_TOKENS = (
    "WorkerExecutionRuntime",
    "AutonomousExecutionRuntime",
    "WorkerTaskRunner",
    "WorkerRunEngine",
    "WorkerExecutionEngine",
    "WorkerBudgetEngine",
    "WorkerTokenLedger",
    "WorkerCostManager",
    "WorkerExecutionEventStore",
    "WorkerExecutionAuditDB",
    "openai",
    "anthropic",
    "langchain",
    "APScheduler",
    "celery",
    "while True",
)

_AW_MODULE_PATHS = (
    "worker_execution_dispatch.py",
)


def _aw_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    return [base / name for name in _AW_MODULE_PATHS]


def test_aw5a_modules_forbidden_runtime_duplication() -> None:
    for path in _aw_paths():
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_AW_TOKENS:
            if token == "while True":
                assert "while true" not in source.lower(), path.name
            else:
                assert token not in source, f"{path.name} contains {token}"


def test_aw5a_dispatch_has_no_recovery_loop() -> None:
    module = importlib.import_module("intergrax.autonomous_work.worker_execution_dispatch")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            raise AssertionError("AW-5A dispatch must not implement recovery/retry loops")


def test_aw5a_dispatch_does_not_import_concrete_strategy() -> None:
    module = importlib.import_module("intergrax.autonomous_work.worker_execution_dispatch")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported).lower()
    for token in (
        "langgraph",
        "nexus.graph_executor",
        "agents.",
        "llm",
        "openai",
    ):
        assert token not in joined


def test_aw5a_contract_module_has_no_runtime_task_duplication() -> None:
    module = importlib.import_module("intergrax.contracts.autonomous_work.execution_dispatch")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    for token in ("WorkerTask", "AutonomousTask", "WorkerRun", "WorkerAttempt"):
        assert token not in source


def test_aw5a_root_admission_does_not_trust_collaborative_allow_alone() -> None:
    module = importlib.import_module(
        "intergrax.runtime.governance.root_execution_authority_admission",
    )
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_runtime_policy_admission.evaluate" in source
    assert source.index("_runtime_policy_admission.evaluate") < source.index(
        "ParentExecutionAuthority.scoped"
    )
