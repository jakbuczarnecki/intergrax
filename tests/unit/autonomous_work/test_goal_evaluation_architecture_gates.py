# © Artur Czarnecki. All rights reserved.

"""AW-4B — goal evaluation architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.contracts.autonomous_work.references import EvaluationCadenceRef
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_FORBIDDEN_TOKENS = (
    "while True",
    "while  True",
    "CollaborativeWorkItem",
    "WorkItem",
    "openai",
    "anthropic",
    "langchain",
    "runtime.task",
    "runtime.events",
    "agents.",
    "execution_authority_admission",
    "APScheduler",
    "celery",
    "cron",
)


def _module_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    return [
        base / "goal_evaluation_service.py",
        base / "goal_evaluation_ports.py",
    ]


def _contract_path() -> Path:
    module = importlib.import_module("intergrax.contracts.autonomous_work.goal_evaluation")
    assert module.__file__ is not None
    return Path(module.__file__)


def test_goal_evaluation_modules_have_no_scheduler_or_llm_loops() -> None:
    for path in _module_paths():
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        assert "while true" not in lowered, path.name
        for token in _FORBIDDEN_TOKENS:
            if token.lower() in {"cron", "apscheduler", "celery"}:
                assert token.lower() not in lowered, f"{path.name} contains {token}"


def test_goal_evaluation_service_has_no_workitem_or_execution_imports() -> None:
    module = importlib.import_module("intergrax.autonomous_work.goal_evaluation_service")
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
        "collaborative_work",
        "runtime.task",
        "runtime.events",
        "agents.",
        "execution_authority_admission",
        "openai",
        "anthropic",
    ):
        assert token not in joined


def test_goal_evaluation_contract_reuses_worker_goal_refs() -> None:
    module = importlib.import_module("intergrax.contracts.autonomous_work.goal_evaluation")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported)
    assert "intergrax.contracts.autonomous_work.goal" not in joined
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "WorkerGoalId" in source
    assert "GoalEvaluationCadenceState" in source


def test_goal_evaluation_service_requires_finite_batch_bound() -> None:
    module = importlib.import_module("intergrax.autonomous_work.goal_evaluation_service")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "max_goals" in source
    assert "goals_skipped_batch_limit" in source


def test_goal_evaluation_service_requires_mandatory_cadence_state_store() -> None:
    module = importlib.import_module("intergrax.autonomous_work.goal_evaluation_service")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "cadence_state_recorder" not in source
    assert "GoalEvaluationCadenceStateStore" in source
    tree = ast.parse(source)
    init_methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    assert init_methods
    init = init_methods[0]
    arg_names = [arg.arg for arg in init.args.kwonlyargs]
    assert "cadence_state" in arg_names
    assert "cadence_state_recorder" not in arg_names


def test_mapping_cadence_resolver_has_no_implicit_string_parsing() -> None:
    module = importlib.import_module("intergrax.autonomous_work.goal_evaluation_ports")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "parse_cadence_interval_seconds" not in source
    resolver = importlib.import_module(
        "intergrax.autonomous_work.goal_evaluation_ports"
    ).MappingGoalEvaluationCadenceResolver({})
    with pytest.raises(KeyError):
        resolver.resolve(
            cadence_ref=EvaluationCadenceRef("cadence/goal-eval-5m"),
            goal=contract_suite.worker_goal(),
        )


def test_goal_evaluation_ports_have_no_provider_dependencies() -> None:
    module = importlib.import_module("intergrax.autonomous_work.goal_evaluation_ports")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            assert root not in {"psycopg", "sqlalchemy", "asyncpg", "boto3"}
