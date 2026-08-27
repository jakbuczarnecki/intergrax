# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from intergrax.runtime.execution import (
    ExecutionResult,
    ExecutionStatus,
    __all__ as execution_public_api,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_PUBLIC_FIELD_NAMES = frozenset(
    {
        "strategy",
        "mode",
        "executor",
        "agent",
        "metadata",
        "options",
        "config",
        "annex",
        "extra",
        "details",
        "payload",
        "error",
        "errors",
        "exception",
        "failure_reason",
        "message",
        "task_id",
        "run_id",
        "attempt_id",
        "execution_id",
        "event_id",
        "parent_execution_id",
        "artifacts",
        "evidence",
        "warnings",
        "cost",
        "duration",
        "used_tools",
        "confidence",
        "agent_decision",
        "human_request",
        "policy_rule_id",
        "response",
    }
)

_FORBIDDEN_DYNAMIC_TOKENS = frozenset(
    {
        "Any",
        "dict[",
        "Mapping[",
        "MutableMapping[",
        "getattr",
        "setattr",
        "hasattr",
        "__getattr__",
        "__dict__",
        "vars(",
        "inspect",
        "importlib",
        "isinstance(",
        "issubclass(",
        "callable(",
        "**kwargs",
    }
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.task",
    "intergrax.agents",
    "intergrax.runtime.nexus",
    "intergrax.llm_adapters",
    "intergrax.runtime.governance",
    "intergrax.runtime.observability",
    "intergrax.runtime.diagnostics",
)


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk: str


def test_execution_status_completed_value() -> None:
    assert ExecutionStatus.COMPLETED.value == "completed"


def test_execution_result_preserves_typed_output() -> None:
    expected = RiskAssessment(risk="low")

    result = ExecutionResult(
        status=ExecutionStatus.COMPLETED,
        output=expected,
    )

    assert result.status is ExecutionStatus.COMPLETED
    assert result.output is expected
    assert result.output.risk == "low"


def test_execution_result_has_only_status_and_output_fields() -> None:
    public_fields = {field.name for field in fields(ExecutionResult)}

    assert public_fields == frozenset({"status", "output"})
    assert public_fields.isdisjoint(_FORBIDDEN_PUBLIC_FIELD_NAMES)


def test_execution_result_is_frozen_dataclass_with_slots() -> None:
    params = ExecutionResult.__dataclass_params__

    assert params.frozen is True
    assert ExecutionResult.__slots__ == ("status", "output")


def test_execution_result_source_has_no_forbidden_dynamic_mechanisms() -> None:
    source = Path("intergrax/runtime/execution/result.py").read_text(encoding="utf-8")
    for token in _FORBIDDEN_DYNAMIC_TOKENS:
        assert token not in source, f"forbidden dynamic token in result.py: {token}"


def test_execution_result_module_has_no_forbidden_imports() -> None:
    result_path = Path("intergrax/runtime/execution/result.py")
    module = ast.parse(result_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
        assert not any(
            name == forbidden or name.startswith(f"{forbidden}.") for name in imported
        )


def test_package_root_exports_execution_result_symbols() -> None:
    from intergrax.runtime.execution import ExecutionResult as ExportedResult
    from intergrax.runtime.execution import ExecutionStatus as ExportedStatus

    assert ExportedResult is ExecutionResult
    assert ExportedStatus is ExecutionStatus
    assert "ExecutionResult" in execution_public_api
    assert "ExecutionStatus" in execution_public_api


def test_inference_executor_not_exported_from_package_root() -> None:
    import intergrax.runtime.execution as execution_package

    assert "InferenceExecutor" not in execution_package.__all__


def test_strategy_resolver_not_exported_from_package_root() -> None:
    import intergrax.runtime.execution as execution_package

    assert "StrategyResolver" not in execution_package.__all__
    assert "ExecutionStrategy" not in execution_package.__all__
