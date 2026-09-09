# © Artur Czarnecki. All rights reserved.

"""NPSC-5C — coordination intent architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NPSC5C_MODULES = (
    _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent.py",
    _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py",
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus",
    "intergrax.supervisor",
)

_FORBIDDEN_RUNTIME_CALLS = (
    "NexusLoop",
    "GraphExecutor",
    "ChildExecutionRunner",
    "DelegatedSubtaskService",
    "OrchestrationTopologySubmissionPort",
)

_REQUIRED_ROUTING_CALLS = (
    ("coordination_intent_executor.py", "MultiAgentCoordinationService"),
    ("coordination_intent_executor.py", "BoundedMultiAgentFanOutService"),
)

_FORBIDDEN_PROHIBITED_PATTERNS = (
    re.compile(r"\bAny\b"),
    re.compile(r"dict\[str,\s*Any\]"),
    re.compile(r"Dict\[str,\s*Any\]"),
    re.compile(r"Callable\[\.\.\."),
    re.compile(r"\bcast\("),
    re.compile(r"#\s*type:\s*ignore"),
    re.compile(r"except\s+Exception\b"),
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _source_calls(path: Path, names: tuple[str, ...]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in names:
                hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in names:
                hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}")
    return hits


@pytest.mark.gate
def test_npsc5c_modules_exist() -> None:
    missing = [path for path in _NPSC5C_MODULES if not path.is_file()]
    assert missing == [], f"missing NPSC-5C modules: {missing}"


@pytest.mark.gate
def test_npsc5c_no_forbidden_runtime_imports() -> None:
    violations: list[str] = []
    for path in _NPSC5C_MODULES:
        modules = _imported_modules(path)
        for module in sorted(modules):
            if any(module.startswith(prefix) for prefix in _FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.name}: {module}")
    assert violations == [], (
        "NPSC-5C modules must not import Nexus or legacy supervisor:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5c_no_direct_runtime_execution_dependencies() -> None:
    executor_path = _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py"
    violations = _source_calls(executor_path, _FORBIDDEN_RUNTIME_CALLS)
    assert violations == [], (
        "CoordinationIntentExecutor must not call forbidden runtime owners:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5c_routes_through_frozen_npsc_services() -> None:
    executor_path = _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py"
    source = executor_path.read_text(encoding="utf-8")
    assert "await self._coordination.coordinate" in source
    assert "await self._fan_out.fan_out" in source
    for filename, symbol in _REQUIRED_ROUTING_CALLS:
        assert symbol in source, f"missing required routing symbol {symbol} in {filename}"


@pytest.mark.gate
def test_npsc5c_coordination_contribution_has_no_physical_agent_fields() -> None:
    intent_path = _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent.py"
    source = intent_path.read_text(encoding="utf-8")
    forbidden_fields = (
        "agent_id",
        "agent_instance_id",
        "lease_id",
    )
    for field in forbidden_fields:
        assert f"{field}:" not in source, (
            f"coordination intent contract must not expose physical agent field {field}"
        )


@pytest.mark.gate
def test_npsc5c_no_prohibited_patterns() -> None:
    violations: list[str] = []
    for path in _NPSC5C_MODULES:
        source = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_PROHIBITED_PATTERNS:
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{path.relative_to(_REPO_ROOT).as_posix()}:{line}:{match.group()}",
                )
    assert violations == [], (
        "NPSC-5C production modules contain prohibited patterns:\n"
        + "\n".join(violations)
    )


def _dataclass_field_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.add(item.target.id)
            return fields
    raise AssertionError(f"class not found: {class_name} in {path}")


def _zip_pairs_contributions_with_bindings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "zip"
        ):
            continue
        arg_names = {
            expr.id
            for expr in node.args
            if isinstance(expr, ast.Attribute)
        }
        if (
            "contributions" in arg_names
            and "contribution_bindings" in arg_names
        ):
            violations.append(
                f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}",
            )
    return violations


@pytest.mark.gate
def test_npsc5c_contribution_binding_requires_contribution_id() -> None:
    executor_path = (
        _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py"
    )
    fields = _dataclass_field_names(executor_path, "CoordinationContributionBinding")
    assert "contribution_id" in fields, (
        "CoordinationContributionBinding must declare contribution_id"
    )


def _task_kind_fake_capability_mapping_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "TaskKind"
        ):
            continue
        if not node.args:
            continue
        argument = node.args[0]
        if not isinstance(argument, ast.Name):
            continue
        if argument.id in {"capability_id", "decision_capability_id"}:
            lineno = getattr(node, "lineno", 0)
            violations.append(
                f"{path.relative_to(_REPO_ROOT).as_posix()}:{lineno}",
            )
    return violations


@pytest.mark.gate
def test_npsc5c_no_fake_task_kind_capability_mapping() -> None:
    intent_path = _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent.py"
    executor_path = (
        _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py"
    )
    violations = [
        *(_task_kind_fake_capability_mapping_calls(intent_path)),
        *(_task_kind_fake_capability_mapping_calls(executor_path)),
    ]
    assert violations == [], (
        "NPSC-5C modules must not map capability_id to TaskKind:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5c_executor_does_not_zip_bindings_by_position() -> None:
    executor_path = (
        _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py"
    )
    violations = _zip_pairs_contributions_with_bindings(executor_path)
    assert violations == [], (
        "CoordinationIntentExecutor must not zip intent.contributions with "
        "binding.contribution_bindings for semantic association:\n"
        + "\n".join(violations)
    )
