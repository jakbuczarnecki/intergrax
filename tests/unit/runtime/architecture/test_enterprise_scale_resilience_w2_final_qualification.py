# © Artur Czarnecki. All rights reserved.

"""W2 Final — dependency resilience qualification gates (flow + orthogonality)."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_BASE_LLM_ADAPTER = "intergrax/llm_adapters/base/base_llm_adapter.py"
_LLM_ADAPTER_CONTRACT = "intergrax/llm_adapters/contracts/llm_adapter.py"
_RESILIENCE = "intergrax/llm_adapters/_shared/resilience.py"


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def _parse_module(rel: str) -> ast.Module:
    path = _REPO_ROOT / rel
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_def(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} missing in {module.body!r}")


def _method_def(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} missing on {cls.name}")


def _module_function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} missing in module")


def _call_attr_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _first_call_lineno(
    root: ast.AST,
    *,
    name: str,
    receiver_attr: str | None = None,
) -> int:
    for node in ast.walk(root):
        if not isinstance(node, ast.Call):
            continue
        if receiver_attr is None:
            if _call_attr_name(node.func) == name:
                return node.lineno
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == name:
            recv = node.func.value
            if isinstance(recv, ast.Name) and recv.id == receiver_attr:
                return node.lineno
    raise AssertionError(f"call {name!r} not found under {type(root).__name__}")


def _nested_function(method: ast.FunctionDef, name: str) -> ast.FunctionDef:
    for node in ast.walk(method):
        if isinstance(node, ast.FunctionDef) and node.name == name and node is not method:
            return node
    raise AssertionError(f"nested function {name!r} missing in {method.name}")


def _is_boundary_none_guard(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Is):
        return False
    left = test.left
    if not isinstance(left, ast.Name) or left.id != "boundary":
        return False
    comparators = test.comparators
    return len(comparators) == 1 and isinstance(comparators[0], ast.Constant) and comparators[0].value is None


def _fn_call_nodes(root: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(root):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fn":
            calls.append(node)
    return calls


def _node_within(parent: ast.AST, child: ast.AST) -> bool:
    for node in ast.walk(parent):
        if node is child:
            return True
    return False


def _assert_provider_boundary_resolution(method: ast.FunctionDef) -> None:
    for stmt in method.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "boundary" for t in stmt.targets):
            continue
        value = stmt.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "_provider_dependency_boundary"
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        ):
            return
    raise AssertionError(
        "BaseLLMAdapter._run_physical_provider_attempt must resolve "
        "boundary = self._provider_dependency_boundary",
    )


def _assert_boundary_enabled_admission(method: ast.FunctionDef) -> None:
    none_guard: ast.If | None = None
    for stmt in method.body:
        if isinstance(stmt, ast.If) and _is_boundary_none_guard(stmt.test):
            none_guard = stmt
            break
    assert none_guard is not None, (
        "BaseLLMAdapter._run_physical_provider_attempt must branch on boundary is None"
    )

    absent_path_fn_calls = _fn_call_nodes(ast.Module(body=none_guard.body, type_ignores=[]))
    assert absent_path_fn_calls, (
        "boundary-absent path must invoke physical fn() directly"
    )

    acquire_lineno = _first_call_lineno(method, name="acquire", receiver_attr="boundary")
    enabled_fn_calls = [
        call
        for call in _fn_call_nodes(method)
        if not _node_within(none_guard, call)
    ]
    assert enabled_fn_calls, (
        "boundary-enabled path must invoke physical fn() after admission"
    )
    for call in enabled_fn_calls:
        assert acquire_lineno < call.lineno, (
            "boundary.acquire must precede physical fn() on boundary-enabled path"
        )

    _first_call_lineno(method, name="complete_direct", receiver_attr="boundary")


def test_llm_final_flow_tenant_then_resilience_then_admission_inside_physical() -> None:
    """Required order: tenant quota → retry budget → rate → CB → admission → SDK."""
    contract_module = _parse_module(_LLM_ADAPTER_CONTRACT)
    for node in ast.walk(contract_module):
        if isinstance(node, ast.FunctionDef) and node.name == "_execute":
            raise AssertionError(
                "LLMAdapter contract must not own concrete _execute; "
                "see BaseLLMAdapter",
            )

    base_module = _parse_module(_BASE_LLM_ADAPTER)
    try:
        base_cls = _class_def(base_module, "BaseLLMAdapter")
    except AssertionError as exc:
        raise AssertionError("BaseLLMAdapter missing") from exc

    try:
        execute_method = _method_def(base_cls, "_execute")
    except AssertionError as exc:
        raise AssertionError("_execute missing on BaseLLMAdapter") from exc

    try:
        physical_method = _method_def(base_cls, "_run_physical_provider_attempt")
    except AssertionError as exc:
        raise AssertionError("_run_physical_provider_attempt missing on BaseLLMAdapter") from exc

    quota_lineno = _first_call_lineno(execute_method, name="check_llm_tenant_quota")
    resilience_lineno = _first_call_lineno(execute_method, name="execute_with_resilience")
    assert quota_lineno < resilience_lineno, (
        "check_llm_tenant_quota must occur before execute_with_resilience in BaseLLMAdapter._execute"
    )

    physical_attempt = _nested_function(execute_method, "physical_attempt")
    _first_call_lineno(physical_attempt, name="_run_physical_provider_attempt")

    resilience_call: ast.Call | None = None
    for node in ast.walk(execute_method):
        if isinstance(node, ast.Call) and _call_attr_name(node.func) == "execute_with_resilience":
            resilience_call = node
            break
    assert resilience_call is not None
    assert resilience_call.args, "execute_with_resilience must receive physical_attempt callback"
    first_arg = resilience_call.args[0]
    assert isinstance(first_arg, ast.Name) and first_arg.id == "physical_attempt", (
        "execute_with_resilience must be wired to local physical_attempt callback"
    )

    _assert_provider_boundary_resolution(physical_method)
    _assert_boundary_enabled_admission(physical_method)

    resilience_module = _parse_module(_RESILIENCE)
    resilience_fn = _module_function(resilience_module, "execute_with_resilience")

    order_checks: list[tuple[str, str, Callable[[ast.Call], bool]]] = [
        (
            "budget.begin_physical_attempt",
            "begin_physical_attempt",
            lambda call: isinstance(call.func, ast.Attribute)
            and call.func.attr == "begin_physical_attempt"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "budget",
        ),
        (
            "_check_distributed_rate_limit",
            "_check_distributed_rate_limit",
            lambda call: _call_attr_name(call.func) == "_check_distributed_rate_limit",
        ),
        (
            "_acquire_local_rate_limit",
            "_acquire_local_rate_limit",
            lambda call: _call_attr_name(call.func) == "_acquire_local_rate_limit",
        ),
        (
            "_check_circuit",
            "_check_circuit",
            lambda call: _call_attr_name(call.func) == "_check_circuit",
        ),
        (
            "fn()",
            "fn",
            lambda call: isinstance(call.func, ast.Name) and call.func.id == "fn",
        ),
    ]

    linos: list[int] = []
    for label, _, predicate in order_checks:
        found: int | None = None
        for node in ast.walk(resilience_fn):
            if isinstance(node, ast.Call) and predicate(node):
                found = node.lineno
                break
        assert found is not None, f"{label} missing in execute_with_resilience"
        linos.append(found)

    for earlier, later in zip(linos[:-1], linos[1:], strict=True):
        assert earlier < later, (
            "execute_with_resilience per-attempt ordering violated "
            f"(line {earlier} must precede line {later})"
        )


def test_w2_local_implementations_do_not_import_orchestration_layers() -> None:
    """ETAP 4 — module-level orthogonality (contracts + local ports only)."""
    retry_budget = _read("intergrax/runtime/resilience/local_provider_retry_budget.py")
    rate_limit = _read("intergrax/runtime/resilience/local_provider_rate_limit.py")
    admission = _read("intergrax/runtime/resilience/local_dependency_concurrency_admission.py")
    boundary = _read("intergrax/runtime/resilience/dependency_attempt_execution_boundary.py")

    forbidden_in_retry_budget = (
        "circuit",
        "rate_limit",
        "resilience",
        "dependency_admission",
        "llm_adapters",
    )
    for token in forbidden_in_retry_budget:
        assert token not in retry_budget

    forbidden_in_rate_limit = ("retry_budget", "circuit", "admission", "resilience", "llm_adapters")
    for token in forbidden_in_rate_limit:
        assert token not in rate_limit

    forbidden_in_admission = ("retry_budget", "provider_rate_limit", "circuit", "execute_with_resilience")
    for token in forbidden_in_admission:
        assert token not in admission

    forbidden_in_boundary = ("retry_budget", "provider_rate_limit", "execute_with_resilience", "resilience.py")
    for token in forbidden_in_boundary:
        assert token not in boundary


def test_qualification_matrix_modules_exist() -> None:
    """Scenarios 1–8 map to existing behavioral suites (no duplicate matrix tests)."""
    expected = (
        "tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py",
        "tests/unit/llm_adapters/test_llm_provider_dependency_admission.py",
        "tests/unit/runtime/resilience/test_dependency_attempt_execution_boundary.py",
        "tests/unit/runtime/resilience/test_local_dependency_concurrency_admission.py",
        "tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py",
    )
    for rel in expected:
        assert (_REPO_ROOT / rel).is_file(), rel
