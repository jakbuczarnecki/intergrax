# © Artur Czarnecki. All rights reserved.

"""HARNESS-02 — cancellation, deadline & cooperative abort qualification gates."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    peek_active_execution_global_deadline_monotonic,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from tests.qualification.harness_02.catalog import (
    HARNESS_02_FINAL_REQUIRED_QUALIFICATION_IDS,
    HARNESS_02_FINDINGS,
    HARNESS_02_PROPAGATION_MATRIX,
    HARNESS_02_R1_QUALIFICATION_MATRIX,
    HARNESS_02_R1_REQUIRED_QUALIFICATION_IDS,
    HARNESS_02_REQUIRED_FLOW_IDS,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_BASE_LLM_ADAPTER = _REPO_ROOT / "intergrax" / "llm_adapters" / "base" / "base_llm_adapter.py"
_LLM_ADAPTER_CONTRACT = _REPO_ROOT / "intergrax" / "llm_adapters" / "contracts" / "llm_adapter.py"
_INTERGRAX = _REPO_ROOT / "intergrax"


def _assert_qualification_evidence_exists(pytest_node_id: str) -> None:
    if "::" not in pytest_node_id:
        raise AssertionError(f"qualification evidence must be a pytest node id: {pytest_node_id}")
    rel_path, test_name = pytest_node_id.split("::", 1)
    path = _REPO_ROOT / rel_path
    assert path.is_file(), f"missing evidence file: {rel_path}"
    source = path.read_text(encoding="utf-8")
    assert f"def {test_name}(" in source, f"missing test function: {test_name} in {rel_path}"


@dataclass(frozen=True)
class _Ping:
    value: str


@dataclass(frozen=True)
class _Pong:
    value: str


def _bind_root_budget_with_deadline(
    *,
    execution_id: object,
    run_id: object,
    attempt_id: object,
    deadline: float,
) -> tuple[object, object]:
    ledger = create_execution_budget_ledger(RunBudget(max_wall_time_seconds=120.0))
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    budget_token = bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=execution_id,
            mode=ExecutionBudgetAllocationMode.SHARED,
            ledger=ledger,
            global_deadline_monotonic=deadline,
        ),
    )
    return identity_token, budget_token


@pytest.mark.asyncio
async def test_harness_02_child_inherits_parent_global_deadline() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_id = mint_execution_id()
    deadline = time.monotonic() + 45.0
    ledger = create_execution_budget_ledger(RunBudget(max_wall_time_seconds=45.0))
    identity_token, budget_token = _bind_root_budget_with_deadline(
        execution_id=root_id,
        run_id=run_id,
        attempt_id=attempt_id,
        deadline=deadline,
    )
    child_runner = ChildExecutionRunner[_Ping, _Pong](ledger=ledger)
    observed: list[float | None] = []

    class ChildDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            observed.append(peek_active_execution_global_deadline_monotonic())
            return _Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            return await child_runner.execute(request=request, delegate=ChildDelegate())

    try:
        await ExecutionBoundary[_Ping, _Pong](
            RootDelegate(),
            identity=ExecutionIdentityBinding(
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=root_id,
            ),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(_Ping(value="child"))
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)

    assert observed == [deadline]


@pytest.mark.asyncio
async def test_harness_02_grandchild_preserves_root_global_deadline() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_id = mint_execution_id()
    deadline = time.monotonic() + 90.0
    ledger = create_execution_budget_ledger(RunBudget(max_wall_time_seconds=90.0))
    identity_token, budget_token = _bind_root_budget_with_deadline(
        execution_id=root_id,
        run_id=run_id,
        attempt_id=attempt_id,
        deadline=deadline,
    )
    child_runner = ChildExecutionRunner[_Ping, _Pong](ledger=ledger)
    observed: list[float | None] = []

    class GrandchildDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            observed.append(peek_active_execution_global_deadline_monotonic())
            return _Pong(value=request.value)

    class ChildDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            observed.append(peek_active_execution_global_deadline_monotonic())
            return await child_runner.execute(
                request=request,
                delegate=GrandchildDelegate(),
            )

    class RootDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            observed.append(peek_active_execution_global_deadline_monotonic())
            return await child_runner.execute(request=request, delegate=ChildDelegate())

    try:
        await ExecutionBoundary[_Ping, _Pong](
            RootDelegate(),
            identity=ExecutionIdentityBinding(
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=root_id,
            ),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(_Ping(value="grandchild"))
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)

    assert observed == [deadline, deadline, deadline]


def test_harness_02_catalog_covers_required_flow_ids() -> None:
    covered = {row.flow_id for row in HARNESS_02_PROPAGATION_MATRIX}
    assert HARNESS_02_REQUIRED_FLOW_IDS <= covered


def test_harness_02_r1_catalog_covers_required_qualification_ids() -> None:
    catalog_ids = {row.qualification_id for row in HARNESS_02_R1_QUALIFICATION_MATRIX}
    assert HARNESS_02_R1_REQUIRED_QUALIFICATION_IDS <= catalog_ids
    for row in HARNESS_02_R1_QUALIFICATION_MATRIX:
        if row.qualification_id == "Q15":
            assert row.status == "PASS"
        elif row.qualification_id in HARNESS_02_R1_REQUIRED_QUALIFICATION_IDS:
            assert row.status == "PASS"


def test_harness_02_global_deadline_monotonic_consumer_allowlist() -> None:
    """Documented authority: mint/bind, child inherit, retry eligibility, graph_runner wiring."""
    allowed_suffixes = (
        "runtime/execution/active_execution_budget.py",
        "runtime/execution/child.py",
        "runtime/execution/retry/policy.py",
        "runtime/nexus/orchestration/graph_runner.py",
        "contracts/execution_retry.py",
    )
    offenders: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith("intergrax/") is False:
            continue
        text = path.read_text(encoding="utf-8")
        if "peek_active_execution_global_deadline_monotonic" not in text:
            continue
        if not any(rel.endswith(suffix) for suffix in allowed_suffixes):
            offenders.append(rel)
    assert offenders == []


def test_harness_02_tool_invoker_cancellation_before_first_attempt() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_INVOKER))
    policy_methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_with_policy"
    ]
    assert policy_methods, "_execute_with_policy must exist"
    for_node = next(
        stmt for stmt in policy_methods[0].body if isinstance(stmt, ast.For)
    )
    first_stmt = for_node.body[0]
    assert isinstance(first_stmt, ast.If)
    test_src = ast.get_source_segment(source, first_stmt.test) or ""
    assert "_cooperative_cancellation_requested" in test_src


def _llm_provider_guard_call_lines(func: ast.FunctionDef) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "assert_protected_provider_call_allowed":
            lines.append(node.lineno)
    return lines


def _llm_provider_io_dispatch_lines(func: ast.FunctionDef) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "execute_with_resilience":
            lines.append(node.lineno)
    return lines


def _llm_execute_methods_guard_precedes_provider_io(source_path: Path) -> bool:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "BaseLLMAdapter":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name not in {"_execute", "_execute_streaming"}:
                continue
            guard_lines = _llm_provider_guard_call_lines(item)
            io_lines = _llm_provider_io_dispatch_lines(item)
            if not guard_lines or not io_lines:
                return False
            if min(guard_lines) >= min(io_lines):
                return False
    return True


def test_harness_02_llm_framework_guard_precedes_provider_io() -> None:
    assert _BASE_LLM_ADAPTER.is_file()
    assert _llm_execute_methods_guard_precedes_provider_io(_BASE_LLM_ADAPTER)
    contract_text = _LLM_ADAPTER_CONTRACT.read_text(encoding="utf-8")
    assert "peek_active_execution_global_deadline" not in contract_text


def test_harness_02_llm_docstring_only_guard_string_is_not_evidence() -> None:
    """Docstring mention of the guard must not satisfy AST execution-path evidence."""
    docstring_only = '''
"""Runs assert_protected_provider_call_allowed in documentation only."""

class BaseLLMAdapter:
    def _execute(self, fn):
        return execute_with_resilience(fn)
'''
    tree = ast.parse(docstring_only)
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    execute_method = next(
        n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "_execute"
    )
    assert _llm_provider_guard_call_lines(execute_method) == []
    assert _llm_provider_io_dispatch_lines(execute_method)
    assert not _llm_execute_methods_guard_precedes_provider_io_from_tree(tree)


def _llm_execute_methods_guard_precedes_provider_io_from_tree(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "BaseLLMAdapter":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name not in {"_execute", "_execute_streaming"}:
                continue
            guard_lines = _llm_provider_guard_call_lines(item)
            io_lines = _llm_provider_io_dispatch_lines(item)
            if not guard_lines or not io_lines:
                return False
            if min(guard_lines) >= min(io_lines):
                return False
    return True


def test_harness_02_llm_provider_guard_blocks_physical_io_when_denied() -> None:
    from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
    from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
    from intergrax.runtime.execution.deadline_provider_guard import (
        ExecutionProtectedWorkDeniedError,
    )
    from intergrax.runtime.execution.deadline_scope import (
        bind_active_execution_deadline_scope,
        reset_active_execution_deadline_scope,
    )
    from intergrax.runtime.execution.protected_work_admission import (
        CanonicalHardProtectedWorkAdmission,
        StaticCancellationView,
    )
    from tests.unit.runtime.execution.deadline_authority.test_harness_02_r1_qualification import (
        _FakeMonotonicClock,
    )

    monotonic = _FakeMonotonicClock(1.0)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        remaining_seconds=0.0,
        is_expired=True,
        global_deadline_monotonic=1.0,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    physical_calls = 0

    class _ProbeAdapter(BaseLLMAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.provider = "openai"

        @property
        def context_window_tokens(self) -> int:
            return 8192

        def generate_messages(self, messages):  # type: ignore[no-untyped-def]
            del messages
            return self._execute(lambda: "ok")

    adapter = _ProbeAdapter()

    def _physical() -> str:
        nonlocal physical_calls
        physical_calls += 1
        return "ok"

    try:
        with pytest.raises(ExecutionProtectedWorkDeniedError):
            adapter._execute(_physical)
        assert physical_calls == 0
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_harness_02_llm_provider_guard_allows_single_physical_io_when_admitted() -> None:
    from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
    from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
    from intergrax.runtime.execution.deadline_scope import (
        bind_active_execution_deadline_scope,
        reset_active_execution_deadline_scope,
    )
    from intergrax.runtime.execution.protected_work_admission import (
        CanonicalHardProtectedWorkAdmission,
        StaticCancellationView,
    )
    from tests.unit.runtime.execution.deadline_authority.test_harness_02_r1_qualification import (
        _FakeMonotonicClock,
    )

    monotonic = _FakeMonotonicClock(100.0)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
        remaining_seconds=30.0,
        is_expired=False,
        global_deadline_monotonic=130.0,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    physical_calls = 0

    class _ProbeAdapter(BaseLLMAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.provider = "openai"

        @property
        def context_window_tokens(self) -> int:
            return 8192

        def generate_messages(self, messages):  # type: ignore[no-untyped-def]
            del messages
            return self._execute(lambda: "ok")

    adapter = _ProbeAdapter()

    def _physical() -> str:
        nonlocal physical_calls
        physical_calls += 1
        return "ok"

    try:
        assert adapter._execute(_physical) == "ok"
        assert physical_calls == 1
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_harness_02_findings_have_no_unresolved_blockers() -> None:
    blockers = [row for row in HARNESS_02_FINDINGS if row.severity == "BLOCKER"]
    assert blockers == []


def test_harness_02_final_closure_gate() -> None:
    """Aggregate qualification evidence for HARNESS-02 closure (no GAP flows, all Q PASS)."""
    gap_flows = [
        row.flow_id for row in HARNESS_02_PROPAGATION_MATRIX if row.status == "GAP"
    ]
    assert gap_flows == []
    blockers = [row for row in HARNESS_02_FINDINGS if row.severity == "BLOCKER"]
    assert blockers == []

    required = HARNESS_02_FINAL_REQUIRED_QUALIFICATION_IDS
    matrix_ids = [row.qualification_id for row in HARNESS_02_R1_QUALIFICATION_MATRIX]
    matrix_id_set = frozenset(matrix_ids)
    assert len(matrix_ids) == len(required)
    assert matrix_id_set == required
    assert len(matrix_ids) == len(set(matrix_ids))

    for row in HARNESS_02_R1_QUALIFICATION_MATRIX:
        assert row.qualification_id in required
        assert row.status == "PASS"
        _assert_qualification_evidence_exists(row.evidence)

    assert any(row.qualification_id == "Q15" and row.status == "PASS" for row in HARNESS_02_R1_QUALIFICATION_MATRIX)
    assert any(row.qualification_id == "Q28" and row.status == "PASS" for row in HARNESS_02_R1_QUALIFICATION_MATRIX)
    assert any(row.qualification_id == "Q29" and row.status == "PASS" for row in HARNESS_02_R1_QUALIFICATION_MATRIX)
