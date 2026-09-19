# © Artur Czarnecki. All rights reserved.

"""HARNESS-02 — cancellation, deadline & cooperative abort qualification gates."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
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
    HARNESS_02_FINDINGS,
    HARNESS_02_PROPAGATION_MATRIX,
    HARNESS_02_REQUIRED_FLOW_IDS,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_INTERGRAX = _REPO_ROOT / "intergrax"


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


def test_harness_02_tool_invoker_cancellation_not_before_first_attempt() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_INVOKER))
    policy_methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_with_policy"
    ]
    assert policy_methods, "_execute_with_policy must exist"
    body = policy_methods[0].body
    cancel_before_first = False
    for stmt in body:
        if isinstance(stmt, ast.For):
            for inner in ast.walk(stmt):
                if (
                    isinstance(inner, ast.If)
                    and isinstance(inner.test, ast.Compare)
                    and any(
                        isinstance(op, ast.Gt)
                        for op in inner.test.ops
                    )
                ):
                    for sub in ast.walk(inner):
                        if (
                            isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "_cooperative_cancellation_requested"
                        ):
                            cancel_before_first = True
    assert cancel_before_first, "cancellation guard must exist for retry attempts"
    assert "if attempt > 1:" in source or "attempt > 1" in source


def test_harness_02_global_deadline_not_wired_to_llm_path() -> None:
    llm_shared = _REPO_ROOT / "intergrax" / "llm_adapters" / "_shared"
    hits: list[str] = []
    for path in llm_shared.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "peek_active_execution_global_deadline" in text:
            hits.append(path.relative_to(_REPO_ROOT).as_posix())
    assert hits == []


def test_harness_02_findings_include_blockers_for_enforcement_gaps() -> None:
    blocker_flows = {
        row.flow
        for row in HARNESS_02_FINDINGS
        if row.severity == "BLOCKER"
    }
    assert "H02-tool-pre-effect" in blocker_flows
    assert "H02-redelivery-resume" in blocker_flows
