# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    peek_active_parent_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.child import ChildExecutionRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True)
class Ping:
    value: str


@dataclass(frozen=True)
class Pong:
    value: str


class ChildObservingDelegate:
    def __init__(self, captured: dict[str, RunId | AttemptId | ExecutionId | None]) -> None:
        self._captured = captured

    async def execute(self, request: Ping) -> Pong:
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        self._captured["run_id"] = run_id
        self._captured["attempt_id"] = attempt_id
        self._captured["execution_id"] = execution_id
        self._captured["parent_execution_id"] = peek_active_parent_execution_id()
        return Pong(value=request.value)


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _root_authority() -> ParentExecutionAuthority:
    return ParentExecutionAuthority.unrestricted_root()


@pytest.mark.asyncio
async def test_child_preserves_run_attempt_mints_new_execution_id() -> None:
    root = _root_identity()
    child_captured: dict[str, RunId | AttemptId | ExecutionId | None] = {}
    child_runner = ChildExecutionRunner[Ping, Pong]()

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildObservingDelegate(child_captured),
            )

    root_boundary = ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority(),
    )

    await root_boundary.execute(Ping(value="child"))

    assert child_captured["run_id"] == root.run_id
    assert child_captured["attempt_id"] == root.attempt_id
    assert child_captured["execution_id"] != root.execution_id
    assert child_captured["parent_execution_id"] == root.execution_id
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_child_restores_parent_execution_after_success() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    parent_seen: list[ExecutionId] = []

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            parent_seen.append(require_active_execution_id())
            await child_runner.execute(
                request=request,
                delegate=ChildObservingDelegate({}),
            )
            parent_seen.append(require_active_execution_id())
            return Pong(value="ok")

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority(),
    ).execute(
        Ping(value="child"),
    )

    assert parent_seen == [root.execution_id, root.execution_id]


@pytest.mark.asyncio
async def test_nested_child_execution_lineage() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    identities: list[
        tuple[ExecutionId, ExecutionId | None, RunId, AttemptId]
    ] = []

    class LeafDelegate:
        async def execute(self, request: Ping) -> Pong:
            run_id, attempt_id = require_active_execution_identity()
            identities.append(
                (
                    require_active_execution_id(),
                    peek_active_parent_execution_id(),
                    run_id,
                    attempt_id,
                ),
            )
            return Pong(value="leaf")

    class MidDelegate:
        async def execute(self, request: Ping) -> Pong:
            run_id, attempt_id = require_active_execution_identity()
            identities.append(
                (
                    require_active_execution_id(),
                    peek_active_parent_execution_id(),
                    run_id,
                    attempt_id,
                ),
            )
            return await child_runner.execute(request=request, delegate=LeafDelegate())

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            run_id, attempt_id = require_active_execution_identity()
            identities.append(
                (
                    require_active_execution_id(),
                    peek_active_parent_execution_id(),
                    run_id,
                    attempt_id,
                ),
            )
            return await child_runner.execute(request=request, delegate=MidDelegate())

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority(),
    ).execute(
        Ping(value="nested"),
    )

    e1_id, e1_parent, e1_run, e1_attempt = identities[0]
    e2_id, e2_parent, e2_run, e2_attempt = identities[1]
    e3_id, e3_parent, e3_run, e3_attempt = identities[2]

    assert e1_id == root.execution_id
    assert e1_parent is None
    assert e2_parent == e1_id
    assert e3_parent == e2_id
    assert e1_id != e2_id != e3_id
    assert e1_run == e2_run == e3_run == root.run_id
    assert e1_attempt == e2_attempt == e3_attempt == root.attempt_id
    assert peek_active_execution_identity() is None


@pytest.mark.asyncio
async def test_parallel_children_isolate_execution_ids() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    child_records: list[tuple[ExecutionId, ExecutionId | None, RunId, AttemptId]] = []
    lock = asyncio.Lock()

    class ParallelChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            run_id, attempt_id = require_active_execution_identity()
            execution_id = require_active_execution_id()
            parent_execution_id = peek_active_parent_execution_id()
            async with lock:
                child_records.append(
                    (execution_id, parent_execution_id, run_id, attempt_id),
                )
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            await asyncio.gather(
                child_runner.execute(
                    request=Ping(value="c1"),
                    delegate=ParallelChildDelegate(),
                ),
                child_runner.execute(
                    request=Ping(value="c2"),
                    delegate=ParallelChildDelegate(),
                ),
                child_runner.execute(
                    request=Ping(value="c3"),
                    delegate=ParallelChildDelegate(),
                ),
            )
            return Pong(value="done")

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority(),
    ).execute(
        Ping(value="parallel"),
    )

    execution_ids = [record[0] for record in child_records]
    assert len(execution_ids) == 3
    assert len(set(execution_ids)) == 3
    for execution_id, parent_execution_id, run_id, attempt_id in child_records:
        assert parent_execution_id == root.execution_id
        assert run_id == root.run_id
        assert attempt_id == root.attempt_id
        assert execution_id != root.execution_id

    assert peek_active_execution_identity() is None


@pytest.mark.asyncio
async def test_child_exception_propagates_and_restores_parent() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()

    class FailingChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            raise ValueError("child-boom")

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            await child_runner.execute(
                request=request,
                delegate=FailingChildDelegate(),
            )
            return Pong(value="never")

    with pytest.raises(ValueError, match="child-boom"):
        await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority(),
    ).execute(
            Ping(value="fail"),
        )

    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_child_runner_fails_without_active_execution() -> None:
    child_runner = ChildExecutionRunner[Ping, Pong]()

    with pytest.raises(RuntimeError, match="active execution identity required"):
        await child_runner.execute(
            request=Ping(value="orphan"),
            delegate=ChildObservingDelegate({}),
        )


@pytest.mark.asyncio
async def test_child_runner_fails_without_active_execution_id() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    child_runner = ChildExecutionRunner[Ping, Pong]()

    try:
        with pytest.raises(RuntimeError, match="active ExecutionId required"):
            await child_runner.execute(
                request=Ping(value="no-exec"),
                delegate=ChildObservingDelegate({}),
            )
    finally:
        reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_child_admission_hook_sees_child_identity() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    admission_captured: dict[str, RunId | AttemptId | ExecutionId | None] = {}

    class AdmissionHook:
        async def admit(self, request: Ping) -> None:
            run_id, attempt_id = require_active_execution_identity()
            admission_captured["run_id"] = run_id
            admission_captured["attempt_id"] = attempt_id
            admission_captured["execution_id"] = require_active_execution_id()
            admission_captured["parent_execution_id"] = peek_active_parent_execution_id()

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildObservingDelegate({}),
                admission_hooks=(AdmissionHook(),),
            )

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority(),
    ).execute(
        Ping(value="admit"),
    )

    assert admission_captured["run_id"] == root.run_id
    assert admission_captured["attempt_id"] == root.attempt_id
    assert admission_captured["execution_id"] != root.execution_id
    assert admission_captured["parent_execution_id"] == root.execution_id
