# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    ParentExecutionAuthority,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.governance.active_execution_authority import (
    peek_active_execution_authority,
    require_active_execution_authority,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True)
class Ping:
    value: str


@dataclass(frozen=True)
class Pong:
    value: str


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _root_authority(
    *scopes: str,
) -> ParentExecutionAuthority:
    return ParentExecutionAuthority.scoped(scopes)


@pytest.mark.asyncio
async def test_root_boundary_binds_authority_from_task() -> None:
    root = _root_identity()
    captured: list[ParentExecutionAuthority] = []

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            captured.append(require_active_execution_authority())
            return Pong(value=request.value)

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read", "write", "delete"),
    ).execute(Ping(value="root"))

    assert captured[0].permission_scopes == ("read", "write", "delete")
    assert peek_active_execution_authority() is None


@pytest.mark.asyncio
async def test_child_narrows_parent_authority() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    child_captured: list[ParentExecutionAuthority] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            child_captured.append(require_active_execution_authority())
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
                requested_permission_scopes=("read", "write"),
            )

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read", "write", "delete"),
    ).execute(Ping(value="child"))

    assert child_captured[0].permission_scopes == ("read", "write")


@pytest.mark.asyncio
async def test_nested_child_cannot_expand_beyond_immediate_parent() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()

    class LeafDelegate:
        async def execute(self, request: Ping) -> Pong:
            return Pong(value=request.value)

    class MidDelegate:
        async def execute(self, request: Ping) -> Pong:
            with pytest.raises(DelegationAuthorityError):
                await child_runner.execute(
                    request=request,
                    delegate=LeafDelegate(),
                    requested_permission_scopes=("read", "delete"),
                )
            return Pong(value="blocked")

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=MidDelegate(),
                requested_permission_scopes=("read", "write"),
            )

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read", "write", "delete"),
    ).execute(Ping(value="nested"))


@pytest.mark.asyncio
async def test_child_inherits_parent_authority_when_no_scopes_requested() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    child_captured: list[ParentExecutionAuthority] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            child_captured.append(require_active_execution_authority())
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
            )

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read", "write"),
    ).execute(Ping(value="inherit"))

    assert child_captured[0].permission_scopes == ("read", "write")


@pytest.mark.asyncio
async def test_child_restores_parent_authority_after_success() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    parent_seen: list[tuple[str, ...]] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            parent_seen.append(
                require_active_execution_authority().permission_scopes,
            )
            await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
                requested_permission_scopes=("read",),
            )
            parent_seen.append(
                require_active_execution_authority().permission_scopes,
            )
            return Pong(value="ok")

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read", "write"),
    ).execute(Ping(value="restore"))

    assert parent_seen == [("read", "write"), ("read", "write")]


@pytest.mark.asyncio
async def test_child_runner_fails_without_active_authority() -> None:
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        reset_active_execution_identity,
    )

    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    child_runner = ChildExecutionRunner[Ping, Pong]()

    try:
        with pytest.raises(RuntimeError, match="active execution authority required"):
            await child_runner.execute(
                request=Ping(value="orphan"),
                delegate=ChildDelegateStub(),
            )
    finally:
        reset_active_execution_identity(token)


class ChildDelegateStub:
    async def execute(self, request: Ping) -> Pong:
        return Pong(value=request.value)
