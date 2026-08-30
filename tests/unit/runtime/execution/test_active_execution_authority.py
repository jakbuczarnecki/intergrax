# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    EffectiveDelegationAuthority,
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
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.nexus.budget.budget_models import RunBudget

_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())
from intergrax.runtime.governance.active_execution_authority import (
    peek_active_effective_delegation,
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
    assert peek_active_effective_delegation() is None


@pytest.mark.asyncio
async def test_root_boundary_has_no_delegation_evidence() -> None:
    root = _root_identity()
    captured: list[EffectiveDelegationAuthority | None] = []

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            captured.append(peek_active_effective_delegation())
            return Pong(value=request.value)

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read", "write"),
    ).execute(Ping(value="root"))

    assert captured == [None]
    assert peek_active_effective_delegation() is None


@pytest.mark.asyncio
async def test_child_narrows_parent_authority() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=_UNLIMITED_LEDGER)
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
async def test_delegated_child_exposes_effective_delegation_evidence() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=_UNLIMITED_LEDGER)
    captured: list[EffectiveDelegationAuthority | None] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            captured.append(peek_active_effective_delegation())
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
    ).execute(Ping(value="delegated"))

    evidence = captured[0]
    assert isinstance(evidence, EffectiveDelegationAuthority)
    assert evidence.requested_permission_scopes == ("read", "write")
    assert evidence.effective_permission_scopes == ("read", "write")
    assert peek_active_effective_delegation() is None


@pytest.mark.asyncio
async def test_nested_child_cannot_expand_beyond_immediate_parent() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=_UNLIMITED_LEDGER)

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
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=_UNLIMITED_LEDGER)
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
    assert peek_active_effective_delegation() is None


@pytest.mark.asyncio
async def test_child_restores_parent_authority_after_success() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=_UNLIMITED_LEDGER)
    parent_seen: list[tuple[str, ...]] = []
    parent_evidence: list[EffectiveDelegationAuthority | None] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            parent_seen.append(
                require_active_execution_authority().permission_scopes,
            )
            parent_evidence.append(peek_active_effective_delegation())
            await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
                requested_permission_scopes=("read",),
            )
            parent_seen.append(
                require_active_execution_authority().permission_scopes,
            )
            parent_evidence.append(peek_active_effective_delegation())
            return Pong(value="ok")

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read", "write"),
    ).execute(Ping(value="restore"))

    assert parent_seen == [("read", "write"), ("read", "write")]
    assert parent_evidence == [None, None]


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
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=_UNLIMITED_LEDGER)

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
