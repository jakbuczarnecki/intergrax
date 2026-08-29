# © Artur Czarnecki. All rights reserved.

"""UE-8P2 — ChildExecutionRunner authority policy extensibility and default semantics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.runtime.execution.authority.policy import (
    ChildAuthorityContext,
    ChildAuthorityResolution,
    DefaultStrictAuthorityPolicy,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.governance.active_execution_authority import (
    peek_active_effective_delegation,
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


def _root_authority(*scopes: str) -> ParentExecutionAuthority:
    return ParentExecutionAuthority.scoped(scopes)


class _CustomAuthorityPolicy:
    def resolve_child_authority(
        self,
        context: ChildAuthorityContext,
    ) -> ChildAuthorityResolution:
        _ = context
        return ChildAuthorityResolution(
            authority=ParentExecutionAuthority.scoped(("plugin-scope",)),
            effective_delegation=None,
        )


@pytest.mark.asyncio
async def test_child_runner_uses_injected_custom_policy() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong](
        authority_policy=_CustomAuthorityPolicy(),
    )
    captured: list[ParentExecutionAuthority] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            captured.append(require_active_execution_authority())
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
        authority=_root_authority("read", "write", "delete"),
    ).execute(Ping(value="child"))

    assert captured[0].permission_scopes == ("plugin-scope",)


@pytest.mark.asyncio
async def test_default_policy_narrows_parent_authority() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    captured: list[ParentExecutionAuthority] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            captured.append(require_active_execution_authority())
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

    assert captured[0].permission_scopes == ("read", "write")


@pytest.mark.asyncio
async def test_default_policy_normal_child_inherits_parent_unchanged() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()
    captured: list[ParentExecutionAuthority] = []
    evidence: list[EffectiveDelegationAuthority | None] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            captured.append(require_active_execution_authority())
            evidence.append(peek_active_effective_delegation())
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
    ).execute(Ping(value="child"))

    assert captured[0].permission_scopes == ("read", "write")
    assert evidence == [None]


@pytest.mark.asyncio
async def test_default_policy_nested_child_overreach_denied() -> None:
    root = _root_identity()
    child_runner = ChildExecutionRunner[Ping, Pong]()

    class GrandchildDelegate:
        async def execute(self, request: Ping) -> Pong:
            return Pong(value=request.value)

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=GrandchildDelegate(),
                requested_permission_scopes=("delete",),
            )

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
                requested_permission_scopes=("read", "write"),
            )

    with pytest.raises(DelegationAuthorityError):
        await ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=root,
            authority=_root_authority("read", "write", "delete"),
        ).execute(Ping(value="nested"))


def test_default_strict_policy_none_vs_empty_tuple_semantics() -> None:
    policy = DefaultStrictAuthorityPolicy()
    parent = ParentExecutionAuthority.scoped(("read", "write"))

    inherited = policy.resolve_child_authority(
        ChildAuthorityContext(
            parent_authority=parent,
            requested_permission_scopes=None,
        )
    )
    assert inherited.authority == parent
    assert inherited.effective_delegation is None

    delegated = policy.resolve_child_authority(
        ChildAuthorityContext(
            parent_authority=parent,
            requested_permission_scopes=(),
        )
    )
    assert delegated.authority.permission_scopes == ()
    assert delegated.effective_delegation is not None
    assert delegated.effective_delegation.effective_permission_scopes == ()
