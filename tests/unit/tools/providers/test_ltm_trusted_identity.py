# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R2 / MEM-XINT-3: LTM tool fail-closed trusted identity and control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlPlaneScope,
    MemoryControlRecallItem,
    MemoryControlRecallRequest,
    MemoryControlRecallResult,
    MemoryControlRememberRequest,
    MemoryControlRememberResult,
    MemoryControlScopeRef,
    user_memory_scope,
)
from intergrax.memory.user_profile_memory import MemoryKind
from intergrax.tools.providers.ltm.contracts import LtmSearchInput, LtmWriteFactInput
from intergrax.tools.providers.ltm.service import ltm_search, ltm_write_fact
from intergrax.tools.registry.runtime_bindings import UserProfileManagerBinding
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.memory_control_plane_test_stub import MemoryControlPlaneTestStub

pytestmark = pytest.mark.gate


@dataclass
class RecordingMemoryControlPlane(MemoryControlPlaneTestStub):
    recall_calls: list[tuple[RequestIdentity, MemoryControlScopeRef, MemoryControlRecallRequest]] = field(
        default_factory=list
    )
    remember_calls: list[tuple[RequestIdentity, MemoryControlScopeRef, MemoryControlRememberRequest]] = field(
        default_factory=list
    )

    async def recall(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> MemoryControlRecallResult:
        self.recall_calls.append((identity, scope, request))
        return MemoryControlRecallResult(
            scope=MemoryControlPlaneScope.USER,
            items=(
                MemoryControlRecallItem(
                    entry_id="e1",
                    content="hit",
                    kind=MemoryKind.USER_FACT,
                    score=0.9,
                ),
            ),
        )

    async def remember(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> MemoryControlRememberResult:
        self.remember_calls.append((identity, scope, request))
        return MemoryControlRememberResult(
            scope=MemoryControlPlaneScope.USER,
            entry_id="saved-1",
        )


def _manager_binding() -> UserProfileManagerBinding:
    binding = MagicMock(spec=UserProfileManagerBinding)
    binding.add_memory_entry = AsyncMock(return_value=MagicMock(entry_id="e1"))
    binding.search_longterm_memory = AsyncMock(return_value={"hits": [], "scores": []})
    return binding


def _ctx(
    *,
    identity: RequestIdentity | None = None,
    plane: RecordingMemoryControlPlane | None = None,
    with_manager: bool = False,
) -> ToolWiringContext:
    extras: dict[str, object] = {}
    if identity is not None:
        extras["request_identity"] = identity
    if plane is not None:
        extras["memory_control_plane"] = plane
    return ToolWiringContext(
        user_profile_manager=_manager_binding() if with_manager else None,
        extras=extras,
    )


def test_ltm_write_fact_requires_trusted_identity() -> None:
    plane = RecordingMemoryControlPlane()
    ctx = _ctx(plane=plane)
    with pytest.raises(MemoryControlAccessDenied, match="trusted request identity required"):
        ltm_write_fact(ctx, LtmWriteFactInput(user_id="u1", content="fact"))
    assert plane.remember_calls == []


def test_ltm_write_fact_requires_plane() -> None:
    identity = RequestIdentity(tenant_id="t1", user_id="u1")
    manager = _manager_binding()
    ctx = ToolWiringContext(
        user_profile_manager=manager,
        extras={"request_identity": identity},
    )
    with pytest.raises(MemoryControlAccessDenied, match="memory_control_plane_not_configured"):
        ltm_write_fact(ctx, LtmWriteFactInput(user_id="u1", content="fact"))
    manager.add_memory_entry.assert_not_awaited()


def test_ltm_write_fact_rejects_user_mismatch() -> None:
    identity = RequestIdentity(tenant_id="t1", user_id="user-a")
    plane = RecordingMemoryControlPlane()
    ctx = _ctx(identity=identity, plane=plane)
    with pytest.raises(MemoryControlAccessDenied, match="conflicts"):
        ltm_write_fact(ctx, LtmWriteFactInput(user_id="user-b", content="fact"))
    assert plane.remember_calls == []


def test_ltm_write_fact_passes_original_identity_to_plane() -> None:
    identity = RequestIdentity(
        tenant_id="t1",
        user_id="u1",
        principal_type=PrincipalType.USER,
        auth_subject="sub-u1",
    )
    plane = RecordingMemoryControlPlane()
    ctx = _ctx(identity=identity, plane=plane, with_manager=True)
    manager = ctx.user_profile_manager
    result = ltm_write_fact(ctx, LtmWriteFactInput(user_id="u1", content="likes tea"))
    assert result.written is True
    assert result.entry_id == "saved-1"
    assert len(plane.remember_calls) == 1
    call_identity, scope, _request = plane.remember_calls[0]
    assert call_identity is identity
    assert scope == user_memory_scope(identity)
    manager.add_memory_entry.assert_not_awaited()


def test_ltm_search_uses_plane_not_manager() -> None:
    identity = RequestIdentity(tenant_id="t1", user_id="u1")
    plane = RecordingMemoryControlPlane()
    ctx = _ctx(identity=identity, plane=plane, with_manager=True)
    manager = ctx.user_profile_manager
    out = ltm_search(ctx, LtmSearchInput(user_id="u1", query="tea"))
    assert out.used is True
    assert len(plane.recall_calls) == 1
    manager.search_longterm_memory.assert_not_awaited()
