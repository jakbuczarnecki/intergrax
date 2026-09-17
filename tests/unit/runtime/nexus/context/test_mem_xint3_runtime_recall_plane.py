# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-3: runtime durable LTM recall routes through MemoryControlPlane."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    MemoryControlScopeRef,
    user_memory_scope,
)
from intergrax.memory.user_profile_memory import MemoryKind
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.memory_context_invocation import (
    populate_request_memory_recall_metadata,
    run_longterm_memory_context,
)
from intergrax.runtime.nexus.context.provider_handles import LTM_ENTRIES_METADATA_KEY
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import (
    FakeLLMAdapter,
    build_runtime_request_for_tests,
    canonical_execution_identity_scope,
)
from testing_support.memory_control_plane_test_stub import MemoryControlPlaneTestStub

pytestmark = pytest.mark.gate

_CANONICAL_TENANT = "tenant-a"
_CANONICAL_USER = "user-u1"


def _canonical_identity() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_CANONICAL_TENANT,
        user_id=_CANONICAL_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_CANONICAL_USER,
    )


@dataclass
class RecordingRecallPlane(MemoryControlPlaneTestStub):
    recall_calls: list[tuple[RequestIdentity, MemoryControlScopeRef, MemoryControlRecallRequest]] = field(
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
                    entry_id="ltm-1",
                    content="prefers dark mode",
                    kind=MemoryKind.USER_FACT,
                    score=0.88,
                ),
            ),
            reason="hits",
        )


def _runtime_config(plane: RecordingRecallPlane | None, **overrides: object) -> RuntimeConfig:
    base: dict[str, object] = {
        "llm_adapter": FakeLLMAdapter(),
        "production_mode": False,
        "enable_rag": False,
        "enable_user_longterm_memory": True,
        "enable_session_vector_index": False,
    }
    base.update(overrides)
    extras: dict[str, object] = {}
    if plane is not None:
        extras["memory_control_plane"] = plane
    base["tool_wiring_context"] = ToolWiringContext(extras=extras)
    return RuntimeConfig(**base)


def _state_for_recall(
    request: RuntimeRequest,
    *,
    config: RuntimeConfig,
    session_manager: SessionManager | None = None,
) -> RuntimeState:
    sm = session_manager or MagicMock(spec=SessionManager)
    ctx = RuntimeContext.build(config=config, session_manager=sm)
    return RuntimeState(context=ctx, request=request, run_id=str(request.run_id))


@pytest.mark.asyncio
async def test_ltm_recall_uses_control_plane_not_session_manager() -> None:
    plane = RecordingRecallPlane()
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_CANONICAL_USER,
            message="find prefs",
        ),
        canonical_identity=_canonical_identity(),
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_user_longterm_memory = AsyncMock()
    state = _state_for_recall(request, config=_runtime_config(plane), session_manager=sm)

    with canonical_execution_identity_scope(str(request.run_id)):
        await run_longterm_memory_context(state)

    sm.search_user_longterm_memory.assert_not_awaited()
    assert len(plane.recall_calls) == 1
    call_identity, scope, recall_req = plane.recall_calls[0]
    assert call_identity.user_id == _CANONICAL_USER
    assert scope == user_memory_scope(call_identity)
    assert recall_req.query == "find prefs"
    assert state.used_user_longterm_memory is True
    assert state.user_longterm_memory_result is not None
    assert len(state.user_longterm_memory_result.get("hits") or []) == 1


@pytest.mark.asyncio
async def test_populate_metadata_ltm_via_plane() -> None:
    plane = RecordingRecallPlane()
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_CANONICAL_USER,
            message="prefs",
        ),
        canonical_identity=_canonical_identity(),
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_user_longterm_memory = AsyncMock()
    config = _runtime_config(plane)

    await populate_request_memory_recall_metadata(request, config=config, session_manager=sm)

    sm.search_user_longterm_memory.assert_not_awaited()
    assert LTM_ENTRIES_METADATA_KEY in request.metadata
    assert len(request.metadata[LTM_ENTRIES_METADATA_KEY]) == 1


@pytest.mark.asyncio
async def test_ltm_recall_fail_closed_without_plane() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_CANONICAL_USER,
            message="find prefs",
        ),
        canonical_identity=_canonical_identity(),
    )
    state = _state_for_recall(request, config=_runtime_config(None))

    with pytest.raises(MemoryControlAccessDenied, match="memory_control_plane_not_configured"):
        await run_longterm_memory_context(state)
