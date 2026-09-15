# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-1R3: LTM and episodic recall use verified canonical RequestIdentity only."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.request_identity_spine import verified_request_identity_for_memory_recall
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.memory_context_invocation import (
    populate_request_memory_recall_metadata,
    run_longterm_memory_context,
    run_session_semantic_recall_context,
)
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.session_manager import SessionManager
from testing_support.builder import (
    FakeLLMAdapter,
    build_runtime_request_for_tests,
    canonical_execution_identity_scope,
)

pytestmark = pytest.mark.gate

_CANONICAL_TENANT = "tenant-a"
_CANONICAL_USER = "user-u1"
_ATTACKER_USER = "attacker-user"
_ATTACKER_TENANT = "attacker-tenant"
_CONFLICT_USER = "user-u2"


def _canonical_identity(
    *,
    tenant_id: str = _CANONICAL_TENANT,
    user_id: str = _CANONICAL_USER,
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _runtime_config(**overrides: object) -> RuntimeConfig:
    base = {
        "llm_adapter": FakeLLMAdapter(),
        "production_mode": False,
        "enable_rag": False,
        "enable_user_longterm_memory": True,
        "enable_session_vector_index": True,
    }
    base.update(overrides)
    return RuntimeConfig(**base)


def _state_for_recall(
    request: RuntimeRequest,
    *,
    session_manager: SessionManager | None = None,
    config: RuntimeConfig | None = None,
) -> RuntimeState:
    sm = session_manager or MagicMock(spec=SessionManager)
    cfg = config or _runtime_config()
    ctx = RuntimeContext.build(config=cfg, session_manager=sm)
    return RuntimeState(context=ctx, request=request, run_id=str(request.run_id))


@pytest.mark.asyncio
async def test_ltm_recall_uses_canonical_user_when_metadata_consistent() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_CANONICAL_USER,
            message="find prefs",
        ),
        canonical_identity=_canonical_identity(),
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_user_longterm_memory = AsyncMock(return_value={"hits": [], "used_longterm": False})
    state = _state_for_recall(request, session_manager=sm)

    with canonical_execution_identity_scope(str(request.run_id)):
        await run_longterm_memory_context(state)

    sm.search_user_longterm_memory.assert_awaited_once()
    assert sm.search_user_longterm_memory.await_args.args[0] == _CANONICAL_USER


@pytest.mark.asyncio
async def test_ltm_recall_skipped_without_canonical_identity() -> None:
    request = replace(
        build_runtime_request_for_tests(
            user_id=_ATTACKER_USER,
            message="find prefs",
            metadata={"user_id": _ATTACKER_USER},
        ),
        canonical_identity=None,
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_user_longterm_memory = AsyncMock()
    state = _state_for_recall(request, session_manager=sm)

    await run_longterm_memory_context(state)

    sm.search_user_longterm_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_ltm_recall_rejects_metadata_user_conflict() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_CANONICAL_USER,
            message="find prefs",
            metadata={"user_id": _CONFLICT_USER},
        ),
        canonical_identity=_canonical_identity(),
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_user_longterm_memory = AsyncMock()
    state = _state_for_recall(request, session_manager=sm)

    with pytest.raises(ValueError, match="metadata user_id conflicts"):
        await run_longterm_memory_context(state)
    sm.search_user_longterm_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_episodic_recall_uses_canonical_tenant() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_CANONICAL_USER,
            session_id="session-s1",
            message="recall turn",
        ),
        canonical_identity=_canonical_identity(),
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_session_semantic_recall = AsyncMock(return_value=[])
    state = _state_for_recall(request, session_manager=sm)

    await run_session_semantic_recall_context(state)

    sm.search_session_semantic_recall.assert_awaited_once()
    kwargs = sm.search_session_semantic_recall.await_args.kwargs
    assert kwargs["tenant_id"] == _CANONICAL_TENANT
    assert kwargs["session_id"] == "session-s1"


@pytest.mark.asyncio
async def test_episodic_recall_rejects_request_tenant_conflict() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_ATTACKER_TENANT,
            user_id=_CANONICAL_USER,
            session_id="session-s1",
            message="recall turn",
        ),
        canonical_identity=_canonical_identity(),
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_session_semantic_recall = AsyncMock()
    state = _state_for_recall(request, session_manager=sm)

    with pytest.raises(ValueError, match="request tenant_id conflicts"):
        await run_session_semantic_recall_context(state)
    sm.search_session_semantic_recall.assert_not_awaited()


@pytest.mark.asyncio
async def test_episodic_recall_skipped_without_canonical_identity() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_ATTACKER_TENANT,
            session_id="session-s1",
            message="recall turn",
        ),
        canonical_identity=None,
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_session_semantic_recall = AsyncMock()
    state = _state_for_recall(request, session_manager=sm)

    await run_session_semantic_recall_context(state)

    sm.search_session_semantic_recall.assert_not_awaited()


@pytest.mark.asyncio
async def test_episodic_recall_uses_canonical_user_not_request_user() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_ATTACKER_USER,
            session_id="session-s1",
            message="recall turn",
        ),
        canonical_identity=_canonical_identity(),
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_session_semantic_recall = AsyncMock(return_value=[])
    state = _state_for_recall(request, session_manager=sm)

    await run_session_semantic_recall_context(state)

    kwargs = sm.search_session_semantic_recall.await_args.kwargs
    assert kwargs["user_id"] == _CANONICAL_USER


@pytest.mark.asyncio
async def test_populate_metadata_ltm_skipped_without_canonical_identity() -> None:
    request = replace(
        build_runtime_request_for_tests(
            user_id=_ATTACKER_USER,
            message="query",
            metadata={"user_id": _ATTACKER_USER, "tenant_id": _ATTACKER_TENANT},
        ),
        canonical_identity=None,
    )
    sm = MagicMock(spec=SessionManager)
    sm.search_user_longterm_memory = AsyncMock()
    sm.search_session_semantic_recall = AsyncMock()
    config = _runtime_config()

    await populate_request_memory_recall_metadata(request, config=config, session_manager=sm)

    assert "memory_profile" in request.metadata
    sm.search_user_longterm_memory.assert_not_awaited()
    sm.search_session_semantic_recall.assert_not_awaited()


def test_verified_request_identity_for_memory_recall_metadata_tenant_conflict() -> None:
    canonical = _canonical_identity()
    with pytest.raises(ValueError, match="metadata tenant_id conflicts"):
        verified_request_identity_for_memory_recall(
            canonical,
            metadata={"tenant_id": _ATTACKER_TENANT},
            legacy_tenant_id=_CANONICAL_TENANT,
        )
