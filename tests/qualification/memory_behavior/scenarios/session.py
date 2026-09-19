# © Artur Czarnecki. All rights reserved.

"""SESSION scope behavioral scenarios."""

from __future__ import annotations

from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlPlaneScope,
    MemoryControlRecallRequest,
    MemoryControlScopeRef,
)
from tests.qualification.memory_behavior.contracts import BehaviorEvalContext
from tests.qualification.memory_behavior.fixtures import (
    InMemoryEpisodicMemoryCapability,
    TENANT_A,
    build_session_control_plane,
    request_identity,
    session_scope,
)


async def run_session_01_basic_episodic_recall(_ctx: BehaviorEvalContext) -> None:
    episodic = InMemoryEpisodicMemoryCapability()
    plane = build_session_control_plane(episodic)
    identity = request_identity(user_id="sess-user")
    episodic.seed_turn(TENANT_A, "sess-1", "turn-1", "We discussed vector memory wiring.", "sess-user")
    scope = session_scope(identity, session_id="sess-1")
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="vector memory", top_k=3),
    )
    assert recall.items
    assert recall.items[0].entry_id == "turn-1"


async def run_session_02_session_isolation(_ctx: BehaviorEvalContext) -> None:
    episodic = InMemoryEpisodicMemoryCapability()
    plane = build_session_control_plane(episodic)
    identity = request_identity(user_id="sess-user")
    episodic.seed_turn(TENANT_A, "sess-a", "turn-a", "session A secret", "sess-user")
    episodic.seed_turn(TENANT_A, "sess-b", "turn-b", "session B other", "sess-user")
    scope_b = session_scope(identity, session_id="sess-b")
    recall = await plane.recall(
        identity,
        scope_b,
        MemoryControlRecallRequest(query="secret", top_k=5),
    )
    assert all("session A" not in item.content for item in recall.items)


async def run_session_03_cross_session_when_enabled(_ctx: BehaviorEvalContext) -> None:
    episodic = InMemoryEpisodicMemoryCapability(include_cross_session=True)
    plane = build_session_control_plane(episodic)
    identity = request_identity(user_id="sess-user")
    episodic.seed_turn(TENANT_A, "sess-a", "turn-a", "earlier episodic fact", "sess-user")
    episodic.seed_turn(TENANT_A, "sess-b", "turn-b", "later episodic fact", "sess-user")
    scope_b = session_scope(identity, session_id="sess-b")
    recall = await plane.recall(
        identity,
        scope_b,
        MemoryControlRecallRequest(query="episodic", top_k=5),
    )
    contents = {item.content for item in recall.items}
    assert "earlier episodic fact" in contents


async def run_session_05_scope_authority_on_recall(_ctx: BehaviorEvalContext) -> None:
    episodic = InMemoryEpisodicMemoryCapability()
    plane = build_session_control_plane(episodic)
    identity = request_identity(user_id="sess-user")
    bad_scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.SESSION,
        tenant_id="other-tenant",
        session_id="sess-1",
    )
    try:
        await plane.recall(
            identity,
            bad_scope,
            MemoryControlRecallRequest(query="x", top_k=3),
        )
    except MemoryControlAccessDenied:
        return
    raise AssertionError("expected MemoryControlAccessDenied for session scope spoof")
