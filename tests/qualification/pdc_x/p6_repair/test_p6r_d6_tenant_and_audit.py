# © Artur Czarnecki. All rights reserved.

"""B-P6-R items 11 and 12 - tenant-bound memory plane and DENY audit before a missing-capability raise."""

from __future__ import annotations

import pytest
from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    ToolAuthorizationDecisionState,
    ToolAuthorizationRiskLevel,
)
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlForgetRequest,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.runtime.agent_governance.errors import CapabilityNotGrantedError
from intergrax.runtime.nexus.artifacts.governed_access import _authorization_request
from tests.qualification.pdc_x.d8._d8_support import TENANT_A, TENANT_B, make_context, make_governance
from tests.qualification.pdc_x.xd.test_xd_02_agent_context import _identity, _memory_for

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------- item 11


_ITEM_11_BLOCKED = (
    "B-P6-R item 11 held for a write-set amendment: a plane that rejects a foreign-tenant identity "
    "breaks six existing tests in tests/unit/memory/test_mem_ent10_memory_security_governance.py whose fixture "
    "binds UserProfileManager to 'tenant-test' (line 79) but acts as identity tenant 'tenant-10'; that fixture "
    "must bind _TENANT first"
)


@pytest.mark.xfail(strict=True, reason=_ITEM_11_BLOCKED)
@pytest.mark.asyncio
async def test_a_foreign_tenant_identity_is_refused_on_every_user_scope_operation() -> None:
    memory_a = _memory_for(TENANT_A)
    entry_id = await memory_a.remember("tenant-a private preference ALPHA-SECRET")
    foreign = _identity(TENANT_B)
    scope = user_memory_scope(foreign)

    with pytest.raises(MemoryControlAccessDenied):
        await memory_a.plane.recall(foreign, scope, MemoryControlRecallRequest(query="preference", top_k=5))
    with pytest.raises(MemoryControlAccessDenied):
        await memory_a.plane.remember(
            foreign, scope, MemoryControlRememberRequest(entry=UserProfileMemoryEntry(content="planted"))
        )
    with pytest.raises(MemoryControlAccessDenied):
        await memory_a.plane.forget(foreign, scope, MemoryControlForgetRequest(entry_id=entry_id))


@pytest.mark.xfail(strict=True, reason=_ITEM_11_BLOCKED)
@pytest.mark.asyncio
async def test_the_bound_tenant_keeps_working_after_a_foreign_attempt() -> None:
    memory_a = _memory_for(TENANT_A)
    await memory_a.remember("tenant-a private preference ALPHA-SECRET")
    own = _identity(TENANT_A)
    with pytest.raises(MemoryControlAccessDenied):
        await memory_a.plane.recall(
            _identity(TENANT_B),
            user_memory_scope(_identity(TENANT_B)),
            MemoryControlRecallRequest(query="preference", top_k=5),
        )
    result = await memory_a.plane.recall(own, user_memory_scope(own), MemoryControlRecallRequest(query="preference"))
    assert "ALPHA-SECRET" in repr(result)


# ----------------------------------------------------------------------- item 12


def test_a_missing_capability_is_audited_as_deny_before_the_error_is_raised() -> None:
    harness = make_governance(granted={})  # nobody holds any capability
    context = make_context(tenant_id=TENANT_A)
    request = _authorization_request(
        context,
        capability="artifact.read",
        artifact_id="a-1",
        requested_action="read",
        risk=ToolAuthorizationRiskLevel.LOW,
    )
    assert isinstance(context.agent, AgentIdentity)

    with pytest.raises(CapabilityNotGrantedError):
        harness.boundary.authorize_tool(request)

    events = harness.sink.events
    assert len(events) == 1
    assert events[0].decision is ToolAuthorizationDecisionState.DENY
    assert events[0].capability == "artifact.read"
    assert events[0].tenant_id == TENANT_A


def test_a_granted_capability_still_records_exactly_one_decision() -> None:
    harness = make_governance(granted={(TENANT_A, "agent-reader"): frozenset({"artifact.read"})})
    request = _authorization_request(
        make_context(tenant_id=TENANT_A),
        capability="artifact.read",
        artifact_id="a-1",
        requested_action="read",
        risk=ToolAuthorizationRiskLevel.LOW,
    )
    decision = harness.boundary.authorize_tool(request)
    assert decision.decision is ToolAuthorizationDecisionState.ALLOW
    assert len(harness.sink.events) == 1
