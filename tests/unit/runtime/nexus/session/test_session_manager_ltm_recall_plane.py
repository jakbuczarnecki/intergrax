# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-2-R: SessionManager LTM recall routes through MemoryControlPlane."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.memory.contracts.memory_control import (
    MemoryControlGovernanceDenied,
    MemoryControlPlaneScope,
    MemoryControlRecallItem,
    MemoryControlRecallResult,
    MemoryKind,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemoryGovernanceOperation,
)
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager

pytestmark = pytest.mark.unit


class _RecordingPlane:
    recall = AsyncMock(
        return_value=MemoryControlRecallResult(
            scope=MemoryControlPlaneScope.USER,
            items=(
                MemoryControlRecallItem(
                    entry_id="e1",
                    content="fact",
                    kind=MemoryKind.OTHER,
                    score=0.9,
                ),
            ),
            used_semantic=True,
            reason="hits",
        )
    )


@pytest.mark.asyncio
async def test_session_manager_ltm_recall_uses_memory_control_plane() -> None:
    plane = _RecordingPlane()
    manager = MagicMock(spec=UserProfileManager)
    manager.search_longterm_memory = AsyncMock()

    session_manager = SessionManager(
        InMemorySessionStorage(),
        user_profile_manager=manager,
        memory_control_plane=plane,
    )
    identity = RequestIdentity(
        tenant_id="tenant-a",
        user_id="user-1",
        principal_type=PrincipalType.USER,
        auth_subject="user-1",
    )

    result = await session_manager.search_user_longterm_memory(identity, "query", top_k=3)

    plane.recall.assert_awaited_once()
    manager.search_longterm_memory.assert_not_awaited()
    assert result is not None
    assert result["used_longterm"] is True
    assert len(result["hits"]) == 1


@pytest.mark.asyncio
async def test_session_manager_ltm_recall_plane_governance_not_bypassed() -> None:
    plane = _RecordingPlane()
    plane.recall.side_effect = MemoryControlGovernanceDenied(
        "denied",
        decision=MemoryGovernanceDecision(
            outcome=MemoryGovernanceOutcome.DENY,
            reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
            policy_id="test.deny",
            policy_version="1",
            operation=MemoryGovernanceOperation.RECALL,
        ),
    )
    manager = MagicMock(spec=UserProfileManager)
    manager.search_longterm_memory = AsyncMock()

    session_manager = SessionManager(
        InMemorySessionStorage(),
        user_profile_manager=manager,
        memory_control_plane=plane,
    )
    identity = RequestIdentity(
        tenant_id="tenant-a",
        user_id="user-1",
        principal_type=PrincipalType.USER,
        auth_subject="user-1",
    )

    with pytest.raises(MemoryControlGovernanceDenied):
        await session_manager.search_user_longterm_memory(identity, "query")

    manager.search_longterm_memory.assert_not_awaited()
