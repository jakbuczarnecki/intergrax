# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-1: SessionManager → consolidation → tenant-scoped user profile memory."""

from __future__ import annotations

import json

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.user_profile.session_memory_consolidation_service import (
    SessionMemoryConsolidationConfig,
    SessionMemoryConsolidationService,
)
from intergrax.runtime.user_profile.user_profile_instructions_service import (
    UserProfileInstructionsService,
)
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.gate]


def _consolidation_payload() -> str:
    return json.dumps(
        {
            "facts": [
                {
                    "title": "Role",
                    "content": "Tenant A engineer",
                    "importance": "HIGH",
                    "tags": ["user"],
                }
            ],
            "preferences": [],
            "session_summary": None,
        }
    )


def _build_manager(
    *,
    tenant_id: str,
    profile_store: InMemoryUserProfileStore,
    consolidation_interval: int,
) -> SessionManager:
    profile_manager = UserProfileManager(profile_store, tenant_id=tenant_id)
    instructions = UserProfileInstructionsService(
        llm=FakeLLMAdapter(fixed_text="instructions"),
        manager=profile_manager,
    )
    consolidation_service = SessionMemoryConsolidationService(
        llm=FakeLLMAdapter(fixed_text=_consolidation_payload()),
        profile_manager=profile_manager,
        instructions_service=instructions,
        config=SessionMemoryConsolidationConfig(
            regenerate_system_instructions=False,
            include_session_summary=False,
        ),
    )
    return SessionManager(
        storage=InMemorySessionStorage(),
        user_profile_manager=profile_manager,
        session_memory_consolidation_service=consolidation_service,
        user_turns_consolidation_interval=consolidation_interval,
        consolidation_cooldown_seconds=0,
        memory_consolidation_mode="auto",
    )


@pytest.mark.asyncio
async def test_session_manager_mid_and_close_consolidation_respect_tenant_scope() -> None:
    profile_store = InMemoryUserProfileStore()
    tenant_a = "tenant-a"
    user_id = "shared-user"
    session_id = "sess-consolidate-1"

    manager_a = _build_manager(
        tenant_id=tenant_a,
        profile_store=profile_store,
        consolidation_interval=2,
    )
    await manager_a.create_session(
        tenant_id=tenant_a,
        session_id=session_id,
        user_id=user_id,
    )
    await manager_a.append_message(
        tenant_id=tenant_a,
        session_id=session_id,
        message=ChatMessage(role="user", content="I work on tenant A systems."),
    )
    await manager_a.append_message(
        tenant_id=tenant_a,
        session_id=session_id,
        message=ChatMessage(role="user", content="Remember my tenant A role."),
    )
    profile_a = await profile_store.get_profile(tenant_id=tenant_a, user_id=user_id)
    assert len(profile_a.memory_entries) >= 1

    await manager_a.close_session(tenant_id=tenant_a, session_id=session_id)
    profile_a_after = await profile_store.get_profile(tenant_id=tenant_a, user_id=user_id)
    assert len(profile_a_after.memory_entries) >= 1

    profile_b = await profile_store.get_profile(tenant_id="tenant-b", user_id=user_id)
    assert profile_b.memory_entries == []
