# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications._shared.session_tool_wiring import wire_session_storage_tool_binding
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.tools.providers.interaction.contracts import InteractionListSessionsInput
from intergrax.tools.providers.interaction.service import interaction_list_sessions
from intergrax.tools.registry.runtime_bindings import SessionStorageBinding
from intergrax.tools.registry.session_storage_binding import SessionStorageToolBinding
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_session_storage_tool_binding_lists_sessions_and_last_input() -> None:
    storage = InMemorySessionStorage()
    session = await storage.create_session(tenant_id="tenant-a", user_id="user-1")
    await storage.append_message(
        tenant_id="tenant-a",
        session_id=session.id,
        message=ChatMessage(role="user", content="hello from session"),
    )

    binding = SessionStorageToolBinding(storage)
    listed = binding.list_sessions("tenant-a", "user-1", limit=5)
    assert len(listed) == 1
    assert listed[0]["session_id"] == session.id

    last = binding.get_last_user_input("tenant-a", session.id)
    assert last == "hello from session"


def test_wire_session_storage_tool_binding_attaches_binding() -> None:
    storage = InMemorySessionStorage()
    ctx = wire_session_storage_tool_binding(ToolWiringContext(), storage)
    assert isinstance(ctx.session_storage, SessionStorageToolBinding)
    assert isinstance(ctx.session_storage, SessionStorageBinding)


def test_environment_memory_wiring_exposes_session_storage_binding() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="session.tool.binding")
    memory_wiring = resolve_memory_platform_wiring(env, integration_profile=IntegrationProfile.lab())
    ctx = wire_session_storage_tool_binding(ToolWiringContext(), memory_wiring.session_storage)
    assert ctx.session_storage is not None

    listed = interaction_list_sessions(
        ctx,
        InteractionListSessionsInput(tenant_id="default", user_id="missing-user"),
    )
    assert listed.used is True
    assert listed.total == 0
