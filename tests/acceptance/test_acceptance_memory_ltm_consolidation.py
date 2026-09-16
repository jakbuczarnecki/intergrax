# © Artur Czarnecki. All rights reserved.

"""MEM-4.2: LTM consolidation E2E with deterministic fake LLM."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.runtime.user_profile.session_memory_consolidation_service import (
    SessionMemoryConsolidationConfig,
    SessionMemoryConsolidationService,
)
from intergrax.runtime.user_profile.user_profile_instructions_service import (
    UserProfileInstructionsService,
)
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.gate]


def _deterministic_consolidation_json() -> str:
    return json.dumps(
        {
            "facts": [
                {
                    "title": "Role",
                    "content": "Senior Python engineer",
                    "importance": "HIGH",
                    "tags": ["user"],
                }
            ],
            "preferences": [
                {
                    "title": "Tone",
                    "content": "Concise technical answers in English",
                    "importance": "MEDIUM",
                    "tags": ["communication"],
                }
            ],
            "session_summary": {
                "title": "Session recap",
                "content": "Discussed memory consolidation wiring",
                "importance": "MEDIUM",
                "tags": ["session_summary"],
            },
        }
    )


@pytest.mark.asyncio
async def test_ltm_consolidation_e2e_with_deterministic_fake_llm() -> None:
    tenant_id = "tenant-ltm"
    profile_manager = UserProfileManager(InMemoryUserProfileStore(), tenant_id=tenant_id)
    memory_control_plane = build_default_memory_control_plane(
        user_profile_manager=profile_manager,
    )

    instructions_service = MagicMock(spec=UserProfileInstructionsService)
    instructions_service.build_and_save_system_instructions = AsyncMock(return_value="Be concise.")

    llm = FakeLLMAdapter(fixed_text=_deterministic_consolidation_json())
    service = SessionMemoryConsolidationService(
        llm=llm,
        profile_manager=profile_manager,
        instructions_service=instructions_service,
        memory_control_plane=memory_control_plane,
        config=SessionMemoryConsolidationConfig(
            regenerate_system_instructions=True,
            include_session_summary=True,
        ),
    )

    messages = [
        ChatMessage(role="user", content="I am a senior Python engineer."),
        ChatMessage(role="assistant", content="Noted."),
        ChatMessage(role="user", content="Please keep answers concise and technical."),
    ]

    entries = await service.consolidate_session(
        user_id="user-ltm",
        session_id="sess-ltm-1",
        messages=messages,
        run_id="run-ltm-1",
        tenant_id=tenant_id,
    )

    assert len(entries) == 4
    kinds = {entry.kind for entry in entries}
    assert MemoryKind.USER_FACT in kinds
    assert MemoryKind.PREFERENCE in kinds
    assert MemoryKind.SESSION_SUMMARY in kinds
    assert MemoryKind.EPISODIC_EVENT in kinds
    assert all(entry.session_id == "sess-ltm-1" for entry in entries)
    profile = await profile_manager.get_profile("user-ltm")
    assert len(profile.memory_entries) == 4
    instructions_service.build_and_save_system_instructions.assert_awaited_once()


@pytest.mark.asyncio
async def test_ltm_consolidation_returns_empty_when_llm_output_unparseable() -> None:
    from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile

    profile_manager = MagicMock()
    profile_manager.get_profile = AsyncMock(
        return_value=UserProfile(
            identity=UserIdentity(user_id="user-ltm"),
            preferences=UserPreferences(),
        )
    )
    plane = MagicMock()
    plane.remember = AsyncMock()

    instructions_service = MagicMock(spec=UserProfileInstructionsService)
    instructions_service.build_and_save_system_instructions = AsyncMock()

    service = SessionMemoryConsolidationService(
        llm=FakeLLMAdapter(fixed_text="not-json"),
        profile_manager=profile_manager,
        instructions_service=instructions_service,
        memory_control_plane=plane,
    )

    entries = await service.consolidate_session(
        user_id="user-ltm",
        session_id="sess-ltm-2",
        messages=[ChatMessage(role="user", content="hello")],
        tenant_id="tenant-ltm",
    )

    assert entries == []
    plane.remember.assert_not_awaited()
    instructions_service.build_and_save_system_instructions.assert_not_awaited()
