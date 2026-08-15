# © Artur Czarnecki. All rights reserved.

"""MEM-4.1: Session persist + resume via memory_wiring on SQLite."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.memory_wiring import (
    build_session_manager_from_environment,
    resolve_memory_platform_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def _sqlite_lab_env(tmp_path: Path) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.session.roundtrip")
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    return env


@pytest.mark.asyncio
async def test_session_sqlite_persist_and_resume_via_memory_wiring(tmp_path: Path) -> None:
    env = _sqlite_lab_env(tmp_path)
    wiring = resolve_memory_platform_wiring(env)
    session_manager = build_session_manager_from_environment(
        env,
        memory_wiring=wiring,
        tenant_id="tenant-lab",
    )

    assert isinstance(wiring.session_storage, SQLiteSessionStorage)

    session = await session_manager.create_session(
        tenant_id="tenant-lab",
        session_id="sess-roundtrip-1",
        user_id="user-1",
        workspace_id="ws-1",
    )
    await session_manager.append_message(
        tenant_id="tenant-lab",
        session_id=session.id,
        message=ChatMessage(role="user", content="first turn"),
    )
    await session_manager.append_message(
        tenant_id="tenant-lab",
        session_id=session.id,
        message=ChatMessage(role="assistant", content="acknowledged"),
    )

    resumed_wiring = resolve_memory_platform_wiring(env)
    resumed_manager = build_session_manager_from_environment(
        env,
        memory_wiring=resumed_wiring,
        tenant_id="tenant-lab",
    )

    loaded_session = await resumed_manager.get_session(
        tenant_id="tenant-lab",
        session_id=session.id,
    )
    history = await resumed_manager.get_history_for_session(
        tenant_id="tenant-lab",
        session_id=session.id,
    )

    assert loaded_session is not None
    assert loaded_session.user_id == "user-1"
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "first turn"
    assert history[1].role == "assistant"
    assert history[1].content == "acknowledged"


@pytest.mark.asyncio
async def test_get_or_create_session_resumes_existing_sqlite_session(tmp_path: Path) -> None:
    env = _sqlite_lab_env(tmp_path)
    wiring = resolve_memory_platform_wiring(env)
    session_manager = build_session_manager_from_environment(
        env,
        memory_wiring=wiring,
        tenant_id="tenant-lab",
    )

    created = await session_manager.get_or_create_session(
        tenant_id="tenant-lab",
        user_id="user-2",
        session_id="sess-resume-2",
        workspace_id="ws-1",
    )
    await session_manager.append_message(
        tenant_id="tenant-lab",
        session_id=created.id,
        message=ChatMessage(role="user", content="resume probe"),
    )

    resumed_manager = build_session_manager_from_environment(
        env,
        memory_wiring=resolve_memory_platform_wiring(env),
        tenant_id="tenant-lab",
    )
    resumed = await resumed_manager.get_or_create_session(
        tenant_id="tenant-lab",
        user_id="user-2",
        session_id="sess-resume-2",
        workspace_id="ws-1",
    )
    history = await resumed_manager.get_history_for_session(
        tenant_id="tenant-lab",
        session_id=resumed.id,
    )

    assert resumed.id == created.id
    assert len(history) == 1
    assert history[0].content == "resume probe"
