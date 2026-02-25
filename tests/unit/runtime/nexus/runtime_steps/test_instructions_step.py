# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.instructions_step import InstructionsStep
from tests._support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_instructions_step_no_instructions_noop():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.instructions = None
    state.profile_user_instructions = None
    state.profile_org_instructions = None

    before_msgs = list(state.messages_for_llm)

    await InstructionsStep().run(state)

    assert state.messages_for_llm == before_msgs


@pytest.mark.asyncio
async def test_instructions_step_request_only():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.instructions = "REQ"
    state.profile_user_instructions = None
    state.profile_org_instructions = None
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]

    await InstructionsStep().run(state)

    assert state.messages_for_llm[0].role == "system"
    assert state.messages_for_llm[0].content == "REQ"
    assert state.messages_for_llm[1].content == "u1"


@pytest.mark.asyncio
async def test_instructions_step_profile_user_only():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.instructions = None
    state.profile_user_instructions = "USER_PROFILE"
    state.profile_org_instructions = None
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]

    await InstructionsStep().run(state)

    assert state.messages_for_llm[0].content == "USER_PROFILE"


@pytest.mark.asyncio
async def test_instructions_step_profile_org_only():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.instructions = None
    state.profile_user_instructions = None
    state.profile_org_instructions = "ORG_PROFILE"
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]

    await InstructionsStep().run(state)

    assert state.messages_for_llm[0].content == "ORG_PROFILE"


@pytest.mark.asyncio
async def test_instructions_step_combines_all_sources():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.instructions = "REQ"
    state.profile_user_instructions = "USER"
    state.profile_org_instructions = "ORG"
    state.messages_for_llm = [ChatMessage(role="assistant", content="a1")]

    await InstructionsStep().run(state)

    assert state.messages_for_llm[0].content == "REQ\n\nUSER\n\nORG"


@pytest.mark.asyncio
async def test_instructions_step_ignores_empty_strings():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.instructions = "  "
    state.profile_user_instructions = ""
    state.profile_org_instructions = None

    before_msgs = list(state.messages_for_llm)

    await InstructionsStep().run(state)

    assert state.messages_for_llm == before_msgs
