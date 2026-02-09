# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.ensure_current_user_message_step import EnsureCurrentUserMessageStep
from tests._support.builder import build_runtime_state_for_tests


@pytest.mark.asyncio
async def test_ensure_user_step_noop_when_request_empty():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.message = "   "

    before = list(state.messages_for_llm)

    await EnsureCurrentUserMessageStep().run(state)

    assert state.messages_for_llm == before


@pytest.mark.asyncio
async def test_ensure_user_step_adds_when_messages_empty():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.message = "hello"
    state.messages_for_llm = []

    await EnsureCurrentUserMessageStep().run(state)

    assert len(state.messages_for_llm) == 1
    assert state.messages_for_llm[0].role == "user"
    assert state.messages_for_llm[0].content == "hello"


@pytest.mark.asyncio
async def test_ensure_user_step_noop_when_last_equals_current():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.message = "hello"
    state.messages_for_llm = [
        ChatMessage(role="user", content="hello")
    ]

    before = list(state.messages_for_llm)

    await EnsureCurrentUserMessageStep().run(state)

    assert state.messages_for_llm == before


@pytest.mark.asyncio
async def test_ensure_user_step_appends_when_last_different_user():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.message = "hello"
    state.messages_for_llm = [
        ChatMessage(role="user", content="old")
    ]

    await EnsureCurrentUserMessageStep().run(state)

    assert [m.content for m in state.messages_for_llm] == ["old", "hello"]


@pytest.mark.asyncio
async def test_ensure_user_step_appends_when_last_not_user():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.message = "hello"
    state.messages_for_llm = [
        ChatMessage(role="assistant", content="a1")
    ]

    await EnsureCurrentUserMessageStep().run(state)

    assert state.messages_for_llm[-1].role == "user"
    assert state.messages_for_llm[-1].content == "hello"


@pytest.mark.asyncio
async def test_ensure_user_step_trims_content_before_compare():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.message = "hello "
    state.messages_for_llm = [
        ChatMessage(role="user", content="  hello")
    ]

    before = list(state.messages_for_llm)

    await EnsureCurrentUserMessageStep().run(state)

    assert state.messages_for_llm == before
