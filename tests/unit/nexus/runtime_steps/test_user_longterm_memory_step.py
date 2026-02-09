# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.user_longterm_memory_step import UserLongtermMemoryStep
from tests._support.builder import build_runtime_state_for_tests


@dataclass
class _FakeBundle:
    context_messages: List[ChatMessage]


class _FakePromptBuilder:
    def __init__(self, bundle: _FakeBundle):
        self.bundle = bundle
        self.called = False

    def build_user_longterm_memory_prompt(self, hits):
        self.called = True
        return self.bundle


class _FakeSessionManager:
    def __init__(self, result):
        self.result = result
        self.called = False

    async def search_user_longterm_memory(self, **kwargs):
        self.called = True
        return self.result


@pytest.mark.asyncio
async def test_ltm_step_noop_when_disabled():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_user_longterm_memory = False

    before_msgs = list(state.messages_for_llm)

    await UserLongtermMemoryStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_user_longterm_memory is False


@pytest.mark.asyncio
async def test_ltm_step_requires_session_for_fallback():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_user_longterm_memory = True
    state.user_longterm_memory_result = None

    with pytest.raises(AssertionError, match=r"Session must be set before user long-term memory step\."):
        await UserLongtermMemoryStep().run(state)


@pytest.mark.asyncio
async def test_ltm_step_empty_query_skips():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_user_longterm_memory = True
    state.session = type("S", (), {"user_id": "u1"})()
    state.request.message = ""

    state.context.session_manager = _FakeSessionManager(result=None)

    before_msgs = list(state.messages_for_llm)

    await UserLongtermMemoryStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_user_longterm_memory is False
    assert state.context.session_manager.called is False


@pytest.mark.asyncio
async def test_ltm_step_no_hits_not_used():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_user_longterm_memory = True
    state.session = type("S", (), {"user_id": "u1"})()
    state.request.message = "hello"

    result = {"hits": [], "debug": {"used": False, "reason": "no_hits"}}
    state.context.session_manager = _FakeSessionManager(result=result)
    state.context.user_longterm_memory_prompt_builder = _FakePromptBuilder(
        _FakeBundle(context_messages=[])
    )

    before_msgs = list(state.messages_for_llm)

    await UserLongtermMemoryStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_user_longterm_memory is False


@pytest.mark.asyncio
async def test_ltm_step_hits_used_injects_context():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="u1"),
    ]

    state.context.config.enable_user_longterm_memory = True
    state.session = type("S", (), {"user_id": "u1"})()
    state.request.message = "query"

    hits = [{"text": "memory"}]
    result = {"hits": hits, "debug": {"used": True}}
    state.context.session_manager = _FakeSessionManager(result=result)

    bundle = _FakeBundle(
        context_messages=[ChatMessage(role="system", content="LTM_CTX")]
    )
    state.context.user_longterm_memory_prompt_builder = _FakePromptBuilder(bundle)

    await UserLongtermMemoryStep().run(state)

    assert state.used_user_longterm_memory is True
    assert [m.content for m in state.messages_for_llm] == [
        "sys",
        "LTM_CTX",
        "u1",
    ]


@pytest.mark.asyncio
async def test_ltm_step_uses_cached_result_without_search():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_user_longterm_memory = True
    state.session = type("S", (), {"user_id": "u1"})()
    state.request.message = "query"

    hits = [{"text": "memory"}]
    state.user_longterm_memory_result = {"hits": hits, "debug": {"used": True}}

    state.context.session_manager = _FakeSessionManager(result=None)
    bundle = _FakeBundle(context_messages=[ChatMessage(role="system", content="LTM_CTX")])
    state.context.user_longterm_memory_prompt_builder = _FakePromptBuilder(bundle)

    await UserLongtermMemoryStep().run(state)

    assert state.context.session_manager.called is False
    assert state.used_user_longterm_memory is True
