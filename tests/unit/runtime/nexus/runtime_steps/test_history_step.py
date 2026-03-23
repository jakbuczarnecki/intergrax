# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.history_step import HistoryStep
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


@dataclass
class _Built:
    history_messages: List[ChatMessage]


class _FakeContextBuilder:
    def __init__(self, built: _Built):
        self.built = built
        self.called = False

    async def build_context(self, **kwargs):
        self.called = True
        return self.built


@pytest.mark.asyncio
async def test_history_step_requires_session():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = None

    with pytest.raises(AssertionError, match=r"Session must be set before history step\."):
        await HistoryStep().run(state)


@pytest.mark.asyncio
async def test_history_step_with_context_builder():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = object()

    built_msgs = [
        ChatMessage(role="user", content="h1"),
        ChatMessage(role="assistant", content="h2"),
    ]
    built = _Built(history_messages=built_msgs)
    builder = _FakeContextBuilder(built)

    state.context.context_builder = builder
    state.base_history = [ChatMessage(role="assistant", content="base")]

    before_len = len(state.messages_for_llm)

    await HistoryStep().run(state)

    assert builder.called is True
    assert state.context_builder_result is built
    assert state.built_history_messages == built_msgs
    assert state.history_includes_current_user is True
    assert state.messages_for_llm[before_len:] == built_msgs


@pytest.mark.asyncio
async def test_history_step_without_context_builder_uses_base_history():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = object()

    base_history = [
        ChatMessage(role="assistant", content="base1"),
        ChatMessage(role="assistant", content="base2"),
    ]
    state.base_history = base_history
    state.context.context_builder = None

    before_len = len(state.messages_for_llm)

    await HistoryStep().run(state)

    assert state.context_builder_result is None
    assert state.built_history_messages == base_history
    assert state.history_includes_current_user is True
    assert state.messages_for_llm[before_len:] == base_history


@pytest.mark.asyncio
async def test_history_step_handles_empty_history_from_builder():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = object()

    built = _Built(history_messages=None)
    state.context.context_builder = _FakeContextBuilder(built)

    before_len = len(state.messages_for_llm)

    await HistoryStep().run(state)

    assert state.built_history_messages == []
    assert state.messages_for_llm[before_len:] == []
    assert state.history_includes_current_user is True
