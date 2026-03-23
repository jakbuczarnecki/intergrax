# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.websearch_step import WebsearchStep
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


@dataclass
class _FakeBundle:
    context_messages: List[ChatMessage]
    no_evidence: bool
    sources_count: int


class _FakeExecutor:
    def __init__(self, results):
        self.results = results
        self.called = False

    async def search_async(self, **kwargs):
        self.called = True
        return self.results


class _FakePromptBuilder:
    def __init__(self, bundle: _FakeBundle):
        self.bundle = bundle
        self.called = False

    async def build_websearch_prompt(self, **kwargs):
        self.called = True
        return self.bundle


@pytest.mark.asyncio
async def test_websearch_step_noop_when_disabled():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_websearch = False

    before_msgs = list(state.messages_for_llm)

    await WebsearchStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_websearch is False


@pytest.mark.asyncio
async def test_websearch_step_skips_when_not_configured():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_websearch = True
    state.context.websearch_executor = None
    state.context.websearch_prompt_builder = None

    before_msgs = list(state.messages_for_llm)

    await WebsearchStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_websearch is False


@pytest.mark.asyncio
async def test_websearch_step_no_results():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_websearch = True

    state.context.websearch_executor = _FakeExecutor(results=[])
    state.context.websearch_prompt_builder = _FakePromptBuilder(
        _FakeBundle(context_messages=[], no_evidence=True, sources_count=0)
    )

    before_msgs = list(state.messages_for_llm)

    await WebsearchStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_websearch is False
    assert state.context.websearch_executor.called is True


@pytest.mark.asyncio
async def test_websearch_step_injects_context():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="u1"),
    ]

    state.context.config.enable_websearch = True

    state.context.websearch_executor = _FakeExecutor(results=[object()])
    bundle = _FakeBundle(
        context_messages=[ChatMessage(role="system", content="WEB_CTX")],
        no_evidence=False,
        sources_count=1,
    )
    state.context.websearch_prompt_builder = _FakePromptBuilder(bundle)

    await WebsearchStep().run(state)

    assert state.used_websearch is True
    assert [m.content for m in state.messages_for_llm] == [
        "sys",
        "WEB_CTX",
        "u1",
    ]


@pytest.mark.asyncio
async def test_websearch_step_bundle_without_context_messages():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_websearch = True

    state.context.websearch_executor = _FakeExecutor(results=[object()])
    state.context.websearch_prompt_builder = _FakePromptBuilder(
        _FakeBundle(context_messages=[], no_evidence=True, sources_count=0)
    )

    before_msgs = list(state.messages_for_llm)

    await WebsearchStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_websearch is False


@pytest.mark.asyncio
async def test_websearch_step_executor_error():
    class _BadExecutor:
        async def search_async(self, **kwargs):
            raise RuntimeError("boom")

    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.config.enable_websearch = True
    state.context.websearch_executor = _BadExecutor()
    state.context.websearch_prompt_builder = _FakePromptBuilder(
        _FakeBundle(context_messages=[], no_evidence=True, sources_count=0)
    )

    before_msgs = list(state.messages_for_llm)

    await WebsearchStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_websearch is False
