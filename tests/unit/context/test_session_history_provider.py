# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 session history provider tests."""

from __future__ import annotations

import inspect

import pytest

from intergrax.context.providers.builtin import _collect_session_history
from intergrax.context.providers.legacy_bridge import SESSION_HISTORY_MESSAGES_HANDLE
from intergrax.context.session_history import (
    SESSION_HISTORY_SNAPSHOT_HANDLE,
    HandleSessionHistoryProvider,
    build_session_history_snapshot,
)
from intergrax.context.contracts import ContextAssemblyRequest, ContextBudgetSnapshot, ContextDecisionSnapshot, ContextProviderContext
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(include_session_history=True),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )


@pytest.mark.asyncio
async def test_canonical_provider_returns_full_snapshot_fragments() -> None:
    messages = [
        ChatMessage(role="user", content=f"turn-{index}", entry_id=f"m{index}")
        for index in range(12)
    ]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=messages,
    )
    ctx = ContextProviderContext(handles={SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot})
    fragments = await _collect_session_history(_request(), ctx)
    assert len(fragments) == 12
    assert fragments[0].metadata["message_id"] == "m0"
    assert not fragments[0].content.startswith("user:")


@pytest.mark.asyncio
async def test_missing_handle_returns_empty() -> None:
    fragments = await _collect_session_history(_request(), ContextProviderContext())
    assert fragments == []


@pytest.mark.asyncio
async def test_include_session_history_false_returns_none() -> None:
    provider = HandleSessionHistoryProvider()
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="x", entry_id="m1")],
    )
    request = ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(include_session_history=False),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    assert await provider.load_snapshot(request, ContextProviderContext(handles={SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot})) is None


def test_canonical_provider_has_no_last_n_slicing() -> None:
    source = inspect.getsource(_collect_session_history)
    assert "[-max_entries:]" not in source
    assert "[-N:]" not in source
    assert "fragments_from_session_history_snapshot(snapshot)" in source


@pytest.mark.asyncio
async def test_snapshot_has_priority_when_snapshot_and_legacy_handles_both_exist() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="canonical", entry_id="m1")],
    )
    ctx = ContextProviderContext(
        handles={
            SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot,
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy", entry_id="legacy-1"),
            ],
        }
    )
    fragments = await _collect_session_history(_request(), ctx)
    assert len(fragments) == 1
    assert fragments[0].metadata["message_id"] == "m1"
    assert "legacy" not in fragments[0].content


def _request_include_history(include_history: bool) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(include_session_history=include_history),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )


@pytest.mark.asyncio
async def test_include_history_false_legacy_handle_returns_empty() -> None:
    ctx = ContextProviderContext(
        handles={
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy", entry_id="legacy-1"),
            ],
        }
    )
    fragments = await _collect_session_history(_request_include_history(False), ctx)
    assert fragments == []


@pytest.mark.asyncio
async def test_include_history_false_snapshot_handle_returns_empty() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="canonical", entry_id="m1")],
    )
    ctx = ContextProviderContext(handles={SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot})
    fragments = await _collect_session_history(_request_include_history(False), ctx)
    assert fragments == []


@pytest.mark.asyncio
async def test_include_history_false_both_handles_returns_empty() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="canonical", entry_id="m1")],
    )
    ctx = ContextProviderContext(
        handles={
            SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot,
            SESSION_HISTORY_MESSAGES_HANDLE: [
                ChatMessage(role="user", content="legacy", entry_id="legacy-1"),
            ],
        }
    )
    fragments = await _collect_session_history(_request_include_history(False), ctx)
    assert fragments == []
