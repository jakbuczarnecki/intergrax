# © Artur Czarnecki. All rights reserved.

"""B-P6-R items 8 and 10 - tool output spills to an artifact reference in the tool path, scrubbed first."""

from __future__ import annotations

import random
import string
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.spill import ToolOutputSpiller
from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from intergrax.runtime.nexus.tools.plan_context_invocation import run_tools_context
from testing_support.builder import build_runtime_state_for_tests, canonical_governed_execution_scope

pytestmark = pytest.mark.unit

_TENANT = "test-tenant"
_BIG_ROW = "order,1,shipped,tabular tool output line\n"


def _big() -> str:
    return _BIG_ROW * 600  # ~24 KB, above the 16 KiB default threshold


def _credential_literal() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "AK" + "IA" + "".join(random.SystemRandom().choice(alphabet) for _ in range(16))


def _trace(output: str) -> ToolCallTrace:
    return ToolCallTrace(
        tool_name="sql.query", arguments={}, output_preview=output, success=True, error_message=None, raw_trace={}
    )


async def _run(state, loop_result) -> None:  # type: ignore[no-untyped-def]
    state.context.config.tools_mode = "auto"
    state.context.config.tool_invoker = object()
    state.context.config.tool_planner = object()
    state.messages_for_llm = [ChatMessage(role="user", content="use tool")]
    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_planner_input",
            return_value=[ChatMessage(role="user", content="use tool")],
        ),
        patch("intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_registry", return_value=None),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_bounded_tool_loop_async",
            return_value=loop_result,
        ),
        canonical_governed_execution_scope("run_p6r_spill"),
    ):
        await run_tools_context(state)


def _loop_result(traces: list[ToolCallTrace], appended: list[ChatMessage] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        pattern_id="p",
        stop_reason="done",
        tool_traces=traces,
        appended_messages=appended or [],
        used_ce_tool_feedback=bool(appended),
        aggregate=None,
    )


@pytest.mark.asyncio
async def test_large_tool_output_reaches_the_context_as_a_reference_not_as_text() -> None:
    state = build_runtime_state_for_tests(run_id="run_p6r_spill")
    store = InMemoryArtifactStore()
    state.context.artifact_store = store

    await _run(state, _loop_result([_trace(_big())]))

    joined = "\n".join(state.tools_context_parts)
    assert "artifact://" in joined
    assert _big() not in joined
    stored = list(store.list_for_run(_TENANT, state.run_id))
    assert len(stored) == 1
    assert _big().strip().encode("utf-8") in stored[0].data  # the aggregated tool context is what is stored


@pytest.mark.asyncio
async def test_large_tool_feedback_message_is_replaced_by_a_reference() -> None:
    state = build_runtime_state_for_tests(run_id="run_p6r_spill")
    store = InMemoryArtifactStore()
    state.context.artifact_store = store
    message = ChatMessage(role="tool", content=_big(), tool_call_id="call-1", name="sql.query")

    await _run(state, _loop_result([_trace("x")], appended=[message]))

    fed = [m for m in state.messages_for_llm if m.role == "tool"]
    assert len(fed) == 1
    assert "artifact://" in fed[0].content and _big() not in fed[0].content
    assert fed[0].tool_call_id == "call-1" and fed[0].name == "sql.query"


@pytest.mark.asyncio
async def test_without_a_store_or_for_small_output_behavior_is_unchanged() -> None:
    no_store = build_runtime_state_for_tests(run_id="run_p6r_spill")
    assert no_store.context.artifact_store is None
    await _run(no_store, _loop_result([_trace(_big())]))
    assert "artifact://" not in "\n".join(no_store.tools_context_parts)
    assert _big().strip() in "\n".join(no_store.tools_context_parts)

    small = build_runtime_state_for_tests(run_id="run_p6r_spill")
    small.context.artifact_store = InMemoryArtifactStore()
    await _run(small, _loop_result([_trace("small output")]))
    assert "small output" in "\n".join(small.tools_context_parts)
    assert "artifact://" not in "\n".join(small.tools_context_parts)
    assert list(small.context.artifact_store.list_for_run(_TENANT, small.run_id)) == []


# --------------------------------------------------------------------------- item 10: scrub before put


def _spill(content: str | bytes):  # type: ignore[no-untyped-def]
    store = InMemoryArtifactStore()
    spill = ToolOutputSpiller(store, threshold_bytes=1024).spill_if_large(
        tenant_id=_TENANT, run_id="run-1", tool_id="vendor.read", agent_id="agent-1", content=content
    )
    return store, spill


def test_a_credential_format_literal_is_not_persisted_verbatim() -> None:
    literal = _credential_literal()
    store, spill = _spill(_big() + f"debug: authorization={literal}\n")
    assert spill.reference is not None
    data = store.get(_TENANT, spill.reference.artifact_id).data
    assert literal.encode() not in data
    assert b"[REDACTED]" in data
    assert literal not in spill.preview
    assert data.startswith(_BIG_ROW.encode())


def test_a_credential_in_the_preview_window_is_scrubbed_too() -> None:
    literal = _credential_literal()
    _, spill = _spill(f"head {literal} tail\n" + _big())
    assert spill.reference is not None
    assert literal not in spill.preview


def test_binary_output_with_a_credential_is_scrubbed_and_other_bytes_are_preserved() -> None:
    literal = _credential_literal()
    payload = b"\xff\xfe\x00binary-prefix " + literal.encode() + b" " + bytes(range(256)) * 8
    store, spill = _spill(payload)
    assert spill.reference is not None
    data = store.get(_TENANT, spill.reference.artifact_id).data
    assert literal.encode() not in data
    assert data.startswith(b"\xff\xfe\x00binary-prefix ")
    assert data.endswith(bytes(range(256)) * 8)


def test_output_without_a_credential_is_stored_byte_identical() -> None:
    content = _big().encode("utf-8")
    store, spill = _spill(content)
    assert spill.reference is not None
    stored = store.get(_TENANT, spill.reference.artifact_id)
    assert stored.data == content
    assert spill.reference.content_digest == stored.content_digest
