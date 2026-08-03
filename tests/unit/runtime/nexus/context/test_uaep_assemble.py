# © Artur Czarnecki. All rights reserved.

"""CE-UAEP-ASM: UAEP session assemble helper."""

from __future__ import annotations

import pytest

from intergrax.context.bootstrap import bootstrap_context_catalog, materialize_context_plugin_registry, reset_context_catalog_bootstrap_for_tests
from intergrax.context.session_history import (
    SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON,
    SessionHistorySnapshotRequiredError,
)
from intergrax.llm.messages import ChatMessage, StructuredModelInputRequiredError
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.context.codebase_engine import CodebaseContextEngine
from intergrax.runtime.nexus.context.uaep_assemble import (
    assemble_uaep_session_messages,
    assemble_uaep_session_prompt,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    bootstrap_context_catalog(register_shipped=True, discover_entry_points=False)
    yield
    reset_context_catalog_bootstrap_for_tests()


class _Adapter(LLMAdapter):
    provider = "fake"
    model = "fake"

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


@pytest.mark.asyncio
async def test_assemble_uaep_session_injects_workspace() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = CodebaseContextEngine(registry=registry)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="fix bug",
        tenant_id="t1",
        metadata={
            "workspace_files": {"app.py": "print('hi')\n"},
            "task_id": "task-1",
        },
    )
    prompt = await assemble_uaep_session_prompt(
        request,
        agent_id="agent-1",
        engine=engine,
        llm_adapter=_Adapter(),
    )
    assert "app.py" in prompt or "print" in prompt


@pytest.mark.asyncio
async def test_assemble_uaep_session_includes_full_history_with_revision() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = CodebaseContextEngine(registry=registry)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="follow up",
        tenant_id="t1",
        metadata={
            "task_id": "task-1",
            "session_context_revision_id": "rev-uaep-1",
            "session_history_messages": [
                ChatMessage(role="user", content="first question", entry_id="uaep-u1"),
                ChatMessage(role="assistant", content="first answer", entry_id="uaep-a1"),
            ],
        },
    )
    messages = await assemble_uaep_session_messages(
        request,
        agent_id="agent-1",
        engine=engine,
        llm_adapter=_Adapter(),
    )
    contents = [message.content for message in messages]
    assert "first question" in contents
    assert "first answer" in contents
    with pytest.raises(StructuredModelInputRequiredError):
        await assemble_uaep_session_prompt(
            request,
            agent_id="agent-1",
            engine=engine,
            llm_adapter=_Adapter(),
        )


@pytest.mark.asyncio
async def test_assemble_uaep_session_rejects_raw_history_without_revision() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = CodebaseContextEngine(registry=registry)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="follow up",
        tenant_id="t1",
        metadata={
            "task_id": "task-1",
            "session_history_messages": [
                ChatMessage(role="user", content="orphan", entry_id="uaep-u2"),
            ],
        },
    )
    with pytest.raises(SessionHistorySnapshotRequiredError) as exc_info:
        await assemble_uaep_session_prompt(
            request,
            agent_id="agent-1",
            engine=engine,
            llm_adapter=_Adapter(),
        )
    assert str(exc_info.value) == SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON


@pytest.mark.asyncio
async def test_uaep_message_assembly_preserves_structured_history() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = CodebaseContextEngine(registry=registry)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="follow up",
        tenant_id="t1",
        metadata={
            "task_id": "task-1",
            "session_context_revision_id": "rev-uaep-2",
            "session_history_messages": [
                ChatMessage(role="user", content="history user", entry_id="uaep-u3"),
                ChatMessage(
                    role="assistant",
                    content="assistant",
                    entry_id="uaep-a3",
                    tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
                ),
                ChatMessage(role="tool", content="tool out", entry_id="uaep-t3", tool_call_id="call-1"),
            ],
        },
    )
    messages = await assemble_uaep_session_messages(
        request,
        agent_id="agent-1",
        engine=engine,
        llm_adapter=_Adapter(),
    )
    roles = [message.role for message in messages]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles
    assert messages[-1].role == "user"
    assert messages[-1].content == "follow up"


@pytest.mark.asyncio
async def test_uaep_prompt_projection_rejects_structured_history() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = CodebaseContextEngine(registry=registry)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="follow up",
        tenant_id="t1",
        metadata={
            "task_id": "task-1",
            "session_context_revision_id": "rev-uaep-3",
            "session_history_messages": [
                ChatMessage(role="user", content="history user", entry_id="uaep-u4"),
                ChatMessage(role="assistant", content="assistant", entry_id="uaep-a4"),
            ],
        },
    )
    with pytest.raises(StructuredModelInputRequiredError):
        await assemble_uaep_session_prompt(
            request,
            agent_id="agent-1",
            engine=engine,
            llm_adapter=_Adapter(),
        )


@pytest.mark.asyncio
async def test_uaep_prompt_projection_keeps_simple_context_compatibility() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = CodebaseContextEngine(registry=registry)
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="simple objective",
        tenant_id="t1",
        metadata={"task_id": "task-1", "workspace_files": {"app.py": "print('hi')\n"}},
    )
    prompt = await assemble_uaep_session_prompt(
        request,
        agent_id="agent-1",
        engine=engine,
        llm_adapter=_Adapter(),
    )
    assert "simple objective" in prompt
    assert "role:" not in prompt
