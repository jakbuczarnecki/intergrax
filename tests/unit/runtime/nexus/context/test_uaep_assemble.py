# © Artur Czarnecki. All rights reserved.

"""CE-UAEP-ASM: UAEP session assemble helper."""

from __future__ import annotations

import pytest

from intergrax.context.bootstrap import bootstrap_context_catalog, materialize_context_plugin_registry, reset_context_catalog_bootstrap_for_tests
from intergrax.context.session_history import (
    SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON,
    SessionHistorySnapshotRequiredError,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.context.codebase_engine import CodebaseContextEngine
from intergrax.runtime.nexus.context.uaep_assemble import assemble_uaep_session_prompt
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
    prompt = await assemble_uaep_session_prompt(
        request,
        agent_id="agent-1",
        engine=engine,
        llm_adapter=_Adapter(),
    )
    assert "first question" in prompt
    assert "first answer" in prompt


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
