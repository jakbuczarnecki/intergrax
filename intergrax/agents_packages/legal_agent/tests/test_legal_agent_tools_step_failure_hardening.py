# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Enterprise hardening: ``ToolsStep`` hits an unexpected exception inside its try-block
(e.g. planner failure). Nexus must record WARNING/ERROR on the ``tools`` trace step and
the legal agent must still return a normal ``RuntimeAnswer`` with a non-empty answer.

Persisted trace payloads are **redacted** by ``RuntimeState.trace_event`` (see
``ToolsSummaryDiagV1.redact``): expect ``error_type`` but not the raw exception text.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents_packages.legal_agent.legal_agent import LegalAgent
from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.tool_decision_component import (
    decide_legal_tool_plan,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_pipeline,
)
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import (
    create_default_vectorstore_manager,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.tools.tools_summary import ToolsSummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, TraceLevel
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tools_agent import ToolsAgent

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e

_FORCED_TOOLS_ERROR = "enterprise-forced-tools-step-failure"


def _tools_trace_events(answer: RuntimeAnswer) -> list:
    return [e for e in (answer.trace_events or []) if e.step == "tools"]


@pytest.mark.asyncio
async def test_legal_agent_completes_when_tools_step_planner_raises(
    tmp_path: Path,
) -> None:
    require_ollama_reachable()
    _ = tmp_path

    tenant_id = "legal-tools-step-failure"
    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)
    session_manager = SessionManager(storage=InMemorySessionStorage())
    tools_agent = ToolsAgent(llm=llm_adapter, tools=ToolRegistry())

    embedding_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(provider_id="ollama"),
    )
    vectorstore_manager = create_default_vectorstore_manager(tenant_id=tenant_id)

    cfg = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        production_mode=False,
        enable_rag=True,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        use_legal_tool_decision=True,
        tools_agent=tools_agent,
        tools_mode="auto",
        tool_providers=[],
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
    )

    request = RuntimeRequest(
        agent_id="legal-agent-tools-fail",
        user_id="user-tools-fail",
        session_id="session-tools-fail",
        message="Summarize late-payment interest rules in the EU at a high level.",
        attachments=[],
        tenant_id=tenant_id,
        workspace_id="ws-tools-fail",
    )

    agent = LegalAgent(config=cfg)

    async def _decide_then_force_tools(**kwargs: object) -> object:
        plan = await decide_legal_tool_plan(**kwargs)  # type: ignore[misc]
        return plan.model_copy(
            update={
                "use_tools": True,
                "use_rag": False,
                "use_websearch": False,
                "intent": "tools",
            }
        )

    with (
        patch(
            "intergrax.agents_packages.legal_agent.legal_execution_loop.decide_legal_tool_plan",
            side_effect=_decide_then_force_tools,
        ),
        patch.object(
            ToolsAgent,
            "plan_tools",
            side_effect=RuntimeError(_FORCED_TOOLS_ERROR),
        ),
    ):
        result = await AgentEngine.run_agent(agent, request)

    assert result is not None
    assert result.answer is not None
    assert result.answer.strip() != ""

    tools_events = _tools_trace_events(result)
    assert tools_events, "expected at least one Nexus tools trace event"
    assert any(
        e.level in (TraceLevel.WARNING, TraceLevel.ERROR) for e in tools_events
    ), f"expected tools trace WARNING or ERROR, got: {[e.level for e in tools_events]!r}"

    summary_events = [
        e for e in tools_events if isinstance(e.payload, ToolsSummaryDiagV1)
    ]
    assert summary_events
    payload = summary_events[-1].payload
    assert payload is not None
    d = payload.to_dict()
    assert d.get("error_type") == "RuntimeError"
    assert d.get("error_message") == DEFAULT_REDACTED_TEXT, (
        "Client-visible trace must redact tool error_message while still recording failure."
    )
