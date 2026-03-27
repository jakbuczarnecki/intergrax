# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
E2E: :class:`CallableLegalToolPlanGovernance` on a real Legal Agent run (Ollama, RAG stack,
``decide_legal_tool_plan``), proving the port runs **inside** ``legal_execution_loop`` and can
strip Nexus RAG before the bridge.

Uses the same patch pattern as ``test_legal_agent_empty_rag`` to force ``use_rag=True`` from
the tool-decision path, then dynamic governance turns RAG off — expect **no** ``rag`` step in trace.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents_packages.legal_agent.legal_agent import LegalAgent
from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.domain.legal_tool_plan import LegalToolPlan
from intergrax.agents_packages.legal_agent.governance.legal_tool_plan_governance_impl import (
    CallableLegalToolPlanGovernance,
)
from intergrax.agents_packages.legal_agent.runtime.tool_decision_component import (
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
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e


def _trace_steps(answer: RuntimeAnswer) -> list[str]:
    return [e.step for e in (answer.trace_events or [])]


def _dynamic_strip_rag(
    plan: LegalToolPlan,
    state: RuntimeState,
    legal_config: LegalAgentConfig,
) -> LegalToolPlan:
    """Production-style hook: degrade RAG for this scenario (deterministic test policy)."""
    _ = state
    _ = legal_config
    if not plan.use_rag:
        return plan
    return plan.model_copy(
        update={
            "use_rag": False,
            "intent": "llm_only",
        },
    )


@pytest.mark.asyncio
async def test_legal_agent_callable_governance_port_strips_rag_before_bridge_e2e(
    tmp_path: Path,
) -> None:
    require_ollama_reachable()
    _ = tmp_path

    tenant_id = "legal-governance-port-e2e"
    embedding_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(provider_id="ollama"),
    )
    vectorstore_manager = create_default_vectorstore_manager(tenant_id=tenant_id)
    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)
    session_manager = SessionManager(storage=InMemorySessionStorage())

    cfg = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        production_mode=False,
        enable_rag=True,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        use_legal_tool_decision=True,
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
        legal_tool_plan_governance=CallableLegalToolPlanGovernance(_dynamic_strip_rag),
    )

    request = RuntimeRequest(
        agent_id="legal-agent-gov-port-e2e",
        user_id="user-gov-e2e",
        session_id="session-gov-e2e",
        message=(
            "Summarize GDPR fine ceilings in the EU. "
            "Do not claim internal retrieval if you did not run RAG."
        ),
        attachments=[],
        tenant_id=tenant_id,
        workspace_id="ws-gov-e2e",
    )

    agent = LegalAgent(config=cfg)

    async def _decide_then_force_rag(**kwargs: object) -> object:
        plan = await decide_legal_tool_plan(**kwargs)  # type: ignore[misc]
        return plan.model_copy(
            update={
                "use_rag": True,
                "use_tools": False,
                "use_websearch": False,
                "intent": "rag",
            }
        )

    with patch(
        "intergrax.agents_packages.legal_agent.pipeline.legal_execution_loop.decide_legal_tool_plan",
        side_effect=_decide_then_force_rag,
    ):
        result = await AgentEngine.run_agent(agent, request)

    assert result is not None
    assert result.answer is not None
    assert result.answer.strip() != ""

    steps = _trace_steps(result)
    assert "LegalToolDecision" in steps
    assert "rag" not in steps, (
        "Callable governance should clear use_rag before the bridge; Nexus RagStep must not run."
    )

    errors = [e for e in result.trace_events if e.level == TraceLevel.ERROR]
    assert not errors, f"unexpected ERROR trace events: {[e.message for e in errors]}"
