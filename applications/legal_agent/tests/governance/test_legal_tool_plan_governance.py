# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
E2E: static organization clamp (:func:`enforce_legal_tool_plan_governance`) on full Legal Agent runs.

Uses real Ollama, RAG stack, and ``AgentEngine``. Patches ``decide_legal_tool_plan`` only to make
layer requests deterministic (same pattern as ``test_legal_agent_empty_rag``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agents.agent_engine import AgentEngine
from legal_agent.legal_agent import LegalAgent
from legal_agent.config.legal_agent_config import LegalAgentConfig
from legal_agent.runtime.tool_decision_component import (
    decide_legal_tool_plan,
)
from legal_agent.tracing.legal_tool_plan_governance_clamp_diag_v1 import (
    LegalToolPlanGovernanceClampDiagV1,
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
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tools_agent import ToolsAgent

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e


def _trace_steps(answer: RuntimeAnswer) -> list[str]:
    return [e.step for e in (answer.trace_events or [])]


def _governance_events(answer: RuntimeAnswer) -> list:
    return [e for e in (answer.trace_events or []) if e.step == "LegalToolPlanGovernance"]


def _base_legal_cfg(
    *,
    tenant_id: str,
    organization_allow_rag: bool = True,
    organization_allow_websearch: bool = True,
    organization_allow_tools: bool = True,
    with_tools: bool = False,
) -> LegalAgentConfig:
    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)
    session_manager = SessionManager(storage=InMemorySessionStorage())
    embedding_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(provider_id="ollama"),
    )
    vectorstore_manager = create_default_vectorstore_manager(tenant_id=tenant_id)

    tools_agent = None
    tools_mode: str = "off"
    if with_tools:
        tools_agent = ToolsAgent(llm=llm_adapter, tools=ToolRegistry())
        tools_mode = "auto"

    return LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        production_mode=False,
        enable_rag=True,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        use_legal_tool_decision=True,
        tools_agent=tools_agent,
        tools_mode=tools_mode,
        tool_providers=[],
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
        organization_allow_rag=organization_allow_rag,
        organization_allow_websearch=organization_allow_websearch,
        organization_allow_tools=organization_allow_tools,
    )


@pytest.mark.asyncio
async def test_legal_agent_organization_governance_no_clamp_when_rag_allowed_e2e(
    tmp_path: Path,
) -> None:
    require_ollama_reachable()
    _ = tmp_path

    tenant_id = "legal-org-gov-allow-e2e"
    cfg = _base_legal_cfg(tenant_id=tenant_id)
    request = RuntimeRequest(
        agent_id="legal-org-gov-allow",
        user_id="u1",
        session_id="s1",
        message="High-level GDPR administrative fines overview.",
        attachments=[],
        tenant_id=tenant_id,
        workspace_id="ws1",
    )
    agent = LegalAgent(config=cfg)

    async def _decide_force_rag(**kwargs: object) -> object:
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
        "legal_agent.pipeline.legal_execution_loop.decide_legal_tool_plan",
        side_effect=_decide_force_rag,
    ):
        result = await AgentEngine.run_agent(agent, request)

    assert result.answer and result.answer.strip()
    assert "LegalToolDecision" in _trace_steps(result)
    assert "rag" in _trace_steps(result)
    assert not _governance_events(result)


@pytest.mark.asyncio
async def test_legal_agent_organization_governance_clamps_rag_trace_and_skips_nexus_rag_e2e(
    tmp_path: Path,
) -> None:
    require_ollama_reachable()
    _ = tmp_path

    tenant_id = "legal-org-gov-clamp-rag-e2e"
    cfg = _base_legal_cfg(tenant_id=tenant_id, organization_allow_rag=False)
    request = RuntimeRequest(
        agent_id="legal-org-gov-clamp-rag",
        user_id="u2",
        session_id="s2",
        message="Summarize statutory interest for late payment (general EU context).",
        attachments=[],
        tenant_id=tenant_id,
        workspace_id="ws2",
    )
    agent = LegalAgent(config=cfg)

    async def _decide_force_rag(**kwargs: object) -> object:
        plan = await decide_legal_tool_plan(**kwargs)  # type: ignore[misc]
        return plan.model_copy(
            update={
                "use_rag": True,
                "use_tools": False,
                "use_websearch": False,
                "intent": "rag",
                "reasoning_summary": "need retrieval",
            }
        )

    with patch(
        "legal_agent.pipeline.legal_execution_loop.decide_legal_tool_plan",
        side_effect=_decide_force_rag,
    ):
        result = await AgentEngine.run_agent(agent, request)

    assert result.answer and result.answer.strip()
    gov = _governance_events(result)
    assert len(gov) == 1
    assert gov[0].level == TraceLevel.WARNING
    assert "rag" in gov[0].message.lower()
    assert isinstance(gov[0].payload, LegalToolPlanGovernanceClampDiagV1)
    assert gov[0].payload.to_dict()["reason_code"] == "organization_disallows_nexus_rag"

    assert "rag" not in _trace_steps(result), "Nexus RagStep must not run when org disallows RAG."

    errors = [e for e in result.trace_events if e.level == TraceLevel.ERROR]
    assert not errors, f"unexpected ERROR: {[e.message for e in errors]}"


@pytest.mark.asyncio
async def test_legal_agent_organization_governance_clamps_all_layers_three_traces_e2e(
    tmp_path: Path,
) -> None:
    require_ollama_reachable()
    _ = tmp_path

    tenant_id = "legal-org-gov-clamp-all-e2e"
    cfg = _base_legal_cfg(
        tenant_id=tenant_id,
        organization_allow_rag=False,
        organization_allow_websearch=False,
        organization_allow_tools=False,
        with_tools=True,
    )
    request = RuntimeRequest(
        agent_id="legal-org-gov-clamp-all",
        user_id="u3",
        session_id="s3",
        message="Contract analysis: payment terms and liability (general guidance).",
        attachments=[],
        tenant_id=tenant_id,
        workspace_id="ws3",
    )
    agent = LegalAgent(config=cfg)

    async def _decide_force_combination(**kwargs: object) -> object:
        plan = await decide_legal_tool_plan(**kwargs)  # type: ignore[misc]
        return plan.model_copy(
            update={
                "use_rag": True,
                "use_tools": True,
                "use_websearch": True,
                "intent": "combination",
            }
        )

    with patch(
        "legal_agent.pipeline.legal_execution_loop.decide_legal_tool_plan",
        side_effect=_decide_force_combination,
    ):
        result = await AgentEngine.run_agent(agent, request)

    assert result.answer and result.answer.strip()
    gov = _governance_events(result)
    assert len(gov) == 3
    reasons = {e.payload.to_dict()["reason_code"] for e in gov if e.payload}
    assert reasons == {
        "organization_disallows_nexus_rag",
        "organization_disallows_nexus_websearch",
        "organization_disallows_nexus_tools",
    }

    steps = _trace_steps(result)
    assert "rag" not in steps
    assert "tools" not in steps
    assert "websearch" not in steps

    errors = [e for e in result.trace_events if e.level == TraceLevel.ERROR]
    assert not errors, f"unexpected ERROR: {[e.message for e in errors]}"
