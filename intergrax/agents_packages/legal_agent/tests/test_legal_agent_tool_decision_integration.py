# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Integration / e2e checks for LegalToolDecision + Nexus bridge (Rag/Websearch/Tools steps).

Uses real Ollama LLM, embedding pipeline, and in-memory vectorstore — no mocked adapters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents_packages.legal_agent.legal_agent import LegalAgent
from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.legal_pipeline_routing import (
    legal_workspace_metrics_json,
)
from intergrax.agents_packages.legal_agent.legal_tool_runtime_bridge import (
    run_legal_tool_runtime_bridge,
    sync_legal_tool_runtime_feedback,
)
from intergrax.agents_packages.legal_agent.tool_decision_component import (
    decide_legal_tool_plan,
)
from intergrax.llm.messages import AttachmentRef
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
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e


def _trace_steps(answer: RuntimeAnswer) -> list[str]:
    return [e.step for e in (answer.trace_events or [])]


def _build_rag_legal_config(
    *,
    tmp_path: Path,
    tenant_id: str,
    use_legal_tool_decision: bool,
) -> tuple[LegalAgentConfig, RuntimeRequest, AttachmentRef]:
    contract_path = tmp_path / "contract_tool_decision.txt"
    contract_path.write_text(
        "The supplier limits liability to direct damages only. "
        "Payment is due within 30 calendar days.\n",
        encoding="utf-8",
    )
    attachment = AttachmentRef(
        id="contract-tool-decision",
        type="txt",
        uri=contract_path.resolve().as_uri(),
    )

    embedding_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(provider_id="ollama"),
    )
    vectorstore_manager = create_default_vectorstore_manager(tenant_id=tenant_id)
    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)
    session_manager = SessionManager(storage=InMemorySessionStorage())

    agent_config = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        enable_rag=True,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        production_mode=False,
        use_legal_tool_decision=use_legal_tool_decision,
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
    )

    request = RuntimeRequest(
        agent_id="legal-agent-tool-decision",
        user_id="user-td",
        session_id="session-td",
        message="Summarize payment terms and liability from the attached contract.",
        attachments=[attachment],
        tenant_id=tenant_id,
        workspace_id="ws-td",
    )
    return agent_config, request, attachment


@pytest.mark.asyncio
async def test_legal_tool_decision_emits_trace_and_rag_when_enabled(tmp_path: Path) -> None:
    require_ollama_reachable()
    tenant_id = "legal-tool-decision-rag"

    cfg, request, _ = _build_rag_legal_config(
        tmp_path=tmp_path,
        tenant_id=tenant_id,
        use_legal_tool_decision=True,
    )
    agent = LegalAgent(config=cfg)

    result = await AgentEngine.run_agent(agent, request)

    assert result.answer and len(result.answer.strip()) > 0
    steps = _trace_steps(result)
    assert "LegalToolDecision" in steps, f"expected LegalToolDecision in trace, got {steps!r}"

    td_events = [e for e in result.trace_events if e.step == "LegalToolDecision"]
    assert td_events
    td_msg = td_events[0].message.lower()
    assert "rag=true" in td_msg, (
        f"expected tool decision to enable RAG for attachment + RAG-capable runtime; got {td_events[0].message!r}"
    )

    rag_steps = [e for e in result.trace_events if e.step == "rag"]
    assert rag_steps, "expected at least one Nexus rag trace event"

    assert result.route.used_rag is True


@pytest.mark.asyncio
async def test_legal_tool_decision_disabled_no_decision_trace(tmp_path: Path) -> None:
    require_ollama_reachable()
    tenant_id = "legal-tool-decision-off"

    cfg, request, _ = _build_rag_legal_config(
        tmp_path=tmp_path,
        tenant_id=tenant_id,
        use_legal_tool_decision=False,
    )
    agent = LegalAgent(config=cfg)

    result = await AgentEngine.run_agent(agent, request)

    assert result.answer and len(result.answer.strip()) > 0
    steps = _trace_steps(result)
    assert "LegalToolDecision" not in steps, f"did not expect LegalToolDecision, got {steps!r}"


@pytest.mark.asyncio
async def test_legal_tool_bridge_and_metrics_sync_with_real_runtime(tmp_path: Path) -> None:
    """
    After SETUP_STEPS + tool decision + bridge, workspace metrics include tool plan + runtime flags.
    """
    require_ollama_reachable()
    tenant_id = "legal-tool-bridge-metrics"

    cfg, request, _ = _build_rag_legal_config(
        tmp_path=tmp_path,
        tenant_id=tenant_id,
        use_legal_tool_decision=True,
    )
    agent = LegalAgent(config=cfg)
    context = agent.build_context(request)

    state = RuntimeState(
        context=context,
        request=request,
        run_id="run-tool-bridge-metrics",
    )
    agent_state = LegalAgentState(config=cfg)
    state.agent_state = agent_state

    await RuntimeStepRunner.execute_pipeline(SETUP_STEPS, state)

    plan = await decide_legal_tool_plan(state=state, legal_config=cfg)
    agent_state.last_legal_tool_plan = plan
    await run_legal_tool_runtime_bridge(state=state, plan=plan)
    sync_legal_tool_runtime_feedback(agent_state, state)

    metrics_raw = legal_workspace_metrics_json(agent_state, runtime_state=state)
    metrics = json.loads(metrics_raw)

    assert metrics.get("legal_tool_intent") == plan.intent
    assert metrics.get("legal_tool_confidence") == plan.confidence
    assert "legal_tool_runtime_feedback" in metrics
    fb = metrics["legal_tool_runtime_feedback"]
    assert isinstance(fb, dict)
    assert "used_rag" in fb
    assert "runtime_used_rag" in metrics
