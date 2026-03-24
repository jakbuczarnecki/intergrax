# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Enterprise regression (TEST 5): Tier-2 tool decision across **two turns** with the same session.

Turn 1 ingests a contract (attachments + natural tool-decision LLM).
Turn 2 is a short follow-up without new attachments; the decision prompt includes
trimmed conversation history (see ``legal_tool_decision_user`` + ``_history_snippet``).

Expectations (behavioral, LLM-dependent within bounds):
  - Turn-2 plan stays in ``intent in {\"llm_only\", \"rag\"}`` (no ``websearch`` /
    ``tools`` / heavy ``combination`` in the happy path).
  - When the model picks ``llm_only`` for the follow-up, Nexus RAG should stay off
    (``use_rag`` false, no ``rag`` trace step). If it picks ``rag``, re-retrieval is allowed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import intergrax.agents_packages.legal_agent.legal_execution_loop as legal_execution_loop
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents_packages.legal_agent.legal_agent import LegalAgent
from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_tool_plan import LegalToolPlan
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
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e


def _trace_steps(answer: RuntimeAnswer) -> list[str]:
    return [e.step for e in (answer.trace_events or [])]


@pytest.mark.asyncio
async def test_legal_tool_decision_second_turn_follow_up_skips_nexus_rag_when_appropriate(
    tmp_path: Path,
) -> None:
    require_ollama_reachable()

    tenant_id = "legal-tool-decision-multiturn"
    contract_path = tmp_path / "contract_multiturn.txt"
    contract_path.write_text(
        "Supplier liability is limited to direct damages. "
        "Indirect or consequential damages are excluded. "
        "Payment is due within 30 days.\n",
        encoding="utf-8",
    )
    attachment = AttachmentRef(
        id="contract-multiturn",
        type="txt",
        uri=contract_path.resolve().as_uri(),
    )

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
    )

    agent = LegalAgent(config=cfg)
    agent_id = "legal-agent-tool-decision-mt"
    session_id = "session-mt"
    user_id = "user-mt"

    captured_plans: list[LegalToolPlan] = []
    _orig_decide = legal_execution_loop.decide_legal_tool_plan

    async def _capture_plan(**kwargs: object) -> LegalToolPlan:
        plan = await _orig_decide(**kwargs)  # type: ignore[misc]
        captured_plans.append(plan)
        return plan

    with patch.object(
        legal_execution_loop,
        "decide_legal_tool_plan",
        side_effect=_capture_plan,
    ):
        req1 = RuntimeRequest(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            message="Analyze the attached contract.",
            attachments=[attachment],
            tenant_id=tenant_id,
            workspace_id="ws-mt",
        )
        r1 = await AgentEngine.run_agent(agent, req1)

        req2 = RuntimeRequest(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            message="What about liability?",
            attachments=[],
            tenant_id=tenant_id,
            workspace_id="ws-mt",
        )
        r2 = await AgentEngine.run_agent(agent, req2)

    assert len(captured_plans) == 2
    plan1, plan2 = captured_plans

    assert r1.answer and r1.answer.strip()
    assert r2.answer and r2.answer.strip()

    assert plan1.use_rag is True, (
        "Turn 1 has a new attachment; tool decision should enable Nexus RAG per policy guidance."
    )
    assert "rag" in _trace_steps(r1)

    assert plan2.intent in ("llm_only", "rag")
    if plan2.intent == "llm_only":
        assert plan2.use_rag is False
        assert "rag" not in _trace_steps(r2)

    for label, ans in (("turn1", r1), ("turn2", r2)):
        errors = [e for e in ans.trace_events if e.level == TraceLevel.ERROR]
        assert not errors, f"{label}: unexpected ERROR traces: {[e.message for e in errors]}"
