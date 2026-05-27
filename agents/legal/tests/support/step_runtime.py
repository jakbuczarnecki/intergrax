# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Shared runtime wiring for Legal Agent per-step integration tests (Ollama).

Mirrors the non-RAG parts of ``test_legal_agent_extract_clauses_step`` so each step
test file stays focused on seed data and assertions.
"""

from __future__ import annotations

from legal.config.legal_agent_config import LegalAgentConfig
from legal.domain.legal_agent_state import LegalAgentState
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


def build_legal_ollama_runtime_state(
    *,
    run_id: str,
    tenant_id: str = "legal-agent-step-test",
    workspace_id: str = "ws-legal-step",
    session_id: str = "session-legal-step",
    user_id: str = "user-legal-step",
    message: str = "Legal agent single-step integration test",
) -> tuple[RuntimeState, LegalAgentState]:
    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)
    session_manager = SessionManager(storage=InMemorySessionStorage())

    runtime_config = RuntimeConfig(
        llm_adapter=llm_adapter,
        enable_rag=False,
        enable_websearch=False,
        production_mode=False,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        tools_mode="off",
    )

    context = RuntimeContext.build(
        config=runtime_config,
        session_manager=session_manager,
        ingestion_service=None,
    )

    request = RuntimeRequest(
        agent_id="legal-agent-test",
        user_id=user_id,
        session_id=session_id,
        message=message,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    )

    state = RuntimeState(
        context=context,
        request=request,
        run_id=run_id,
    )

    legal_config = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        production_mode=False,
        enable_rag=False,
    )
    agent_state = LegalAgentState(config=legal_config)
    return state, agent_state
