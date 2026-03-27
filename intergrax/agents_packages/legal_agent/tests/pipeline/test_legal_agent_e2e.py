# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agents.agent_engine import AgentEngine
from testing_support.builder import require_ollama_reachable

from intergrax.agents_packages.legal_agent.legal_agent import LegalAgent
from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig

from intergrax.llm.messages import AttachmentRef
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_pipeline
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import create_default_vectorstore_manager

from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_legal_agent_e2e(tmp_path: Path) -> None:
    require_ollama_reachable()

    tenant_id = "legal-agent-e2e"
    workspace_id = "ws-legal-e2e"
    session_id = "session-legal-e2e"
    user_id = "user-legal-e2e"

    # --- TEST DOCUMENT ---
    contract_path = tmp_path / "contract.txt"
    contract_path.write_text(
        """
        The supplier shall not be liable for indirect damages.
        The client agrees to pay within 14 days.
        """,
        encoding="utf-8",
    )

    attachment = AttachmentRef(
        id="contract-e2e",
        type="txt",
        uri=contract_path.resolve().as_uri(),
    )

    # --- RAG SETUP ---
    embedding_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(provider_id="ollama"),
    )

    vectorstore_manager = create_default_vectorstore_manager(tenant_id=tenant_id)

    # --- LLM ---
    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)

    # --- SESSION ---
    session_manager = SessionManager(storage=InMemorySessionStorage())

    # --- AGENT CONFIG ---
    agent_config = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        enable_rag=True,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        production_mode=False,
    )

    agent = LegalAgent(config=agent_config)

    # --- REQUEST ---
    request = RuntimeRequest(
        agent_id="legal-agent",
        user_id=user_id,
        session_id=session_id,
        message="Analyze contract and highlight risks",
        attachments=[attachment],
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    )

    # --- RUN ---    
    result = await AgentEngine.run_agent(agent, request)

    # --- ASSERTIONS ---

    # 1. RuntimeAnswer exists
    assert result is not None

    # 2. Answer content
    assert result.answer is not None
    assert len(result.answer.strip()) > 0

    # 3. Should reference legal content
    assert any(word in result.answer.lower() for word in ["risk", "liable", "clause"])

    # 4. Routing
    assert result.route is not None
    assert result.route.strategy == "legal_agent"

    # 5. RAG usage
    assert result.route.used_rag is True