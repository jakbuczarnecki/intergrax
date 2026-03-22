# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Integration test: full ingestion + retrieval + structured LLM extraction.

Requires a local Ollama instance (default http://127.0.0.1:11434) with:
  - a chat model (override via INTERGRAX_DEFAULT_OLLAMA_MODEL or LangChain ChatOllama defaults)
  - an embedding model (see OllamaEmbeddingProvider / INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL)

If Ollama is unreachable, the test is skipped.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.steps.legal_extract_clauses_step import LegalExtractClausesStep
from intergrax.llm.messages import AttachmentRef
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_pipeline
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import create_default_vectorstore_manager
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.ingestion.attachments import FileSystemAttachmentResolver
from intergrax.runtime.nexus.ingestion.ingestion_service import AttachmentIngestionService
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager

pytestmark = pytest.mark.integration


def _require_ollama_reachable() -> None:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3.0)
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"Ollama not reachable at 127.0.0.1:11434: {e}")


@pytest.mark.asyncio
async def test_extract_clauses_uses_rag_bootstraps_and_ollama(tmp_path: Path) -> None:
    _require_ollama_reachable()

    tenant_id = "legal-agent-extract-test"
    workspace_id = "ws-legal-1"
    session_id = "session-legal-1"
    user_id = "user-legal-1"

    contract_path = tmp_path / "contract.txt"
    contract_path.write_text(
        """
        The supplier shall not be liable for indirect damages.
        The client agrees to pay within 14 days.
        """,
        encoding="utf-8",
    )

    attachment = AttachmentRef(
        id="contract-1",
        type="txt",
        uri=contract_path.resolve().as_uri(),
    )

    embedding_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(provider_id="ollama"),
    )
    vectorstore_manager = create_default_vectorstore_manager(tenant_id=tenant_id)

    ingestion_service = AttachmentIngestionService(
        resolver=FileSystemAttachmentResolver(),
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
    )

    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)

    session_manager = SessionManager(storage=InMemorySessionStorage())

    runtime_config = RuntimeConfig(
        llm_adapter=llm_adapter,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        enable_rag=True,
        enable_websearch=False,
        production_mode=False,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        tools_mode="off",
    )

    context = RuntimeContext.build(
        config=runtime_config,
        session_manager=session_manager,
        ingestion_service=ingestion_service,
    )

    request = RuntimeRequest(
        agent_id="legal-agent-test",
        user_id=user_id,
        session_id=session_id,
        message="Analyze contract for risks",
        attachments=[attachment],
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    )

    state = RuntimeState(
        context=context,
        request=request,
        run_id="run-legal-extract-1",
    )

    legal_config = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        enable_rag=True,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        production_mode=False,
    )
    agent_state = LegalAgentState(config=legal_config)

    await LegalExtractClausesStep().run_step(state=state, agent_state=agent_state)

    assert len(agent_state.clauses) > 0, "expected at least one clause from Ollama structured output"
    texts = [c.text for c in agent_state.clauses]
    assert any("liable" in t.lower() for t in texts)
