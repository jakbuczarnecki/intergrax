# © Artur Czarnecki. All rights reserved.

"""MEM-VEC-1.3: LTM vector wiring integration gate."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.memory_wiring import (
    MemoryPlatformWiring,
    build_session_manager_from_environment,
)
from intergrax.applications._shared.memory_vector_wiring import resolve_rag_stack_for_memory_wiring
from intergrax.applications.contracts.environment_profile import MemoryProfile
from langchain_core.documents import Document
from intergrax.llm.messages import ChatMessage
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack, create_default_rag_stack
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from lab_application.host.settings import LabApplicationSettings

pytestmark = [pytest.mark.integration, pytest.mark.gate]


class _DeterministicEmbeddingManager(BaseEmbeddingManager):
    """Hash-stable embeddings for gate tests — identical text => identical vector."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed_texts([text])[0].tolist()

    def embed_documents(self, documents: Sequence[Document]) -> EmbeddingResult:
        texts = [doc.page_content for doc in documents]
        matrix = self.embed_texts(texts)
        return EmbeddingResult(embeddings=matrix, texts=list(texts))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 997
            rows.append([float((seed + i) % 17) / 17.0 for i in range(16)])
        return np.array(rows, dtype=np.float32)


def _memory_rag_stack() -> RagStack:
    store = InMemoryVectorStore(tenant_id="lab")
    embedding = _DeterministicEmbeddingManager()
    vectorstore = VectorstoreManager(store=store)
    return create_default_rag_stack(
        vectorstore_manager=vectorstore,
        embedding_manager=embedding,
    )


@pytest.mark.asyncio
async def test_ltm_vector_wiring_search_returns_hits_after_write() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.memory_profile = MemoryProfile(
        enable_user_memory=True,
        enable_long_term_memory=True,
        enable_task_memory=False,
    )

    rag_stack = _memory_rag_stack()
    wiring = MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        organization_profile_store=None,
    )
    session_manager = build_session_manager_from_environment(
        env,
        memory_wiring=wiring,
        rag_stack=rag_stack,
    )
    assert session_manager.user_profile_manager is not None
    assert session_manager.user_profile_manager.is_longterm_rag_enabled()

    fact = "User prefers concise technical answers in Polish."
    await session_manager.user_profile_manager.add_memory_entry(
        "tester",
        UserProfileMemoryEntry(content=fact, kind="user_fact"),
    )

    result = await session_manager.search_user_longterm_memory(
        "tester",
        fact,
        top_k=4,
    )
    assert result is not None
    assert result["used_longterm"] is True
    assert len(result["hits"]) >= 1
    assert fact in (result["hits"][0].content or "")


@pytest.mark.asyncio
async def test_episodic_index_recall_after_append() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.memory_profile = MemoryProfile(
        enable_user_memory=True,
        enable_long_term_memory=False,
        enable_session_vector_index=True,
    )

    rag_stack = _memory_rag_stack()
    wiring = MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        organization_profile_store=None,
    )
    session_manager = build_session_manager_from_environment(
        env,
        memory_wiring=wiring,
        rag_stack=rag_stack,
    )

    await session_manager.create_session(
        tenant_id="lab",
        session_id="sess-episodic",
        user_id="tester",
        workspace_id="default",
    )
    turn_text = "We discussed vector memory wiring for episodic recall."
    await session_manager.append_message(
        tenant_id="lab",
        session_id="sess-episodic",
        message=ChatMessage(role="user", content=turn_text),
    )

    hits = await session_manager.search_session_semantic_recall(
        tenant_id="lab",
        session_id="sess-episodic",
        user_id="tester",
        query=turn_text,
    )
    assert hits
    assert turn_text in hits[0]["text"]


def test_resolve_rag_stack_for_memory_wiring_when_ltm_enabled_without_rag_context() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.memory_profile = MemoryProfile(enable_long_term_memory=True)
    env.context_profile = env.context_profile.model_copy(update={"enable_rag": False})

    stack = resolve_rag_stack_for_memory_wiring(env)
    assert stack is not None
    assert stack.vectorstore_manager is not None
    assert stack.embedding_manager is not None
