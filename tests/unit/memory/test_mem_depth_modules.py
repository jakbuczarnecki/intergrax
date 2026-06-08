# © Artur Czarnecki. All rights reserved.

"""MEM-DEPTH memory module unit tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.delegation import DelegationSpec, ExploreDelegationProfile
from intergrax.llm.messages import ChatMessage
from intergrax.memory.entity_graph_memory import EntityGraphMemoryStore, EntityEdge, EntityNode
from intergrax.memory.session_summary_schema import SessionSummarySchema
from intergrax.memory.stores.postgres_memory_backend_rfc import evaluate_postgres_memory_backend_spike
from intergrax.memory.user_profile_dedup import deduplicate_memory_entries
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.memory.workspace_index_spike import build_workspace_index_spike
from intergrax.rag.retrieval.hybrid_retrieval_orchestrator import orchestrate_hybrid_retrieval
from intergrax.runtime.architecture.hybrid_retrieval import ChannelRetrievalHit, RetrievalChannel
from intergrax.runtime.nexus.delegation.explore_runner import ExploreDelegationRunner
from intergrax.runtime.nexus.session.document_store_session_storage import DocumentStoreSessionStorage
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore

pytestmark = pytest.mark.gate


def test_dedup_drops_near_duplicate_facts() -> None:
    existing = [
        UserProfileMemoryEntry(content="Senior Python engineer", kind=MemoryKind.USER_FACT),
    ]
    incoming = [
        UserProfileMemoryEntry(content="Senior python engineer.", kind=MemoryKind.USER_FACT),
    ]
    result = deduplicate_memory_entries(existing, incoming)
    assert result == []


def test_structured_session_summary_roundtrip() -> None:
    summary = SessionSummarySchema(
        title="Sprint",
        narrative="Planned memory depth",
        facts=["MEM-DEPTH done"],
        open_tasks=["ship"],
        decisions=["use compiler"],
    )
    text = summary.to_storage_text()
    assert "Planned memory depth" in text
    assert "ship" in text


def test_entity_graph_neighbors() -> None:
    store = EntityGraphMemoryStore()
    store.upsert_node(EntityNode(entity_id="u1", label="Artur"))
    store.upsert_node(EntityNode(entity_id="proj1", label="Intergrax"))
    store.add_edge(EntityEdge(source_id="u1", target_id="proj1", relation="works_on"))
    assert len(store.neighbors("u1")) == 1


@pytest.mark.asyncio
async def test_document_store_session_roundtrip() -> None:
    store = InMemoryDocumentStore()
    storage = DocumentStoreSessionStorage(store)
    session = await storage.create_session(tenant_id="t1", user_id="u1")
    await storage.append_message(
        tenant_id="t1",
        session_id=session.id,
        message=ChatMessage(role="user", content="hello"),
    )
    loaded = await storage.get_history(tenant_id="t1", session_id=session.id)
    assert len(loaded) == 1


def test_explore_delegation_runner_synthesis_only() -> None:
    spec = DelegationSpec(
        child_agent_id="explore",
        objective="find API usage",
        explore=ExploreDelegationProfile(parallel_search_budget=2),
    )
    runner = ExploreDelegationRunner()
    result = runner.run(
        spec,
        task_id="task-1",
        node_id="node-1",
        vector_hits=[
            ChannelRetrievalHit(
                channel=RetrievalChannel.VECTOR,
                document_id="doc-a",
                score=0.9,
            )
        ],
    )
    assert result.synthesis_text
    assert "task-1/delegation/node-1" in result.memory_namespace


def test_hybrid_retrieval_orchestrator_merges_channels() -> None:
    result = orchestrate_hybrid_retrieval(
        query_id="q1",
        vector_document_ids=[("a", 0.8)],
        keyword_document_ids=[("b", 0.7)],
        top_k=2,
    )
    assert len(result.merged_document_ids) <= 2


def test_workspace_index_spike_merkle() -> None:
    report = build_workspace_index_spike({"a.py": "print(1)\n", "b.py": "pass\n"})
    assert report.root_merkle
    assert len(report.chunks) == 2


def test_postgres_memory_spike_without_dsn() -> None:
    result = evaluate_postgres_memory_backend_spike(dsn_present=False)
    assert not result.configured
