# © Artur Czarnecki. All rights reserved.

"""XD-02 (Agent <-> Context): a tenant A agent cannot read tenant B memory or RAG; revoked agents lose access.

PDC-D6-R019 (cross-tenant negatives). Every store here is a real one: the memory control plane over
the user-profile store with the LTM vector projection, the tenant-bound vector store manager (the RAG
boundary), the real context engine and the governed context-artifact port. Fakes: the embedding model
(external provider) only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from intergrax.context.contracts import (
    ContextFragmentScopeRef,
    ContextFragmentSource,
)
from intergrax.context.registry import ContextPluginRegistry
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.contracts.artifact_ref import ArtifactClass
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessDenied,
    ArtifactReadRequest,
)
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from tests.qualification.pdc_x.d2._support import (
    FakeAdapter,
    FragmentProvider,
    assembly_request,
    make_fragment,
    message_text,
    provider_context,
)
from tests.qualification.pdc_x.d8._d8_support import (
    TENANT_A,
    TENANT_B,
    make_artifact,
    make_context,
    make_governance,
    make_port,
    reader_grants,
    sqlite_store,
)

pytestmark = [pytest.mark.integration]

_NOW = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)


class _FixedEmbedding:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _identity(tenant_id: str, user_id: str = "user-1") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


@dataclass
class _TenantMemory:
    tenant_id: str
    plane: DefaultMemoryControlPlane
    manager: VectorstoreManager

    async def remember(self, text: str, *, user_id: str = "user-1") -> str:
        identity = _identity(self.tenant_id, user_id)
        result = await self.plane.remember(
            identity,
            user_memory_scope(identity),
            MemoryControlRememberRequest(
                entry=UserProfileMemoryEntry(
                    content=text,
                    valid_from=(_NOW - timedelta(days=1)).isoformat(),
                )
            ),
        )
        assert result.entry_id
        return result.entry_id


def _memory_for(tenant_id: str) -> _TenantMemory:
    backend = InMemoryVectorStore(tenant_id)
    manager = VectorstoreManager(backend, scope=VectorStoreScope(tenant_id=tenant_id))
    projection = UserProfileLtmVectorProjection(
        embedding_manager=_FixedEmbedding(),  # type: ignore[arg-type]
        vectorstore_manager=manager,
        tenant_id=tenant_id,
        vector_index_namespace=None,
        workspace_id=None,
    )
    profile_manager = UserProfileManager(
        InMemoryUserProfileStore(), tenant_id=tenant_id, memory_projections=(projection,)
    )
    capability = UserProfileManagerMemoryCapability(_manager=profile_manager)
    return _TenantMemory(tenant_id, DefaultMemoryControlPlane(user_profile=capability), manager)


def _recalled_text(result) -> str:  # type: ignore[no-untyped-def]
    return repr(result)


# ----------------------------------------------------------------------- memory


async def test_xd02_tenant_a_identity_cannot_touch_tenant_b_memory_scope() -> None:
    memory_a, memory_b = _memory_for(TENANT_A), _memory_for(TENANT_B)
    await memory_a.remember("tenant-a private preference ALPHA-SECRET")
    await memory_b.remember("tenant-b private preference BRAVO-SECRET")

    identity_a = _identity(TENANT_A)
    foreign_scope = user_memory_scope(_identity(TENANT_B))  # a scope naming tenant B
    for call in (
        memory_a.plane.recall(identity_a, foreign_scope, MemoryControlRecallRequest(query="preference", top_k=5)),
        memory_b.plane.recall(identity_a, foreign_scope, MemoryControlRecallRequest(query="preference", top_k=5)),
        memory_b.plane.remember(
            identity_a, foreign_scope, MemoryControlRememberRequest(entry=UserProfileMemoryEntry(content="x"))
        ),
    ):
        with pytest.raises(MemoryControlAccessDenied):
            await call


async def test_xd02_memory_recall_never_returns_another_tenants_entries() -> None:
    memory_a, memory_b = _memory_for(TENANT_A), _memory_for(TENANT_B)
    await memory_a.remember("tenant-a private preference ALPHA-SECRET")
    await memory_b.remember("tenant-b private preference BRAVO-SECRET")

    request = MemoryControlRecallRequest(query="private preference", top_k=10)
    recalled_a = _recalled_text(
        await memory_a.plane.recall(_identity(TENANT_A), user_memory_scope(_identity(TENANT_A)), request)
    )
    recalled_b = _recalled_text(
        await memory_b.plane.recall(_identity(TENANT_B), user_memory_scope(_identity(TENANT_B)), request)
    )
    assert "ALPHA-SECRET" in recalled_a and "BRAVO-SECRET" not in recalled_a
    assert "BRAVO-SECRET" in recalled_b and "ALPHA-SECRET" not in recalled_b


async def test_xd02_a_foreign_tenant_identity_on_a_tenant_bound_plane_recalls_nothing() -> None:
    memory_a = _memory_for(TENANT_A)
    await memory_a.remember("tenant-a private preference ALPHA-SECRET")
    cross = memory_a.plane.recall(
        _identity(TENANT_B),
        user_memory_scope(_identity(TENANT_B)),
        MemoryControlRecallRequest(query="preference", top_k=10),
    )
    try:
        leaked = _recalled_text(await cross)
    except MemoryControlAccessDenied:
        return  # fail-closed is the expected behavior
    assert "ALPHA-SECRET" not in leaked


# ----------------------------------------------------------------------- RAG vector boundary


_NS = "ns"


def _record(document_id: str, tenant_id: str, text: str) -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": document_id, "root_document_id": document_id},
            "scope": {"tenant_id": tenant_id, "namespace": _NS},
            "content": text,
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": document_id},
        }
    )
    return VectorStoreRecord(
        document=document, embedding=np.asarray([0.1, 0.2, 0.3], dtype=np.float32), vector_id=f"vec-{document_id}"
    )


def test_xd02_rag_manager_of_one_tenant_refuses_foreign_scope_and_foreign_documents() -> None:
    scope_a = VectorStoreScope(tenant_id=TENANT_A, namespace=_NS)
    scope_b = VectorStoreScope(tenant_id=TENANT_B, namespace=_NS)
    manager_a = VectorstoreManager(InMemoryVectorStore(TENANT_A), scope=scope_a)
    manager_b = VectorstoreManager(InMemoryVectorStore(TENANT_B), scope=scope_b)
    manager_a.add_records([_record("doc-a", TENANT_A, "alpha handbook")], scope=scope_a)
    manager_b.add_records([_record("doc-b", TENANT_B, "bravo handbook")], scope=scope_b)

    # Query with the agent's own scope returns only its own tenant's documents.
    hits_a = manager_a.query([0.1, 0.2, 0.3], scope=scope_a, top_k=10)
    assert [hit.document.identity.document_id for hit in hits_a] == ["doc-a"]
    # A tenant A agent addressing tenant B's scope through tenant A's manager is refused ...
    with pytest.raises(VectorStoreContractError):
        manager_a.query([0.1, 0.2, 0.3], scope=scope_b, top_k=10)
    # ... and cannot plant a tenant B record into tenant A's index.
    with pytest.raises(VectorStoreContractError):
        manager_a.add_records([_record("doc-b2", TENANT_B, "planted")])
    # Tenant B's index is untouched by any of it.
    hits_b = manager_b.query([0.1, 0.2, 0.3], scope=scope_b, top_k=10)
    assert [hit.document.identity.document_id for hit in hits_b] == ["doc-b"]


# ----------------------------------------------------------------------- context assembly


async def test_xd02_context_engine_does_not_admit_a_foreign_tenant_fragment() -> None:
    """A provider handing tenant B scoped data to a tenant A assembly request must not reach the model."""
    foreign = make_fragment(
        "foreign-rag",
        ContextFragmentSource.RAG,
        "BRAVO-SECRET tenant b handbook",
        relevance=0.99,
        scope_ref=ContextFragmentScopeRef(tenant_id=TENANT_B),
    )
    own = make_fragment(
        "own-rag",
        ContextFragmentSource.RAG,
        "ALPHA handbook of tenant a",
        relevance=0.9,
        scope_ref=ContextFragmentScopeRef(tenant_id=TENANT_A),
    )
    registry = ContextPluginRegistry()
    registry.add_provider(FragmentProvider("xd02.rag", {ContextFragmentSource.RAG}, lambda: [foreign, own]))
    assembled = await DefaultNexusContextEngine(registry=registry).assemble(
        assembly_request(budget_tokens=4000),  # request tenant is tenant-a
        provider_ctx=provider_context(FakeAdapter(), "what does the handbook say?"),
    )
    text = message_text(assembled.messages)
    assert "ALPHA handbook" in text
    assert "BRAVO-SECRET" not in text, "tenant B scoped content reached a tenant A model call"


# ----------------------------------------------------------------------- revoked agent


def test_xd02_revoked_agent_loses_context_access_and_the_denial_is_audited(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    store.put(make_artifact(tenant_id=TENANT_A, data=b"tenant-a context record"))
    request = ArtifactReadRequest(
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="art-1"
    )
    agent = make_context(tenant_id=TENANT_A, agent_id="agent-reader")

    granted = make_governance(granted=reader_grants(tenant_id=TENANT_A, agent_id="agent-reader"))
    port, _ = make_port(store, harness=granted)
    assert port.read(agent, request).content == b"tenant-a context record"

    # The agent is revoked: governance is composed without its grant. Access ends at once.
    revoked = make_governance(granted={})
    revoked_port, _ = make_port(store, harness=revoked)
    with pytest.raises(ArtifactAccessDenied):
        revoked_port.read(agent, request)
    assert revoked.sink.events, "the denial is recorded in governance evidence"
    # A different, still-granted agent of the same tenant is not affected by the revocation.
    other = make_governance(granted=reader_grants(tenant_id=TENANT_A, agent_id="agent-other"))
    other_port, _ = make_port(store, harness=other)
    assert other_port.read(make_context(tenant_id=TENANT_A, agent_id="agent-other"), request).content == (
        b"tenant-a context record"
    )
