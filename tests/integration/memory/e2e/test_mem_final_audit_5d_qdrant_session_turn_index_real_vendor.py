# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5D — Qdrant SessionTurnIndex real-vendor qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.memory_wiring import (
    MemoryPlatformWiring,
    build_session_manager_from_environment,
)
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications.contracts.environment_profile import MemoryProfile
from intergrax.integrations.providers.vector_store.qdrant.integration import (
    QDRANT_VECTOR_STORE_PROVIDER_ID,
)
from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidence,
    MemoryProviderDurabilityProofKind,
    MemoryProviderTrustedDurabilityStatus,
)
from intergrax.memory.contracts.provider_identity import (
    BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
    builtin_session_turn_index_store_identity,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderDescriptor,
    MemoryProviderQualificationRequest,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidence,
    qualification_evidence_from_result,
)
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    MemoryProviderInstanceFactory,
    MemoryProviderQualificationRunner,
)
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from lab_application.host.settings import LabApplicationSettings
from tests.integration.memory.e2e.qdrant_session_turn_index_real_vendor_support import (
    QdrantSessionTurnIndexQualificationEnv,
    assert_qdrant_backend_unavailable,
    build_qualification_env,
    build_qdrant_memory_rag_stack,
    build_vector_session_turn_index_store,
    close_qdrant_integration,
    drop_qualification_collection,
    ensure_qdrant_available,
    qdrant_reachable,
    qdrant_topology_metadata,
    session_turn_index_store_factory,
    unique_qualification_run_id,
)
from tests.unit.memory.test_mem_ent13_provider_qualification import _context

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.qualification,
    pytest.mark.no_ci,
]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MEMORY_ROOT = _REPO_ROOT / "intergrax" / "memory"
_EVIDENCE_SOURCE = "qdrant_session_turn_index_real_vendor_qualification"
_PROOF_KIND = MemoryProviderDurabilityProofKind.REAL_VENDOR_RECONNECT


def _require_qdrant() -> None:
    if not qdrant_reachable():
        pytest.skip("Qdrant not configured (set INTERGRAX_QDRANT_URL or host/port)")
    try:
        ensure_qdrant_available()
    except Exception as exc:
        pytest.skip(f"Qdrant backend unavailable: {exc}")


@pytest.fixture
def qdrant_sti_env() -> QdrantSessionTurnIndexQualificationEnv:
    _require_qdrant()
    run_id = unique_qualification_run_id()
    env = build_qualification_env(qualification_run_id=run_id)
    yield env
    try:
        drop_qualification_collection(env)
    except Exception as exc:
        pytest.fail(f"qualification collection cleanup failed: {exc}")


def test_memory_core_has_no_qdrant_sdk_imports() -> None:
    forbidden = ("qdrant_client", "qdrant")
    for path in _MEMORY_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod == "qdrant_client" or mod.startswith("qdrant."):
                        pytest.fail(f"forbidden import in memory core: {path}")
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if mod == "qdrant_client" or mod.startswith("qdrant."):
                    pytest.fail(f"forbidden import in memory core: {path}")


def test_memory_core_has_no_rag_bootstrap_imports() -> None:
    forbidden = "intergrax.rag.bootstrap"
    for path in _MEMORY_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(forbidden):
                pytest.fail(f"rag bootstrap import in memory core: {path}")


async def _run_qdrant_sti_qualification(
    env: QdrantSessionTurnIndexQualificationEnv,
    run_id: str,
) -> object:
    create, dispose = session_turn_index_store_factory(env)

    class _QdrantStiFactory(MemoryProviderInstanceFactory[SessionTurnIndexStore]):
        async def create(self) -> SessionTurnIndexStore:
            store = create()
            assert isinstance(store, VectorSessionTurnIndexStore)
            return store

        async def dispose(self, instance: SessionTurnIndexStore) -> None:
            await dispose(instance)

    runner = MemoryProviderQualificationRunner()
    return await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
            capabilities=(MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,),
            backing_provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
        ),
        context=_context(run_id),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(session_turn_index_store=_QdrantStiFactory()),
    )


@pytest.mark.asyncio
async def test_qdrant_topology_metadata(qdrant_sti_env: QdrantSessionTurnIndexQualificationEnv) -> None:
    _, integration = build_vector_session_turn_index_store(qdrant_sti_env)
    try:
        meta = qdrant_topology_metadata(integration)
        assert meta["backend_provider_id"] == QDRANT_VECTOR_STORE_PROVIDER_ID
        assert meta["transport"] in {"http", "https"}
        assert meta["topology"] == "single-node"
    finally:
        close_qdrant_integration(integration)


@pytest.mark.asyncio
async def test_qdrant_sti_behavioral_qualification_produces_trusted_evidence(
    qdrant_sti_env: QdrantSessionTurnIndexQualificationEnv,
) -> None:
    run_id = qdrant_sti_env.qualification_run_id
    result = await _run_qdrant_sti_qualification(qdrant_sti_env, run_id)
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    evidence = qualification_evidence_from_result(
        result,
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        evidence_source=_EVIDENCE_SOURCE,
    )
    assert evidence is not None
    assert evidence.provider_id == BUILTIN_VECTOR_SESSION_TURN_INDEX_ID
    assert evidence.backing_provider_id == QDRANT_VECTOR_STORE_PROVIDER_ID
    assert evidence.qualification_run_id == run_id


@pytest.mark.asyncio
async def test_qdrant_sti_turns_survive_client_reconnect(
    qdrant_sti_env: QdrantSessionTurnIndexQualificationEnv,
) -> None:
    tenant_id = qdrant_sti_env.tenant_id
    session_id = f"sess-reconnect-{qdrant_sti_env.qualification_run_id}"
    marker = f"reconnect-marker-{qdrant_sti_env.qualification_run_id}"
    entry_id = f"entry-reconnect-{qdrant_sti_env.qualification_run_id}"

    store_a, integration_a = build_vector_session_turn_index_store(qdrant_sti_env, tenant_id=tenant_id)
    try:
        await store_a.upsert_turn(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
            message=ChatMessage(role="user", content=marker, entry_id=entry_id),
        )
    finally:
        close_qdrant_integration(integration_a)

    store_b, integration_b = build_vector_session_turn_index_store(qdrant_sti_env, tenant_id=tenant_id)
    try:
        hits = await store_b.search_turns(
            query=marker,
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
        )
        assert any(hit.entry_id == entry_id for hit in hits)
    finally:
        close_qdrant_integration(integration_b)


@pytest.mark.asyncio
async def test_qdrant_sti_durability_evidence_bundle(
    qdrant_sti_env: QdrantSessionTurnIndexQualificationEnv,
) -> None:
    run_id = qdrant_sti_env.qualification_run_id
    result = await _run_qdrant_sti_qualification(qdrant_sti_env, run_id)
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    behavioral = qualification_evidence_from_result(
        result,
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        evidence_source=_EVIDENCE_SOURCE,
    )
    assert behavioral is not None
    dur = MemoryProviderDurabilityEvidence(
        provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        durability_status=MemoryProviderTrustedDurabilityStatus.DURABLE,
        qualification_run_id=run_id,
        reference_time_iso=behavioral.reference_time_iso,
        evidence_source=_EVIDENCE_SOURCE,
        proof_kind=_PROOF_KIND,
        backing_provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
    )
    assert dur.backing_provider_id == QDRANT_VECTOR_STORE_PROVIDER_ID
    assert dur.proof_kind is _PROOF_KIND


@pytest.mark.asyncio
async def test_qdrant_cross_tenant_isolation_two_stores(
    qdrant_sti_env: QdrantSessionTurnIndexQualificationEnv,
) -> None:
    marker = f"tenant-secret-{qdrant_sti_env.qualification_run_id}"
    session_id = f"sess-x-{qdrant_sti_env.qualification_run_id}"

    store_a, int_a = build_vector_session_turn_index_store(
        qdrant_sti_env,
        tenant_id="tenant-5d-a",
    )
    try:
        await store_a.upsert_turn(
            tenant_id="tenant-5d-a",
            session_id=session_id,
            user_id="user-x",
            message=ChatMessage(
                role="user",
                content=marker,
                entry_id=f"entry-a-{qdrant_sti_env.qualification_run_id}",
            ),
        )
    finally:
        close_qdrant_integration(int_a)

    store_b, int_b = build_vector_session_turn_index_store(
        qdrant_sti_env,
        tenant_id="tenant-5d-b",
    )
    try:
        await store_b.upsert_turn(
            tenant_id="tenant-5d-b",
            session_id=session_id,
            user_id="user-x",
            message=ChatMessage(
                role="user",
                content="decoy-tenant-b",
                entry_id=f"entry-b-{qdrant_sti_env.qualification_run_id}",
            ),
        )
        hits = await store_b.search_turns(
            query=marker,
            tenant_id="tenant-5d-b",
            session_id=session_id,
            user_id="user-x",
        )
        leaked = [hit for hit in hits if marker in (hit.message.content or "")]
        assert not leaked
    finally:
        close_qdrant_integration(int_b)


@pytest.mark.asyncio
async def test_qdrant_duplicate_turn_id_is_deterministic(
    qdrant_sti_env: QdrantSessionTurnIndexQualificationEnv,
) -> None:
    tenant_id = qdrant_sti_env.tenant_id
    session_id = f"sess-dup-{qdrant_sti_env.qualification_run_id}"
    entry_id = f"entry-dup-{qdrant_sti_env.qualification_run_id}"
    message = ChatMessage(role="user", content="dup-content", entry_id=entry_id)
    store, integration = build_vector_session_turn_index_store(qdrant_sti_env, tenant_id=tenant_id)
    try:
        await store.upsert_turn(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
            message=message,
        )
        await store.upsert_turn(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
            message=message,
        )
        hits = await store.search_turns(
            query="dup-content",
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
            top_k=8,
        )
        matching = [hit for hit in hits if hit.entry_id == entry_id]
        assert len(matching) == 1
    finally:
        close_qdrant_integration(integration)


@pytest.mark.asyncio
async def test_qdrant_backend_unavailable_fails_explicitly() -> None:
    assert_qdrant_backend_unavailable("http://127.0.0.1:1")


@pytest.mark.asyncio
async def test_application_episodic_recall_after_qdrant_reconnect(
    qdrant_sti_env: QdrantSessionTurnIndexQualificationEnv,
) -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.memory_profile = MemoryProfile(
        enable_user_memory=False,
        enable_long_term_memory=False,
        enable_session_vector_index=True,
    )
    tenant_id = qdrant_sti_env.tenant_id
    turn_text = f"episodic-e2e-{qdrant_sti_env.qualification_run_id}"

    stack_a, integration_a = build_qdrant_memory_rag_stack(qdrant_sti_env, tenant_id=tenant_id)
    wiring = MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        organization_profile_store=None,
    )
    session_a = build_session_manager_from_environment(
        env,
        tenant_id=tenant_id,
        memory_wiring=wiring,
        rag_stack=stack_a,
    )
    assert isinstance(session_a._session_turn_index_store, VectorSessionTurnIndexStore)
    await session_a.create_session(
        tenant_id=tenant_id,
        session_id="sess-e2e",
        user_id="qual-user",
        workspace_id="default",
    )
    await session_a.append_message(
        tenant_id=tenant_id,
        session_id="sess-e2e",
        message=ChatMessage(role="user", content=turn_text),
    )
    close_qdrant_integration(integration_a)

    stack_b, integration_b = build_qdrant_memory_rag_stack(qdrant_sti_env, tenant_id=tenant_id)
    session_b = build_session_manager_from_environment(
        env,
        tenant_id=tenant_id,
        memory_wiring=wiring,
        rag_stack=stack_b,
    )
    try:
        hits = await session_b.search_session_semantic_recall(
            tenant_id=tenant_id,
            session_id="sess-e2e",
            user_id="qual-user",
            query=turn_text,
        )
        assert hits
        assert turn_text in hits[0]["text"]
    finally:
        close_qdrant_integration(integration_b)


def test_composite_identity_descriptor() -> None:
    identity = builtin_session_turn_index_store_identity(
        BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        backing_provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
    )
    assert identity.provider_id == BUILTIN_VECTOR_SESSION_TURN_INDEX_ID
    assert identity.backing_provider_id == QDRANT_VECTOR_STORE_PROVIDER_ID
    assert identity.capability is MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE
