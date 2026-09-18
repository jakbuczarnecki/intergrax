# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5E — pgvector SessionTurnIndex real-vendor qualification."""

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
from intergrax.integrations.providers.vector_store.pgvector.integration import (
    PGVECTOR_VECTOR_STORE_PROVIDER_ID,
)
from intergrax.integrations.providers.vector_store.qdrant.integration import (
    QDRANT_VECTOR_STORE_PROVIDER_ID,
)
from intergrax.integrations.registry.catalog_manifests import PGVECTOR
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
)
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
from tests.integration.memory.e2e.pgvector_session_turn_index_real_vendor_support import (
    DSN_ENV,
    PgvectorSessionTurnIndexQualificationEnv,
    assert_pgvector_backend_unavailable,
    build_pgvector_memory_rag_stack,
    build_qualification_env,
    build_vector_session_turn_index_store,
    close_pgvector_integration,
    drop_qualification_tenant_rows,
    ensure_pgvector_available,
    pgvector_backend_side_scope_clauses,
    pgvector_reachable,
    pgvector_topology_metadata,
    session_turn_index_store_factory,
    unavailable_probe_dsn,
    unique_qualification_run_id,
)
from tests.unit.memory.test_mem_ent13_provider_qualification import _context

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.docker,
    pytest.mark.qualification,
    pytest.mark.no_ci,
]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MEMORY_ROOT = _REPO_ROOT / "intergrax" / "memory"
_WIRING_ROOT = _REPO_ROOT / "intergrax" / "applications" / "_shared"
_EVIDENCE_SOURCE = "pgvector_session_turn_index_real_vendor_qualification"
_PROOF_KIND = MemoryProviderDurabilityProofKind.REAL_VENDOR_RECONNECT


def _require_pgvector() -> None:
    if not pgvector_reachable():
        pytest.skip(f"PgVector not configured (set {DSN_ENV})")
    try:
        ensure_pgvector_available()
    except Exception as exc:
        pytest.skip(f"PgVector backend unavailable: {exc}")


@pytest.fixture
def pgvector_sti_env() -> PgvectorSessionTurnIndexQualificationEnv:
    _require_pgvector()
    run_id = unique_qualification_run_id()
    env = build_qualification_env(qualification_run_id=run_id)
    yield env
    try:
        drop_qualification_tenant_rows(env)
    except Exception as exc:
        pytest.fail(f"qualification tenant cleanup failed: {exc}")


def test_memory_core_has_no_pgvector_or_psycopg_imports() -> None:
    forbidden_prefixes = ("pgvector", "psycopg", "psycopg2")
    for path in _MEMORY_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod.startswith(forbidden_prefixes):
                        pytest.fail(f"forbidden import in memory core: {path}")
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if mod.startswith(forbidden_prefixes):
                    pytest.fail(f"forbidden import in memory core: {path}")


def test_memory_core_has_no_rag_bootstrap_imports() -> None:
    forbidden = "intergrax.rag.bootstrap"
    for path in _MEMORY_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(forbidden):
                pytest.fail(f"rag bootstrap import in memory core: {path}")


def test_canonical_wiring_has_no_silent_inmemory_sti_fallback() -> None:
    targets = (
        _WIRING_ROOT / "memory_vector_wiring.py",
        _WIRING_ROOT / "memory_wiring.py",
    )
    for path in targets:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body_text = ast.unparse(node.body)
                if "InMemorySessionTurnIndexStore" in body_text:
                    pytest.fail(f"silent InMemory STI fallback in {path}")


async def _run_pgvector_sti_qualification(
    env: PgvectorSessionTurnIndexQualificationEnv,
    run_id: str,
) -> object:
    create, dispose = session_turn_index_store_factory(env)

    class _PgvectorStiFactory(MemoryProviderInstanceFactory[SessionTurnIndexStore]):
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
            backing_provider_id=PGVECTOR_VECTOR_STORE_PROVIDER_ID,
        ),
        context=_context(run_id),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(session_turn_index_store=_PgvectorStiFactory()),
    )


@pytest.mark.asyncio
async def test_pgvector_topology_metadata(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    _, integration = build_vector_session_turn_index_store(pgvector_sti_env)
    try:
        meta = pgvector_topology_metadata(integration)
        assert meta["backend_provider_id"] == PGVECTOR_VECTOR_STORE_PROVIDER_ID
        assert meta["pgvector_extension_version"]
        assert meta["postgres_version"]
        assert meta["table"] == "intergrax_pgvector"
        assert meta["metric"] == "cosine (<=> operator)"
    finally:
        close_pgvector_integration(integration)


@pytest.mark.asyncio
async def test_pgvector_backend_side_scope_filtering_is_sql_enforced() -> None:
    clauses = pgvector_backend_side_scope_clauses()
    assert "tenant_id = %s" in clauses
    assert any("payload @>" in clause for clause in clauses)


@pytest.mark.asyncio
async def test_pgvector_sti_behavioral_qualification_produces_trusted_evidence(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    run_id = pgvector_sti_env.qualification_run_id
    result = await _run_pgvector_sti_qualification(pgvector_sti_env, run_id)
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    evidence = qualification_evidence_from_result(
        result,
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        evidence_source=_EVIDENCE_SOURCE,
    )
    assert evidence is not None
    assert evidence.provider_id == BUILTIN_VECTOR_SESSION_TURN_INDEX_ID
    assert evidence.backing_provider_id == PGVECTOR_VECTOR_STORE_PROVIDER_ID
    assert evidence.qualification_run_id == run_id


@pytest.mark.asyncio
async def test_pgvector_sti_turns_survive_client_reconnect(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    tenant_id = pgvector_sti_env.tenant_id
    session_id = f"sess-reconnect-{pgvector_sti_env.qualification_run_id}"
    marker = f"reconnect-marker-{pgvector_sti_env.qualification_run_id}"
    entry_id = f"entry-reconnect-{pgvector_sti_env.qualification_run_id}"

    store_a, integration_a = build_vector_session_turn_index_store(pgvector_sti_env, tenant_id=tenant_id)
    try:
        await store_a.upsert_turn(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
            message=ChatMessage(role="user", content=marker, entry_id=entry_id),
        )
    finally:
        close_pgvector_integration(integration_a)

    store_b, integration_b = build_vector_session_turn_index_store(pgvector_sti_env, tenant_id=tenant_id)
    try:
        hits = await store_b.search_turns(
            query=marker,
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
        )
        assert any(hit.entry_id == entry_id for hit in hits)
    finally:
        close_pgvector_integration(integration_b)


@pytest.mark.asyncio
async def test_pgvector_sti_durability_evidence_bundle(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    run_id = pgvector_sti_env.qualification_run_id
    result = await _run_pgvector_sti_qualification(pgvector_sti_env, run_id)
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
        backing_provider_id=PGVECTOR_VECTOR_STORE_PROVIDER_ID,
    )
    assert dur.backing_provider_id == PGVECTOR_VECTOR_STORE_PROVIDER_ID
    assert dur.proof_kind is _PROOF_KIND


@pytest.mark.asyncio
async def test_pgvector_cross_tenant_isolation_two_stores(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    marker = f"tenant-secret-{pgvector_sti_env.qualification_run_id}"
    session_id = f"sess-x-{pgvector_sti_env.qualification_run_id}"

    store_a, int_a = build_vector_session_turn_index_store(
        pgvector_sti_env,
        tenant_id="tenant-5e-a",
    )
    try:
        await store_a.upsert_turn(
            tenant_id="tenant-5e-a",
            session_id=session_id,
            user_id="user-x",
            message=ChatMessage(
                role="user",
                content=marker,
                entry_id=f"entry-a-{pgvector_sti_env.qualification_run_id}",
            ),
        )
    finally:
        close_pgvector_integration(int_a)

    store_b, int_b = build_vector_session_turn_index_store(
        pgvector_sti_env,
        tenant_id="tenant-5e-b",
    )
    try:
        await store_b.upsert_turn(
            tenant_id="tenant-5e-b",
            session_id=session_id,
            user_id="user-x",
            message=ChatMessage(
                role="user",
                content="decoy-tenant-b",
                entry_id=f"entry-b-{pgvector_sti_env.qualification_run_id}",
            ),
        )
        hits = await store_b.search_turns(
            query=marker,
            tenant_id="tenant-5e-b",
            session_id=session_id,
            user_id="user-x",
        )
        leaked = [hit for hit in hits if marker in (hit.message.content or "")]
        assert not leaked
    finally:
        close_pgvector_integration(int_b)


@pytest.mark.asyncio
async def test_pgvector_duplicate_turn_id_is_deterministic(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    tenant_id = pgvector_sti_env.tenant_id
    session_id = f"sess-dup-{pgvector_sti_env.qualification_run_id}"
    entry_id = f"entry-dup-{pgvector_sti_env.qualification_run_id}"
    message = ChatMessage(role="user", content="dup-content", entry_id=entry_id)
    store, integration = build_vector_session_turn_index_store(pgvector_sti_env, tenant_id=tenant_id)
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
        close_pgvector_integration(integration)


@pytest.mark.asyncio
async def test_pgvector_tombstone_survives_reconnect(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    tenant_id = pgvector_sti_env.tenant_id
    session_id = f"sess-tomb-{pgvector_sti_env.qualification_run_id}"
    entry_id = f"entry-tomb-{pgvector_sti_env.qualification_run_id}"
    marker = f"tombstone-{pgvector_sti_env.qualification_run_id}"

    store_a, int_a = build_vector_session_turn_index_store(pgvector_sti_env, tenant_id=tenant_id)
    try:
        await store_a.upsert_turn(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
            message=ChatMessage(role="user", content=marker, entry_id=entry_id),
        )
    finally:
        close_pgvector_integration(int_a)

    store_b, int_b = build_vector_session_turn_index_store(pgvector_sti_env, tenant_id=tenant_id)
    try:
        await store_b.tombstone_turn(entry_id)
    finally:
        close_pgvector_integration(int_b)

    store_c, int_c = build_vector_session_turn_index_store(pgvector_sti_env, tenant_id=tenant_id)
    try:
        hits = await store_c.search_turns(
            query=marker,
            tenant_id=tenant_id,
            session_id=session_id,
            user_id="qual-user",
        )
        assert not any(hit.entry_id == entry_id for hit in hits)
    finally:
        close_pgvector_integration(int_c)


@pytest.mark.asyncio
async def test_pgvector_backend_unavailable_fails_explicitly() -> None:
    assert_pgvector_backend_unavailable(unavailable_probe_dsn())


def _sti_qualification_registry(
    run_id: str,
    *,
    backing_provider_id: str = PGVECTOR_VECTOR_STORE_PROVIDER_ID,
) -> object:
    from intergrax.memory.provider_qualification.in_memory_evidence_registry import (
        InMemoryMemoryProviderQualificationEvidenceRegistry,
    )

    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    registry.register(
        MemoryProviderQualificationEvidence(
            provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
            capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id=run_id,
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source=_EVIDENCE_SOURCE,
            backing_provider_id=backing_provider_id,
        ),
    )
    return registry


def test_product_session_manager_requires_trusted_pgvector_sti_evidence(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5e.product.gate")
    env.memory_profile = MemoryProfile(
        enable_user_memory=False,
        enable_long_term_memory=False,
        enable_session_vector_index=True,
    )
    tenant_id = pgvector_sti_env.tenant_id
    stack, integration = build_pgvector_memory_rag_stack(pgvector_sti_env, tenant_id=tenant_id)
    wiring = MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        organization_profile_store=None,
    )
    try:
        with pytest.raises(MemoryProviderAdmissionError) as exc_info:
            build_session_manager_from_environment(
                env,
                tenant_id=tenant_id,
                memory_wiring=wiring,
                rag_stack=stack,
                integration_profile=IntegrationProfile(vector_store=PGVECTOR),
            )
        assert (
            exc_info.value.reason_code
            is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING
        )
    finally:
        close_pgvector_integration(integration)


@pytest.mark.asyncio
async def test_product_episodic_recall_with_trusted_pgvector_sti_evidence(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5e.product.e2e")
    env.memory_profile = MemoryProfile(
        enable_user_memory=False,
        enable_long_term_memory=False,
        enable_session_vector_index=True,
    )
    tenant_id = pgvector_sti_env.tenant_id
    turn_text = f"product-e2e-{pgvector_sti_env.qualification_run_id}"
    registry = _sti_qualification_registry(pgvector_sti_env.qualification_run_id)
    integration_profile = IntegrationProfile(vector_store=PGVECTOR)
    wiring = MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        organization_profile_store=None,
    )

    stack_a, integration_a = build_pgvector_memory_rag_stack(pgvector_sti_env, tenant_id=tenant_id)
    session_a = build_session_manager_from_environment(
        env,
        tenant_id=tenant_id,
        memory_wiring=wiring,
        rag_stack=stack_a,
        integration_profile=integration_profile,
        qualification_evidence_registry=registry,
    )
    assert isinstance(session_a._session_turn_index_store, VectorSessionTurnIndexStore)
    await session_a.create_session(
        tenant_id=tenant_id,
        session_id="sess-product-e2e",
        user_id="qual-user",
        workspace_id="default",
    )
    await session_a.append_message(
        tenant_id=tenant_id,
        session_id="sess-product-e2e",
        message=ChatMessage(role="user", content=turn_text),
    )
    close_pgvector_integration(integration_a)

    stack_b, integration_b = build_pgvector_memory_rag_stack(pgvector_sti_env, tenant_id=tenant_id)
    session_b = build_session_manager_from_environment(
        env,
        tenant_id=tenant_id,
        memory_wiring=wiring,
        rag_stack=stack_b,
        integration_profile=integration_profile,
        qualification_evidence_registry=registry,
    )
    try:
        hits = await session_b.search_session_semantic_recall(
            tenant_id=tenant_id,
            session_id="sess-product-e2e",
            user_id="qual-user",
            query=turn_text,
        )
        assert hits
        assert turn_text in hits[0]["text"]
    finally:
        close_pgvector_integration(integration_b)


def test_product_pgvector_runtime_with_qdrant_sti_evidence_fails(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications._shared.memory_vector_wiring import build_session_turn_index_store

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5e.qdrant.evidence")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    tenant_id = pgvector_sti_env.tenant_id
    stack, integration = build_pgvector_memory_rag_stack(pgvector_sti_env, tenant_id=tenant_id)
    registry = _sti_qualification_registry(
        pgvector_sti_env.qualification_run_id,
        backing_provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
    )
    try:
        with pytest.raises(MemoryProviderAdmissionError) as exc_info:
            build_session_turn_index_store(
                env,
                tenant_id=tenant_id,
                rag_stack=stack,
                integration_profile=IntegrationProfile(vector_store=PGVECTOR),
                qualification_evidence_registry=registry,
            )
        assert exc_info.value.reason_code in {
            MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING,
            MemoryProviderAdmissionReasonCode.PROVIDER_BACKING_IDENTITY_MISMATCH,
        }
    finally:
        close_pgvector_integration(integration)


def test_product_pgvector_runtime_with_adapter_only_evidence_fails(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications._shared.memory_vector_wiring import build_session_turn_index_store

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5e.adapter.evidence")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    tenant_id = pgvector_sti_env.tenant_id
    stack, integration = build_pgvector_memory_rag_stack(pgvector_sti_env, tenant_id=tenant_id)
    from intergrax.memory.provider_qualification.in_memory_evidence_registry import (
        InMemoryMemoryProviderQualificationEvidenceRegistry,
    )

    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    registry.register(
        MemoryProviderQualificationEvidence(
            provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
            capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id=pgvector_sti_env.qualification_run_id,
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source=_EVIDENCE_SOURCE,
            backing_provider_id=None,
        ),
    )
    try:
        with pytest.raises(MemoryProviderAdmissionError) as exc_info:
            build_session_turn_index_store(
                env,
                tenant_id=tenant_id,
                rag_stack=stack,
                integration_profile=IntegrationProfile(vector_store=PGVECTOR),
                qualification_evidence_registry=registry,
            )
        assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING
    finally:
        close_pgvector_integration(integration)


@pytest.mark.asyncio
async def test_application_episodic_recall_after_pgvector_reconnect(
    pgvector_sti_env: PgvectorSessionTurnIndexQualificationEnv,
) -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.memory_profile = MemoryProfile(
        enable_user_memory=False,
        enable_long_term_memory=False,
        enable_session_vector_index=True,
    )
    tenant_id = pgvector_sti_env.tenant_id
    turn_text = f"episodic-e2e-{pgvector_sti_env.qualification_run_id}"

    stack_a, integration_a = build_pgvector_memory_rag_stack(pgvector_sti_env, tenant_id=tenant_id)
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
        integration_profile=IntegrationProfile(vector_store=PGVECTOR),
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
    close_pgvector_integration(integration_a)

    stack_b, integration_b = build_pgvector_memory_rag_stack(pgvector_sti_env, tenant_id=tenant_id)
    session_b = build_session_manager_from_environment(
        env,
        tenant_id=tenant_id,
        memory_wiring=wiring,
        rag_stack=stack_b,
        integration_profile=IntegrationProfile(vector_store=PGVECTOR),
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
        close_pgvector_integration(integration_b)


def test_composite_identity_descriptor() -> None:
    identity = builtin_session_turn_index_store_identity(
        BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        backing_provider_id=PGVECTOR_VECTOR_STORE_PROVIDER_ID,
    )
    assert identity.provider_id == BUILTIN_VECTOR_SESSION_TURN_INDEX_ID
    assert identity.backing_provider_id == PGVECTOR_VECTOR_STORE_PROVIDER_ID
    assert identity.capability is MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE


def test_postgresql_service_restart_not_executed() -> None:
    """P2: service restart proof deferred; reconnect only qualifies as REAL_VENDOR_RECONNECT."""

    assert _PROOF_KIND is MemoryProviderDurabilityProofKind.REAL_VENDOR_RECONNECT
