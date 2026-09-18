# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5C — Mongo UserProfile real-vendor qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.memory_wiring import build_session_manager_from_environment
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from intergrax.memory.contracts.memory_control import (
    MemoryControlForgetRequest,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
    classify_user_profile_store_provider,
    evaluate_production_persistent_user_profile_admission,
    lookup_trusted_user_profile_durability_evidence,
    lookup_trusted_user_profile_qualification_evidence,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityProofKind,
    MemoryProviderTrustedDurabilityStatus,
)
from intergrax.memory.contracts.provider_identity import (
    BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
    MemoryProviderIdentity,
    MemoryProviderIdentitySource,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderDescriptor,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    MemoryProviderInstanceFactory,
    build_user_profile_admission_evidence_from_durable_qualification,
)
from intergrax.memory.stores.document_store_user_profile_store import DocumentStoreUserProfileStore
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_memory import (
    MemoryImportance,
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    ensure_mongo_running,
    proof_env,
    require_docker_for_harden_4f_proof,
    resolve_mongodb_uri,
)
from tests.integration.memory.e2e.mongo_user_profile_real_vendor_support import (
    MongoUserProfileQualificationEnv,
    apply_mongo_env,
    assert_unique_document_key_index,
    build_qualification_env,
    close_mongo_wiring,
    count_profile_documents,
    drop_qualification_collection,
    mongo_topology_metadata,
    mongo_user_profile_store_factory,
    open_mongo_document_store,
    product_mongo_environment,
    resolve_product_mongo_wiring,
    unique_qualification_run_id,
)
from tests.unit.applications.test_mem_audit5a_production_provider_admission import (
    _evidence_for_provider,
    _persistent_memory_profile,
)
from tests.unit.memory.durable_provider_qualification_harness import (
    DurabilityQualificationMode,
    run_durable_user_profile_production_qualification,
    user_profile_qualification_request,
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
_EVIDENCE_SOURCE = "mongo_real_vendor_qualification"
_PROOF_KIND = MemoryProviderDurabilityProofKind.REAL_VENDOR_RECONNECT


@pytest.fixture
def mongo_qualification_env(monkeypatch: pytest.MonkeyPatch) -> MongoUserProfileQualificationEnv:
    require_docker_for_harden_4f_proof()
    run_id = unique_qualification_run_id()
    env = build_qualification_env(uri=resolve_mongodb_uri(), qualification_run_id=run_id)
    for key, value in proof_env(collection_name=env.collection).items():
        if key.startswith("INTERGRAX_MONGODB"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("INTERGRAX_MONGODB_DATABASE", env.database)
    ensure_mongo_running()
    yield env
    try:
        drop_qualification_collection(env)
    except Exception as exc:
        pytest.fail(f"qualification collection cleanup failed: {exc}")


async def _run_mongo_durable_qualification(
    env: MongoUserProfileQualificationEnv,
    run_id: str,
) -> object:
    create, dispose = mongo_user_profile_store_factory(env)

    class _MongoFactory(MemoryProviderInstanceFactory[DocumentStoreUserProfileStore]):
        async def create(self) -> DocumentStoreUserProfileStore:
            store = create()
            assert isinstance(store, DocumentStoreUserProfileStore)
            return store

        async def dispose(self, instance: DocumentStoreUserProfileStore) -> None:
            await dispose(instance)

    return await run_durable_user_profile_production_qualification(
        descriptor=MemoryProviderDescriptor(
            provider_id=BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(run_id),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_MongoFactory()),
        create_store=create,
        dispose_store=dispose,
        durability_mode=DurabilityQualificationMode.DURABLE_PERSISTENCE,
    )


def test_memory_core_has_no_mongodb_sdk_imports() -> None:
    forbidden = ("pymongo", "mongodb")
    for path in _MEMORY_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(token in alias.name for token in forbidden), path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(token in node.module for token in forbidden), path


@pytest.mark.asyncio
async def test_mongo_document_store_health_and_indexes(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    store = open_mongo_document_store(mongo_qualification_env)
    try:
        meta = mongo_topology_metadata(store)
        assert meta["server_version"]
        assert meta["driver_version"]
        assert meta["backend_provider_id"] == "mongodb"
        assert_unique_document_key_index(store)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_mongo_user_profile_durable_harness_produces_trusted_bundle(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    run_id = mongo_qualification_env.qualification_run_id
    evidence = await _run_mongo_durable_qualification(mongo_qualification_env, run_id)
    assert evidence.canonical.status is MemoryProviderQualificationStatus.QUALIFIED
    assert evidence.production_durable_qualified
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
        durability_evidence_source=_EVIDENCE_SOURCE,
        proof_kind=_PROOF_KIND,
    )
    assert bundle.qualification_run_id == evidence.canonical.qualification_run_id
    dur = bundle.admission_evidence.durability_registry.resolve(
        BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        MemoryProviderCapabilityKind.USER_PROFILE_STORE,
    )
    assert dur.evidence is not None
    assert dur.evidence.proof_kind is _PROOF_KIND
    assert dur.evidence.evidence_source == _EVIDENCE_SOURCE
    assert dur.evidence.durability_status is MemoryProviderTrustedDurabilityStatus.DURABLE


async def test_unicode_and_memory_entries_survive_mongo_reconnect(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    create, dispose = mongo_user_profile_store_factory(mongo_qualification_env)
    marker = "Instrukcja: żółć 🚀\nmulti-line"
    profile = UserProfile(
        identity=UserIdentity(user_id="u-pl", display_name="Użytkownik"),
        preferences=UserPreferences(preferred_language="pl", tone="formal"),
        system_instructions=marker,
        memory_entries=[
            UserProfileMemoryEntry(
                entry_id="e1",
                content="Lubi kawę",
                kind=MemoryKind.PREFERENCE,
                importance=MemoryImportance.HIGH,
            ),
        ],
    )
    store_a = create()
    await store_a.save_profile(tenant_id="tenant-unicode", profile=profile)
    await dispose(store_a)

    store_b = create()
    loaded = await store_b.get_profile(tenant_id="tenant-unicode", user_id="u-pl")
    await dispose(store_b)

    assert loaded.system_instructions == marker
    assert loaded.memory_entries[0].content == "Lubi kawę"


async def test_tenant_isolation_on_real_mongo(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    create, dispose = mongo_user_profile_store_factory(mongo_qualification_env)
    store = create()
    await store.save_profile(
        tenant_id="tenant-a",
        profile=UserProfile(
            identity=UserIdentity(user_id="shared-user", display_name="A"),
            preferences=UserPreferences(),
        ),
    )
    await store.save_profile(
        tenant_id="tenant-b",
        profile=UserProfile(
            identity=UserIdentity(user_id="shared-user", display_name="B"),
            preferences=UserPreferences(),
        ),
    )
    await dispose(store)

    reopened = create()
    loaded_a = await reopened.get_profile(tenant_id="tenant-a", user_id="shared-user")
    loaded_b = await reopened.get_profile(tenant_id="tenant-b", user_id="shared-user")
    await dispose(reopened)
    assert loaded_a.identity.display_name == "A"
    assert loaded_b.identity.display_name == "B"


async def test_user_isolation_on_real_mongo(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    create, dispose = mongo_user_profile_store_factory(mongo_qualification_env)
    store = create()
    await store.save_profile(
        tenant_id="tenant-a",
        profile=UserProfile(
            identity=UserIdentity(user_id="user-a", display_name="User A"),
            preferences=UserPreferences(),
        ),
    )
    await store.save_profile(
        tenant_id="tenant-a",
        profile=UserProfile(
            identity=UserIdentity(user_id="user-b", display_name="User B"),
            preferences=UserPreferences(),
        ),
    )
    await dispose(store)

    reopened = create()
    assert (await reopened.get_profile(tenant_id="tenant-a", user_id="user-a")).identity.display_name == "User A"
    assert (await reopened.get_profile(tenant_id="tenant-a", user_id="user-b")).identity.display_name == "User B"
    await dispose(reopened)


async def test_idempotent_save_single_logical_document(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    create, dispose = mongo_user_profile_store_factory(mongo_qualification_env)
    profile = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(preferred_language="pl"),
        system_instructions="same",
    )
    store = create()
    await store.save_profile(tenant_id="tenant-idem", profile=profile)
    await store.save_profile(tenant_id="tenant-idem", profile=profile)
    await dispose(store)

    probe = open_mongo_document_store(mongo_qualification_env)
    try:
        assert count_profile_documents(probe, tenant_id="tenant-idem", user_id="u1") == 1
    finally:
        probe.close()


async def test_delete_durability_sibling_tenant_safe(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    create, dispose = mongo_user_profile_store_factory(mongo_qualification_env)
    store = create()
    for tenant in ("tenant-a", "tenant-b"):
        await store.save_profile(
            tenant_id=tenant,
            profile=UserProfile(
                identity=UserIdentity(user_id="u1", display_name=tenant),
                preferences=UserPreferences(),
            ),
        )
    await store.delete_profile(tenant_id="tenant-a", user_id="u1")
    await dispose(store)

    reopened = create()
    deleted = await reopened.get_profile(tenant_id="tenant-a", user_id="u1")
    sibling = await reopened.get_profile(tenant_id="tenant-b", user_id="u1")
    await dispose(reopened)
    assert deleted.identity.display_name is None
    assert sibling.identity.display_name == "tenant-b"


async def test_triple_client_reconnect_cycle(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    create, dispose = mongo_user_profile_store_factory(mongo_qualification_env)
    for label in ("a", "b", "c"):
        store = create()
        await store.save_profile(
            tenant_id="tenant-cycle",
            profile=UserProfile(
                identity=UserIdentity(user_id="u1"),
                preferences=UserPreferences(),
                system_instructions=f"cycle-{label}",
            ),
        )
        await dispose(store)

    final = create()
    loaded = await final.get_profile(tenant_id="tenant-cycle", user_id="u1")
    await dispose(final)
    assert loaded.system_instructions == "cycle-c"


@pytest.mark.asyncio
async def test_unreachable_mongo_endpoint_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_uri = "mongodb://mem-audit-5c-unreachable.invalid:27017/?serverSelectionTimeoutMS=500"
    env = build_qualification_env(uri=bad_uri, qualification_run_id=unique_qualification_run_id("mem-5c-fail"))
    apply_mongo_env(monkeypatch, env)
    create, dispose = mongo_user_profile_store_factory(env)
    failure_types = (IntegrationConfigurationError, OSError, ConnectionError, TimeoutError)
    try:
        store = create()
    except failure_types:
        return
    try:
        await store.save_profile(
            tenant_id="tenant-fail",
            profile=UserProfile(
                identity=UserIdentity(user_id="u1"),
                preferences=UserPreferences(),
                system_instructions="must-not-commit",
            ),
        )
        pytest.fail("unreachable Mongo endpoint must not accept profile save")
    except failure_types:
        pass
    finally:
        await dispose(store)


async def test_product_mongo_admission_passes_with_real_evidence(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    run_id = mongo_qualification_env.qualification_run_id
    evidence = await _run_mongo_durable_qualification(mongo_qualification_env, run_id)
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
        durability_evidence_source=_EVIDENCE_SOURCE,
        proof_kind=_PROOF_KIND,
    )
    application = product_mongo_environment(
        profile_id="mem.5c.admission",
        env=mongo_qualification_env,
    )
    wiring = resolve_product_mongo_wiring(
        application,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    assert isinstance(wiring.user_profile_store, DocumentStoreUserProfileStore)
    assert wiring.user_profile_store.memory_provider_id == BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID
    close_mongo_wiring(wiring)


def test_product_mongo_without_durability_evidence_fails(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    application = product_mongo_environment(
        profile_id="mem.5c.missing.durability",
        env=mongo_qualification_env,
    )
    application.memory_profile = _persistent_memory_profile()
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        resolve_product_mongo_wiring(
            application,
            qualification_evidence_registry=_evidence_for_provider(BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID),
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING


async def test_product_session_recall_after_mongo_client_reconnect(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    run_id = mongo_qualification_env.qualification_run_id
    evidence = await _run_mongo_durable_qualification(mongo_qualification_env, run_id)
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
        durability_evidence_source=_EVIDENCE_SOURCE,
        proof_kind=_PROOF_KIND,
    )
    application = product_mongo_environment(profile_id="mem.5c.session", env=mongo_qualification_env)
    tenant_id = "tenant-5c"

    wiring_a = resolve_product_mongo_wiring(
        application,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_a = build_session_manager_from_environment(
        application,
        memory_wiring=wiring_a,
        tenant_id=tenant_id,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    plane_a = session_a.memory_control_plane
    assert plane_a is not None
    identity = RequestIdentity(
        tenant_id=tenant_id,
        user_id="user-5c",
        principal_type=PrincipalType.USER,
        auth_subject="user-5c",
    )
    scope = user_memory_scope(identity)
    await plane_a.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="audit-5c-recall-marker"),
    )
    close_mongo_wiring(wiring_a)

    wiring_b = resolve_product_mongo_wiring(
        application,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_b = build_session_manager_from_environment(
        application,
        memory_wiring=wiring_b,
        tenant_id=tenant_id,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    plane_b = session_b.memory_control_plane
    assert plane_b is not None
    recall = await plane_b.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="audit-5c", top_k=5),
    )
    assert any(item.content == "audit-5c-recall-marker" for item in recall.items)
    close_mongo_wiring(wiring_b)


async def test_cross_tenant_e2e_recall_isolation(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    run_id = mongo_qualification_env.qualification_run_id
    evidence = await _run_mongo_durable_qualification(mongo_qualification_env, run_id)
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
        durability_evidence_source=_EVIDENCE_SOURCE,
        proof_kind=_PROOF_KIND,
    )
    application = product_mongo_environment(profile_id="mem.5c.cross", env=mongo_qualification_env)
    wiring = resolve_product_mongo_wiring(
        application,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_a = build_session_manager_from_environment(
        application,
        memory_wiring=wiring,
        tenant_id="tenant-a",
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    plane_a = session_a.memory_control_plane
    assert plane_a is not None
    identity_a = RequestIdentity(
        tenant_id="tenant-a",
        user_id="user-x",
        principal_type=PrincipalType.USER,
        auth_subject="user-x",
    )
    await plane_a.remember(
        identity_a,
        user_memory_scope(identity_a),
        MemoryControlRememberRequest(content="secret-tenant-a"),
    )
    close_mongo_wiring(wiring)

    wiring_b = resolve_product_mongo_wiring(
        application,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_b = build_session_manager_from_environment(
        application,
        memory_wiring=wiring_b,
        tenant_id="tenant-b",
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    plane_b = session_b.memory_control_plane
    assert plane_b is not None
    identity_b = RequestIdentity(
        tenant_id="tenant-b",
        user_id="user-x",
        principal_type=PrincipalType.USER,
        auth_subject="user-x",
    )
    recall = await plane_b.recall(
        identity_b,
        user_memory_scope(identity_b),
        MemoryControlRecallRequest(query="secret", top_k=10),
    )
    assert not any(item.content == "secret-tenant-a" for item in recall.items)
    close_mongo_wiring(wiring_b)


async def test_admission_evaluation_pass_with_trusted_evidence(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    run_id = mongo_qualification_env.qualification_run_id
    evidence = await _run_mongo_durable_qualification(mongo_qualification_env, run_id)
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
        durability_evidence_source=_EVIDENCE_SOURCE,
        proof_kind=_PROOF_KIND,
    )
    create, dispose = mongo_user_profile_store_factory(mongo_qualification_env)
    store = create()
    classification = classify_user_profile_store_provider(store)
    identity = MemoryProviderIdentity(
        provider_id=BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.BUILT_IN,
    )
    qual_lookup = lookup_trusted_user_profile_qualification_evidence(
        bundle.admission_evidence.qualification_registry,
        identity,
    )
    dur_lookup = lookup_trusted_user_profile_durability_evidence(
        bundle.admission_evidence.durability_registry,
        identity,
    )
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        qual_lookup,
        dur_lookup,
    )
    await dispose(store)
    assert evaluation.admitted


def test_mongo_write_semantics_documented_as_replace_upsert() -> None:
    """UserProfile persistence uses DocumentStore.put → Mongo replace_one(upsert=True)."""
    source = (_REPO_ROOT / "intergrax" / "integrations" / "providers" / "document_store" / "mongodb" / "client.py").read_text(
        encoding="utf-8",
    )
    assert "replace_one" in source and "upsert=True" in source


async def test_closed_store_rejects_use(
    mongo_qualification_env: MongoUserProfileQualificationEnv,
) -> None:
    store = open_mongo_document_store(mongo_qualification_env)
    store.close()
    with pytest.raises(IntegrationConfigurationError):
        store.get("tenant", "user")
