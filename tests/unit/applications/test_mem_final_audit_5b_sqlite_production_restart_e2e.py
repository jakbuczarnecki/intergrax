# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5B — PRODUCT SQLite admission + application restart E2E."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from intergrax.applications._shared.memory_wiring import (
    build_session_manager_from_environment,
    resolve_memory_platform_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
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
from intergrax.memory.contracts.provider_identity import (
    MemoryProviderIdentity,
    MemoryProviderIdentitySource,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderDescriptor,
)
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    MemoryProviderInstanceFactory,
    build_user_profile_admission_evidence_from_durable_qualification,
)
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
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

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _product_sqlite_env(tmp_path: Path, profile_id: str) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=profile_id)
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    return env


async def _run_sqlite_durable_qualification(
    db_path: Path,
    run_id: str,
) -> tuple[object, Callable[[], SQLiteUserProfileStore], Callable[[SQLiteUserProfileStore], Awaitable[None]]]:
    def _create() -> SQLiteUserProfileStore:
        return SQLiteUserProfileStore(str(db_path))

    async def _dispose(store: SQLiteUserProfileStore) -> None:
        store.close()

    class _SqliteFactory(MemoryProviderInstanceFactory[SQLiteUserProfileStore]):
        async def create(self) -> SQLiteUserProfileStore:
            return _create()

        async def dispose(self, instance: SQLiteUserProfileStore) -> None:
            await _dispose(instance)

    evidence = await run_durable_user_profile_production_qualification(
        descriptor=MemoryProviderDescriptor(
            provider_id="sqlite.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(run_id),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_SqliteFactory()),
        create_store=_create,
        dispose_store=_dispose,
        durability_mode=DurabilityQualificationMode.DURABLE_PERSISTENCE,
    )
    return evidence, _create, _dispose


@pytest.mark.asyncio
async def test_product_sqlite_admission_from_real_durable_harness_bundle(tmp_path: Path) -> None:
    db_path = tmp_path / "admission.db"
    evidence, _, _ = await _run_sqlite_durable_qualification(db_path, "mem-5b-admission")
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
    )
    env = _product_sqlite_env(tmp_path, "mem.5b.admission.bundle")
    wiring = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    assert isinstance(wiring.user_profile_store, SQLiteUserProfileStore)
    assert wiring.user_profile_store.memory_provider_id == "sqlite.user_profile"
    wiring.user_profile_store.close()


@pytest.mark.asyncio
async def test_mismatched_qualification_run_ids_fail_admission(tmp_path: Path) -> None:
    db_path = tmp_path / "mismatch.db"
    evidence, _, _ = await _run_sqlite_durable_qualification(db_path, "run-behavioral")
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
    )
    qual_registry = bundle.admission_evidence.qualification_registry
    dur_registry = bundle.admission_evidence.durability_registry

    store = SQLiteUserProfileStore(str(db_path))
    classification = classify_user_profile_store_provider(store)
    identity = MemoryProviderIdentity(
        provider_id="sqlite.user_profile",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.BUILT_IN,
    )
    qual_lookup = lookup_trusted_user_profile_qualification_evidence(qual_registry, identity)
    dur_lookup = lookup_trusted_user_profile_durability_evidence(dur_registry, identity)
    assert qual_lookup.evidence is not None
    assert dur_lookup.evidence is not None
    dur_evidence = dur_lookup.evidence
    assert dur_evidence is not None
    from intergrax.memory.provider_qualification import InMemoryMemoryProviderDurabilityEvidenceRegistry
    from intergrax.memory.contracts.provider_durability_evidence import (
        MemoryProviderDurabilityEvidence,
        MemoryProviderDurabilityProofKind,
        MemoryProviderTrustedDurabilityStatus,
    )

    mismatched = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    mismatched.register(
        MemoryProviderDurabilityEvidence(
            provider_id=dur_evidence.provider_id,
            capability=dur_evidence.capability,
            durability_status=MemoryProviderTrustedDurabilityStatus.DURABLE,
            qualification_run_id="run-different",
            reference_time_iso=dur_evidence.reference_time_iso,
            evidence_source=dur_evidence.evidence_source,
            proof_kind=MemoryProviderDurabilityProofKind.RESTART_REOPEN,
            provider_version=dur_evidence.provider_version,
        ),
    )
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        qual_lookup,
        lookup_trusted_user_profile_durability_evidence(mismatched, identity),
    )
    store.close()
    assert not evaluation.admitted
    assert evaluation.reason_code is MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISMATCH


def test_product_sqlite_without_durability_evidence_still_fails(tmp_path: Path) -> None:
    env = _product_sqlite_env(tmp_path, "mem.5b.missing.durability")
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        resolve_memory_platform_wiring(
            env,
            qualification_evidence_registry=_evidence_for_provider("sqlite.user_profile"),
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING


@pytest.mark.asyncio
async def test_product_session_recall_after_sqlite_store_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "session-restart.db"
    evidence, _, _ = await _run_sqlite_durable_qualification(db_path, "mem-5b-session")
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
    )
    env = _product_sqlite_env(tmp_path, "mem.5b.session.restart")
    tenant_id = "tenant-5b"

    wiring_a = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_a = build_session_manager_from_environment(
        env,
        memory_wiring=wiring_a,
        tenant_id=tenant_id,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    plane_a = session_a.memory_control_plane
    assert plane_a is not None
    identity = RequestIdentity(
        tenant_id=tenant_id,
        user_id="user-5b",
        principal_type=PrincipalType.USER,
        auth_subject="user-5b",
    )
    scope = user_memory_scope(identity)
    await plane_a.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="audit-5b-recall-marker"),
    )
    wiring_a.user_profile_store.close()

    wiring_b = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_b = build_session_manager_from_environment(
        env,
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
        MemoryControlRecallRequest(query="audit-5b", top_k=5),
    )
    assert any(item.content == "audit-5b-recall-marker" for item in recall.items)
    wiring_b.user_profile_store.close()


@pytest.mark.asyncio
async def test_product_forget_survives_sqlite_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "forget-restart.db"
    evidence, _, _ = await _run_sqlite_durable_qualification(db_path, "mem-5b-forget")
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
    )
    env = _product_sqlite_env(tmp_path, "mem.5b.forget")
    tenant_id = "tenant-forget"

    wiring_a = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_a = build_session_manager_from_environment(
        env,
        memory_wiring=wiring_a,
        tenant_id=tenant_id,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    plane_a = session_a.memory_control_plane
    assert plane_a is not None
    identity = RequestIdentity(
        tenant_id=tenant_id,
        user_id="user-forget",
        principal_type=PrincipalType.USER,
        auth_subject="user-forget",
    )
    scope = user_memory_scope(identity)
    remembered = await plane_a.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="to-be-forgotten"),
    )
    entry_id = remembered.entry_id
    assert entry_id is not None
    wiring_a.user_profile_store.close()

    wiring_b = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_b = build_session_manager_from_environment(
        env,
        memory_wiring=wiring_b,
        tenant_id=tenant_id,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    plane_b = session_b.memory_control_plane
    assert plane_b is not None
    await plane_b.forget(
        identity,
        scope,
        MemoryControlForgetRequest(entry_id=entry_id),
    )
    wiring_b.user_profile_store.close()

    wiring_c = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    session_c = build_session_manager_from_environment(
        env,
        memory_wiring=wiring_c,
        tenant_id=tenant_id,
        qualification_evidence_registry=bundle.admission_evidence.qualification_registry,
        durability_evidence_registry=bundle.admission_evidence.durability_registry,
    )
    plane_c = session_c.memory_control_plane
    assert plane_c is not None
    recall = await plane_c.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="forgotten", top_k=10),
    )
    assert not any(item.content == "to-be-forgotten" for item in recall.items)
    wiring_c.user_profile_store.close()
