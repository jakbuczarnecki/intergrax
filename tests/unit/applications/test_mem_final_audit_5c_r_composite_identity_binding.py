# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5C-R — composite adapter/backend qualification identity binding."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.memory_provider_admission import (
    validate_memory_platform_wiring_admission,
)
from intergrax.applications._shared.memory_wiring import (
    MemoryPlatformWiring,
    _resolve_baseline_memory_platform_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MONGODB_DOCUMENT_STORE_PROVIDER_ID,
)
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
    MemoryProviderDurability,
    classify_user_profile_store_provider,
    evaluate_production_persistent_user_profile_admission,
    lookup_trusted_user_profile_durability_evidence,
    lookup_trusted_user_profile_qualification_evidence,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityProofKind,
    MemoryProviderDurabilityEvidence,
    MemoryProviderTrustedDurabilityStatus,
)
from intergrax.memory.contracts.provider_identity import (
    BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
    MemoryProviderIdentity,
    MemoryProviderIdentitySource,
    builtin_user_profile_store_identity,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderDescriptor,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidence,
    MemoryProviderQualificationEvidenceResolveStatus,
    qualification_evidence_from_result,
)
from intergrax.memory.provider_qualification import (
    InMemoryMemoryProviderDurabilityEvidenceRegistry,
    InMemoryMemoryProviderQualificationEvidenceRegistry,
    MemoryProviderCapabilityFactories,
    build_user_profile_admission_evidence_from_durable_qualification,
)
from intergrax.memory.provider_qualification.factory import MemoryProviderInstanceFactory
from intergrax.memory.stores.document_store_user_profile_store import DocumentStoreUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from tests.unit.applications.test_mem_audit5a_production_provider_admission import (
    _durability_for_provider,
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

_IN_MEMORY_DOCUMENT_STORE_BACKING_ID = "in_memory_document_store"


def _mongo_composite_identity() -> MemoryProviderIdentity:
    return builtin_user_profile_store_identity(
        BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        backing_provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    )


def _mongo_evidence_bundle(run_id: str = "run-mongo-5cr"):
    qual = MemoryProviderQualificationEvidence(
        provider_id=BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        status=MemoryProviderQualificationStatus.QUALIFIED,
        qualification_run_id=run_id,
        reference_time_iso="2025-01-01T00:00:00+00:00",
        evidence_source="test",
        backing_provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    )
    dur = MemoryProviderDurabilityEvidence(
        provider_id=BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        durability_status=MemoryProviderTrustedDurabilityStatus.DURABLE,
        qualification_run_id=run_id,
        reference_time_iso="2025-01-01T00:00:00+00:00",
        evidence_source="test",
        proof_kind=MemoryProviderDurabilityProofKind.REAL_VENDOR_RECONNECT,
        backing_provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    )
    qual_registry = InMemoryMemoryProviderQualificationEvidenceRegistry((qual,))
    dur_registry = InMemoryMemoryProviderDurabilityEvidenceRegistry((dur,))
    return qual_registry, dur_registry


def test_registry_distinguishes_adapter_backing_pairs() -> None:
    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    registry.register(
        MemoryProviderQualificationEvidence(
            provider_id=BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id="run-mongo",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test",
            backing_provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
        ),
    )
    registry.register(
        MemoryProviderQualificationEvidence(
            provider_id=BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id="run-inmem",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test",
            backing_provider_id=_IN_MEMORY_DOCUMENT_STORE_BACKING_ID,
        ),
    )
    mongo_lookup = registry.resolve(
        BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        None,
        MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    )
    assert mongo_lookup.evidence is not None
    assert mongo_lookup.evidence.qualification_run_id == "run-mongo"
    generic_lookup = registry.resolve(
        BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        MemoryProviderCapabilityKind.USER_PROFILE_STORE,
    )
    assert generic_lookup.resolve_status is MemoryProviderQualificationEvidenceResolveStatus.MISSING


def test_mongo_evidence_with_mongo_runtime_identity_passes() -> None:
    store = DocumentStoreUserProfileStore(InMemoryDocumentStore())
    classification = classify_user_profile_store_provider(store)
    identity = _mongo_composite_identity()
    qual_registry, dur_registry = _mongo_evidence_bundle()
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        lookup_trusted_user_profile_qualification_evidence(qual_registry, identity),
        lookup_trusted_user_profile_durability_evidence(dur_registry, identity),
    )
    assert evaluation.admitted


def test_mongo_evidence_with_in_memory_backing_runtime_fails() -> None:
    store = DocumentStoreUserProfileStore(InMemoryDocumentStore())
    classification = classify_user_profile_store_provider(store)
    identity = builtin_user_profile_store_identity(
        BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        backing_provider_id=_IN_MEMORY_DOCUMENT_STORE_BACKING_ID,
    )
    qual_registry, dur_registry = _mongo_evidence_bundle()
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        lookup_trusted_user_profile_qualification_evidence(qual_registry, identity),
        lookup_trusted_user_profile_durability_evidence(dur_registry, identity),
    )
    assert not evaluation.admitted
    assert (
        evaluation.reason_code
        in {
            MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING,
            MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING,
        }
    )


def test_mongo_evidence_with_missing_runtime_backing_fails() -> None:
    store = DocumentStoreUserProfileStore(InMemoryDocumentStore())
    classification = classify_user_profile_store_provider(store)
    identity = builtin_user_profile_store_identity(BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID)
    qual_registry, dur_registry = _mongo_evidence_bundle()
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        lookup_trusted_user_profile_qualification_evidence(qual_registry, identity),
        lookup_trusted_user_profile_durability_evidence(dur_registry, identity),
    )
    assert not evaluation.admitted


def test_generic_adapter_evidence_with_mongo_runtime_fails() -> None:
    store = DocumentStoreUserProfileStore(InMemoryDocumentStore())
    classification = classify_user_profile_store_provider(store)
    identity = _mongo_composite_identity()
    qual_registry = _evidence_for_provider(BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID)
    dur_registry = _durability_for_provider(BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID)
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        lookup_trusted_user_profile_qualification_evidence(qual_registry, identity),
        lookup_trusted_user_profile_durability_evidence(dur_registry, identity),
    )
    assert not evaluation.admitted


def test_wrong_backing_provider_on_runtime_fails_closed() -> None:
    store = DocumentStoreUserProfileStore(InMemoryDocumentStore())
    classification = classify_user_profile_store_provider(store)
    identity = builtin_user_profile_store_identity(
        BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        backing_provider_id="cosmos",
    )
    qual_registry, dur_registry = _mongo_evidence_bundle()
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        lookup_trusted_user_profile_qualification_evidence(qual_registry, identity),
        lookup_trusted_user_profile_durability_evidence(dur_registry, identity),
    )
    assert not evaluation.admitted


@pytest.mark.asyncio
async def test_qualification_descriptor_preserves_backing_into_evidence() -> None:
    class _StaticFactory(MemoryProviderInstanceFactory[DocumentStoreUserProfileStore]):
        async def create(self) -> DocumentStoreUserProfileStore:
            return DocumentStoreUserProfileStore(InMemoryDocumentStore())

        async def dispose(self, instance: DocumentStoreUserProfileStore) -> None:
            return None

    def _create() -> DocumentStoreUserProfileStore:
        return DocumentStoreUserProfileStore(InMemoryDocumentStore())

    async def _dispose(_store: DocumentStoreUserProfileStore) -> None:
        return None

    evidence = await run_durable_user_profile_production_qualification(
        descriptor=MemoryProviderDescriptor(
            provider_id=BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            backing_provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
        ),
        context=_context("5cr-backing"),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_StaticFactory()),
        create_store=_create,
        dispose_store=_dispose,
        durability_mode=DurabilityQualificationMode.ADAPTER_RECREATION_ONLY,
    )
    behavioral = qualification_evidence_from_result(
        evidence.canonical,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
    )
    assert behavioral is not None
    assert behavioral.backing_provider_id == MONGODB_DOCUMENT_STORE_PROVIDER_ID
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=False,
    )
    dur = bundle.admission_evidence.durability_registry.resolve(
        BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
        MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        None,
        MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    )
    assert dur.evidence is not None
    assert dur.evidence.backing_provider_id == MONGODB_DOCUMENT_STORE_PROVIDER_ID


def test_backend_substitution_attack_product_admission_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5cr.attack")
    env.memory_profile = _persistent_memory_profile()
    qual_registry, dur_registry = _mongo_evidence_bundle()
    wiring = MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=DocumentStoreUserProfileStore(InMemoryDocumentStore()),
        organization_profile_store=None,
        user_profile_store_identity=builtin_user_profile_store_identity(
            BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
            backing_provider_id=_IN_MEMORY_DOCUMENT_STORE_BACKING_ID,
        ),
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        validate_memory_platform_wiring_admission(
            env,
            wiring.user_profile_store,
            user_profile_store_identity=wiring.user_profile_store_identity,
            qualification_evidence_registry=qual_registry,
            durability_evidence_registry=dur_registry,
        )
    assert exc_info.value.reason_code in {
        MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING,
        MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING,
        MemoryProviderAdmissionReasonCode.PROVIDER_BACKING_IDENTITY_MISMATCH,
    }


def test_sqlite_wiring_identity_has_no_backing(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5cr.sqlite")
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    wiring = _resolve_baseline_memory_platform_wiring(env, env.integration_profile)
    assert wiring.user_profile_store_identity is not None
    assert wiring.user_profile_store_identity.backing_provider_id is None
