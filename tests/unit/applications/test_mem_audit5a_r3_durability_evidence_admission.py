# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5A-R3: trusted durability evidence admission."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.applications._shared.memory_provider_admission import (
    validate_memory_platform_wiring_admission,
)
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
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
    MemoryProviderDurabilityEvidence,
    MemoryProviderDurabilityEvidenceResolveStatus,
    MemoryProviderDurabilityProofKind,
    MemoryProviderTrustedDurabilityStatus,
    durability_evidence_from_reopen_proof,
)
from intergrax.memory.contracts.provider_identity import (
    MemoryProviderIdentity,
    MemoryProviderIdentitySource,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderDescriptor,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationRequest,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    qualification_evidence_from_result,
)
from intergrax.memory.provider_qualification import (
    InMemoryMemoryProviderDurabilityEvidenceRegistry,
    InMemoryMemoryProviderQualificationEvidenceRegistry,
    MemoryProviderCapabilityFactories,
    MemoryProviderInstanceFactory,
    MemoryProviderQualificationRunner,
)
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.user_profile_store import UserProfileStore
from tests.unit.applications.test_mem_audit5a_production_provider_admission import (
    _SelfCertifiedDurableUserProfileStore,
    _durability_for_provider,
    _evidence_for_provider,
    _persistent_memory_profile,
)
from tests.unit.memory.durable_provider_qualification_harness import (
    run_durable_user_profile_production_qualification,
    user_profile_qualification_request,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


async def _dispose_sqlite(store: SQLiteUserProfileStore) -> None:
    store.close()


class _FakeEphemeralRamStore(UserProfileStore):
    """Attack store: RAM-only but claims DURABLE production provider."""

    def __init__(self) -> None:
        self._profiles: dict[tuple[str, str], UserProfile] = {}

    @property
    def memory_provider_id(self) -> str:
        return "fake.ephemeral"

    @property
    def memory_provider_durability(self) -> MemoryProviderDurability:
        return MemoryProviderDurability.DURABLE

    @property
    def memory_provider_reference_only(self) -> bool:
        return False

    @property
    def memory_provider_declared_qualification_status(self) -> MemoryProviderQualificationStatus:
        return MemoryProviderQualificationStatus.QUALIFIED

    @property
    def memory_provider_version(self) -> str | None:
        return None

    async def get_profile(self, *, tenant_id: str, user_id: str) -> UserProfile:
        key = (tenant_id, user_id)
        if key in self._profiles:
            return self._profiles[key]
        profile = UserProfile(
            identity=UserIdentity(user_id=user_id),
            preferences=UserPreferences(),
        )
        self._profiles[key] = profile
        return profile

    async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
        self._profiles[(tenant_id, profile.identity.user_id)] = profile

    async def delete_profile(self, *, tenant_id: str, user_id: str) -> None:
        self._profiles.pop((tenant_id, user_id), None)


class _FakeEphemeralFactory(MemoryProviderInstanceFactory[_FakeEphemeralRamStore]):
    async def create(self) -> _FakeEphemeralRamStore:
        return _FakeEphemeralRamStore()

    async def dispose(self, instance: _FakeEphemeralRamStore) -> None:
        _ = instance


@pytest.mark.asyncio
async def test_fake_ram_behavioral_qualified_without_durability_evidence_fails_product() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="fake.ephemeral",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=MemoryProviderQualificationContext(
            qualification_run_id="fake-ephemeral-behavioral",
            tenant_qualification_id="t",
            user_qualification_id="u",
            workspace_qualification_id="w",
            reference_time_iso="2025-01-01T00:00:00+00:00",
        ),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_FakeEphemeralFactory()),
    )
    behavioral = qualification_evidence_from_result(
        result,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
    )
    assert behavioral is not None
    assert behavioral.status is MemoryProviderQualificationStatus.QUALIFIED

    qual_registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    qual_registry.register(behavioral)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar3.fake.ram")
    env.memory_profile = _persistent_memory_profile()
    identity = MemoryProviderIdentity(
        provider_id="fake.ephemeral",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        validate_memory_platform_wiring_admission(
            env,
            _FakeEphemeralRamStore(),
            user_profile_store_identity=identity,
            qualification_evidence_registry=qual_registry,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING
    assert exc_info.value.trusted_qualification_status is MemoryProviderQualificationStatus.QUALIFIED


def test_declared_durable_alone_cannot_pass_admission() -> None:
    from intergrax.memory.contracts import provider_admission as admission_contract

    source = inspect.getsource(admission_contract.evaluate_production_persistent_user_profile_admission)
    assert "classification.durability" not in source


def test_not_durable_trusted_evidence_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar3.not.durable")
    env.memory_profile = _persistent_memory_profile()
    identity = MemoryProviderIdentity(
        provider_id="fake.ephemeral",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    qual_registry = _evidence_for_provider("fake.ephemeral")
    durability_registry = _durability_for_provider(
        "fake.ephemeral",
        status=MemoryProviderTrustedDurabilityStatus.NOT_DURABLE,
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        validate_memory_platform_wiring_admission(
            env,
            _FakeEphemeralRamStore(),
            user_profile_store_identity=identity,
            qualification_evidence_registry=qual_registry,
            durability_evidence_registry=durability_registry,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_NOT_DURABLE


def test_synthetic_durable_evidence_contract_only_passes_with_behavioral() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar3.synthetic")
    env.memory_profile = _persistent_memory_profile()
    identity = MemoryProviderIdentity(
        provider_id="fake.ephemeral",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    validate_memory_platform_wiring_admission(
        env,
        _FakeEphemeralRamStore(),
        user_profile_store_identity=identity,
        qualification_evidence_registry=_evidence_for_provider("fake.ephemeral"),
        durability_evidence_registry=_durability_for_provider("fake.ephemeral"),
    )


@pytest.mark.asyncio
async def test_sqlite_durable_harness_maps_to_trusted_durability_evidence(tmp_path: Path) -> None:
    db_path = str(tmp_path / "dur.db")

    class _SqliteFactory(MemoryProviderInstanceFactory[SQLiteUserProfileStore]):
        async def create(self) -> SQLiteUserProfileStore:
            return SQLiteUserProfileStore(db_path)

        async def dispose(self, instance: SQLiteUserProfileStore) -> None:
            instance.close()

    durable = await run_durable_user_profile_production_qualification(
        descriptor=MemoryProviderDescriptor(
            provider_id="sqlite.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=MemoryProviderQualificationContext(
            qualification_run_id="mem-ent-13c-admission",
            tenant_qualification_id="t",
            user_qualification_id="u",
            workspace_qualification_id="w",
            reference_time_iso="2025-01-01T00:00:00+00:00",
        ),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_SqliteFactory()),
        create_store=lambda: SQLiteUserProfileStore(db_path),
        dispose_store=lambda store: _dispose_sqlite(store),
    )
    assert durable.production_durable_qualified

    durability_evidence = durability_evidence_from_reopen_proof(
        provider_id=durable.canonical.descriptor.provider_id,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        qualification_run_id=durable.canonical.qualification_run_id,
        reference_time_iso=durable.canonical.reference_time_iso,
        reopen_passed=bool(durable.reopen_passed),
        delete_reopen_passed=durable.delete_reopen_passed,
        provider_version=durable.canonical.descriptor.provider_version,
        evidence_source="mem_ent13c_durable_harness",
    )
    assert (
        durability_evidence.durability_status is MemoryProviderTrustedDurabilityStatus.DURABLE
    )

    behavioral = qualification_evidence_from_result(
        durable.canonical,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
    )
    assert behavioral is not None

    qual_registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    qual_registry.register(behavioral)
    dur_registry = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    dur_registry.register(durability_evidence)

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar3.sqlite.harness")
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    wiring = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=qual_registry,
        durability_evidence_registry=dur_registry,
    )
    assert isinstance(wiring.user_profile_store, SQLiteUserProfileStore)
    wiring.user_profile_store.close()


def test_wrong_provider_durability_evidence_fails() -> None:
    store = _SelfCertifiedDurableUserProfileStore(provider_id="provider.a")
    classification = classify_user_profile_store_provider(store)
    identity = MemoryProviderIdentity(
        provider_id="provider.a",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    qual_registry = _evidence_for_provider("provider.a")
    durability_registry = _durability_for_provider("provider.b")
    qual_lookup = lookup_trusted_user_profile_qualification_evidence(qual_registry, identity)
    dur_lookup = lookup_trusted_user_profile_durability_evidence(durability_registry, identity)
    assert dur_lookup.resolve_status is MemoryProviderDurabilityEvidenceResolveStatus.MISSING
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        qual_lookup,
        dur_lookup,
    )
    assert not evaluation.admitted
    assert evaluation.reason_code is MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING


def test_duplicate_durability_evidence_is_ambiguous() -> None:
    store = _SelfCertifiedDurableUserProfileStore(provider_id="dup.dur")
    classification = classify_user_profile_store_provider(store)
    identity = MemoryProviderIdentity(
        provider_id="dup.dur",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    qual_registry = _evidence_for_provider("dup.dur")
    dur_registry = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    for run_id in ("dur-a", "dur-b"):
        dur_registry.register(
            MemoryProviderDurabilityEvidence(
                provider_id="dup.dur",
                capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
                durability_status=MemoryProviderTrustedDurabilityStatus.DURABLE,
                qualification_run_id=run_id,
                reference_time_iso="2025-01-01T00:00:00+00:00",
                evidence_source="test",
                proof_kind=MemoryProviderDurabilityProofKind.RESTART_REOPEN,
            ),
        )
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        lookup_trusted_user_profile_qualification_evidence(qual_registry, identity),
        lookup_trusted_user_profile_durability_evidence(dur_registry, identity),
    )
    assert not evaluation.admitted
    assert evaluation.reason_code is MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISMATCH
