# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5A-R2: trusted provider identity binding."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.memory_provider_admission import (
    validate_memory_platform_wiring_admission,
)
from intergrax.applications._shared.memory_wiring import (
    MemoryPlatformWiring,
    resolve_memory_platform_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
    MemoryProviderDurability,
    evaluate_production_persistent_user_profile_admission,
    classify_user_profile_store_provider,
)
from intergrax.memory.contracts.provider_identity import (
    BUILTIN_SQLITE_USER_PROFILE_ID,
    MemoryProviderIdentity,
    MemoryProviderIdentitySource,
    builtin_user_profile_store_identity,
    plugin_user_profile_store_identity,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidence,
    MemoryProviderDurabilityEvidenceLookup,
    MemoryProviderDurabilityEvidenceResolveStatus,
    MemoryProviderDurabilityProofKind,
    MemoryProviderTrustedDurabilityStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidence,
    MemoryProviderQualificationEvidenceLookup,
    MemoryProviderQualificationEvidenceResolveStatus,
)
from intergrax.memory.provider_qualification import InMemoryMemoryProviderDurabilityEvidenceRegistry
from intergrax.memory.provider_qualification import InMemoryMemoryProviderQualificationEvidenceRegistry
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.contracts.memory_store_creation_context import (
    UserProfileStoreCreationContext,
)
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from tests.unit.applications.test_mem_audit5a_production_provider_admission import (
    _DurableSelfCertifiedUserProfilePlugin,
    _SelfCertifiedDurableUserProfileStore,
    _durability_for_provider,
    _evidence_for_provider,
    _persistent_memory_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _EvilSqliteImpersonatorStore(UserProfileStore):
    """External plugin store claiming a built-in provider id."""

    def __init__(self) -> None:
        self._profiles: dict[tuple[str, str], UserProfile] = {}

    @property
    def memory_provider_id(self) -> str:
        return BUILTIN_SQLITE_USER_PROFILE_ID

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


class _EvilSqliteImpersonatorPlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "evil.plugin"

    @classmethod
    def create_user_profile_store(
        cls,
        context: UserProfileStoreCreationContext,
    ) -> UserProfileStore:
        _ = context
        return _EvilSqliteImpersonatorStore()


def test_gap_5a_02_evil_plugin_cannot_use_sqlite_trusted_evidence() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar2.evil")
    env.memory_profile = _persistent_memory_profile()
    env.memory_profile = env.memory_profile.model_copy(
        update={"user_profile_store_plugin_id": "evil.plugin"},
    )
    registry = _evidence_for_provider(BUILTIN_SQLITE_USER_PROFILE_ID)
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        resolve_memory_platform_wiring(
            env,
            discover_entry_points=False,
            explicit_memory_plugins=(_EvilSqliteImpersonatorPlugin,),
            qualification_evidence_registry=registry,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_IDENTITY_MISMATCH
    assert exc_info.value.trusted_provider_id == "evil.plugin"
    assert exc_info.value.declared_provider_id == BUILTIN_SQLITE_USER_PROFILE_ID


def test_legitimate_external_plugin_passes_with_plugin_id_evidence() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar2.legit")
    env.memory_profile = _persistent_memory_profile()
    env.memory_profile = env.memory_profile.model_copy(
        update={"user_profile_store_plugin_id": "test.durable_qualified_user_profile"},
    )
    registry = _evidence_for_provider("test.durable_qualified_user_profile")
    durability_registry = _durability_for_provider("test.durable_qualified_user_profile")
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(_DurableSelfCertifiedUserProfilePlugin,),
        qualification_evidence_registry=registry,
        durability_evidence_registry=durability_registry,
    )
    assert wiring.user_profile_store_identity is not None
    assert wiring.user_profile_store_identity.provider_id == "test.durable_qualified_user_profile"


def test_product_persistent_direct_wiring_without_identity_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar2.no.identity")
    env.memory_profile = _persistent_memory_profile()
    store = _SelfCertifiedDurableUserProfileStore()
    registry = _evidence_for_provider("evil.self.certified")
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        validate_memory_platform_wiring_admission(
            env,
            store,
            qualification_evidence_registry=registry,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_IDENTITY_MISSING


def test_direct_wiring_with_trusted_identity_and_evidence_passes() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar2.direct.ok")
    env.memory_profile = _persistent_memory_profile()
    store = _SelfCertifiedDurableUserProfileStore(provider_id="custom.direct")
    identity = MemoryProviderIdentity(
        provider_id="custom.direct",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    registry = _evidence_for_provider("custom.direct")
    validate_memory_platform_wiring_admission(
        env,
        store,
        user_profile_store_identity=identity,
        qualification_evidence_registry=registry,
        durability_evidence_registry=_durability_for_provider("custom.direct"),
    )


@pytest.mark.parametrize(
    ("identity_version", "evidence_version", "expected_admitted"),
    [
        (None, None, True),
        ("1", "1", True),
        ("1", "2", False),
        (None, "1", False),
        ("1", None, False),
    ],
)
def test_version_binding_semantics(
    identity_version: str | None,
    evidence_version: str | None,
    expected_admitted: bool,
) -> None:
    store = _SelfCertifiedDurableUserProfileStore(provider_id="versioned.provider")
    classification = classify_user_profile_store_provider(store)
    identity = MemoryProviderIdentity(
        provider_id="versioned.provider",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
        provider_version=identity_version,
    )
    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    registry.register(
        MemoryProviderQualificationEvidence(
            provider_id="versioned.provider",
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id="run-version",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test",
            provider_version=evidence_version,
        ),
    )
    lookup = registry.resolve(
        identity.provider_id,
        identity.capability,
        identity.provider_version,
    )
    durability_registry = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    durability_registry.register(
        MemoryProviderDurabilityEvidence(
            provider_id="versioned.provider",
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            durability_status=MemoryProviderTrustedDurabilityStatus.DURABLE,
            qualification_run_id="run-version" if expected_admitted else "run-dur-version",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test",
            proof_kind=MemoryProviderDurabilityProofKind.RESTART_REOPEN,
            provider_version=evidence_version,
        ),
    )
    durability_lookup = durability_registry.resolve(
        identity.provider_id,
        identity.capability,
        identity.provider_version,
    )
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        lookup,
        durability_lookup,
    )
    assert evaluation.admitted is expected_admitted


def test_duplicate_qualification_evidence_is_ambiguous_fail_closed() -> None:
    store = _SelfCertifiedDurableUserProfileStore(provider_id="dup.provider")
    classification = classify_user_profile_store_provider(store)
    identity = MemoryProviderIdentity(
        provider_id="dup.provider",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    for run_id in ("run-a", "run-b"):
        registry.register(
            MemoryProviderQualificationEvidence(
                provider_id=identity.provider_id,
                capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
                status=MemoryProviderQualificationStatus.QUALIFIED,
                qualification_run_id=run_id,
                reference_time_iso="2025-01-01T00:00:00+00:00",
                evidence_source="test",
            ),
        )
    lookup = registry.resolve(identity.provider_id, identity.capability, None)
    assert lookup.resolve_status is MemoryProviderQualificationEvidenceResolveStatus.AMBIGUOUS
    missing_durability = MemoryProviderDurabilityEvidenceLookup(
        resolve_status=MemoryProviderDurabilityEvidenceResolveStatus.MISSING,
    )
    evaluation = evaluate_production_persistent_user_profile_admission(
        classification,
        identity,
        lookup,
        missing_durability,
    )
    assert not evaluation.admitted
    assert (
        evaluation.reason_code
        is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISMATCH
    )


def test_overlay_replaces_store_and_identity_together(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar2.overlay")
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    env.memory_profile = env.memory_profile.model_copy(
        update={"user_profile_store_plugin_id": "test.durable_qualified_user_profile"},
    )
    registry = _evidence_for_provider("test.durable_qualified_user_profile")
    durability_registry = _durability_for_provider("test.durable_qualified_user_profile")
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(_DurableSelfCertifiedUserProfilePlugin,),
        qualification_evidence_registry=registry,
        durability_evidence_registry=durability_registry,
    )
    assert not isinstance(wiring.user_profile_store, SQLiteUserProfileStore)
    assert wiring.user_profile_store_identity is not None
    assert wiring.user_profile_store_identity.provider_id == "test.durable_qualified_user_profile"
    assert wiring.user_profile_store_identity.source is MemoryProviderIdentitySource.PLUGIN
    assert wiring.user_profile_store_identity.backing_provider_id is None


def test_sqlite_wiring_identity_matches_runner_descriptor(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar2.sqlite.chain")
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    registry = _evidence_for_provider(BUILTIN_SQLITE_USER_PROFILE_ID)
    durability_registry = _durability_for_provider(BUILTIN_SQLITE_USER_PROFILE_ID)
    wiring = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=registry,
        durability_evidence_registry=durability_registry,
    )
    assert wiring.user_profile_store_identity == builtin_user_profile_store_identity(
        BUILTIN_SQLITE_USER_PROFILE_ID,
    )
    wiring.user_profile_store.close()
