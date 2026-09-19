# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5A / 5A-R: production memory provider admission."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.applications._shared.memory_provider_admission import (
    validate_memory_platform_wiring_admission,
)
from intergrax.applications._shared.memory_wiring import (
    MemoryPlatformWiring,
    build_session_manager_from_environment,
    resolve_memory_platform_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
    MemoryProviderDurability,
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
    MemoryProviderQualificationEvidence,
    qualification_evidence_from_result,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidence,
    MemoryProviderDurabilityProofKind,
    MemoryProviderTrustedDurabilityStatus,
)
from intergrax.memory.provider_qualification import (
    InMemoryMemoryProviderDurabilityEvidenceRegistry,
    InMemoryMemoryProviderQualificationEvidenceRegistry,
    MemoryProviderCapabilityFactories,
    MemoryProviderQualificationRunner,
)
from intergrax.memory.provider_qualification.factory import MemoryProviderInstanceFactory
from intergrax.memory.resolver import MemoryStorePluginResolutionError
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.contracts.memory_store_creation_context import (
    UserProfileStoreCreationContext,
)
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin import (
    ExternalInMemoryUserProfileStorePlugin,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _persistent_memory_profile() -> MemoryProfile:
    return MemoryProfile(
        enable_user_memory=True,
        enable_long_term_memory=True,
        enable_entity_graph_memory=True,
    )


class _SelfCertifiedDurableUserProfileStore(UserProfileStore):
    """Spoof: declares QUALIFIED without platform evidence."""

    def __init__(self, provider_id: str = "evil.self.certified") -> None:
        self._provider_id = provider_id
        self._profiles: dict[tuple[str, str], UserProfile] = {}

    @property
    def memory_provider_id(self) -> str:
        return self._provider_id

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


class _DurableSelfCertifiedUserProfilePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.durable_qualified_user_profile"

    @classmethod
    def create_user_profile_store(
        cls,
        context: UserProfileStoreCreationContext,
    ) -> UserProfileStore:
        _ = context
        return _SelfCertifiedDurableUserProfileStore(provider_id=cls.plugin_id())


def _evidence_for_provider(
    provider_id: str,
    *,
    status: MemoryProviderQualificationStatus = MemoryProviderQualificationStatus.QUALIFIED,
    provider_version: str | None = None,
    backing_provider_id: str | None = None,
    backing_provider_version: str | None = None,
) -> InMemoryMemoryProviderQualificationEvidenceRegistry:
    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    registry.register(
        MemoryProviderQualificationEvidence(
            provider_id=provider_id,
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            status=status,
            qualification_run_id="run-test",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test_registry",
            provider_version=provider_version,
            backing_provider_id=backing_provider_id,
            backing_provider_version=backing_provider_version,
        ),
    )
    return registry


def _durability_for_provider(
    provider_id: str,
    *,
    status: MemoryProviderTrustedDurabilityStatus = MemoryProviderTrustedDurabilityStatus.DURABLE,
    provider_version: str | None = None,
    backing_provider_id: str | None = None,
    backing_provider_version: str | None = None,
    run_id: str = "run-test",
) -> InMemoryMemoryProviderDurabilityEvidenceRegistry:
    registry = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    registry.register(
        MemoryProviderDurabilityEvidence(
            provider_id=provider_id,
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            durability_status=status,
            qualification_run_id=run_id,
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test_durability_registry",
            proof_kind=MemoryProviderDurabilityProofKind.RESTART_REOPEN,
            provider_version=provider_version,
            backing_provider_id=backing_provider_id,
            backing_provider_version=backing_provider_version,
        ),
    )
    return registry


def test_gap_4_01_product_postgres_persistent_memory_fails_closed() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.gap401")
    env.memory_profile = _persistent_memory_profile()

    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        resolve_memory_platform_wiring(env)

    err = exc_info.value
    assert err.reason_code is MemoryProviderAdmissionReasonCode.REFERENCE_PROVIDER_NOT_ADMISSIBLE
    assert "user_profile_store" in err.capability.value
    assert err.reference_only is True
    assert "mongodb://" not in str(err)


def test_product_persistent_memory_disabled_allows_in_memory_baseline() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.disabled")
    wiring = resolve_memory_platform_wiring(env)
    assert isinstance(wiring.user_profile_store, InMemoryUserProfileStore)


def test_lab_persistent_memory_allows_in_memory() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.audit5a.lab")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = _persistent_memory_profile()
    wiring = resolve_memory_platform_wiring(env)
    assert isinstance(wiring.user_profile_store, InMemoryUserProfileStore)


def test_product_sqlite_without_trusted_evidence_fails(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.sqlite.no_evidence")
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        resolve_memory_platform_wiring(env)
    assert (
        exc_info.value.reason_code
        is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING
    )


def test_product_sqlite_with_trusted_evidence_passes(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.sqlite.evidence")
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    registry = _evidence_for_provider("sqlite.user_profile")
    durability_registry = _durability_for_provider("sqlite.user_profile")
    wiring = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=registry,
        durability_evidence_registry=durability_registry,
    )
    assert isinstance(wiring.user_profile_store, SQLiteUserProfileStore)


def test_product_external_reference_plugin_rejected() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.plugin.ref")
    env.memory_profile = _persistent_memory_profile()
    env.memory_profile = env.memory_profile.model_copy(
        update={"user_profile_store_plugin_id": "external.in_memory_user_profile"},
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        resolve_memory_platform_wiring(
            env,
            discover_entry_points=False,
            explicit_memory_plugins=(ExternalInMemoryUserProfileStorePlugin,),
        )
    assert (
        exc_info.value.reason_code
        is MemoryProviderAdmissionReasonCode.REFERENCE_PROVIDER_NOT_ADMISSIBLE
    )


def test_product_durable_external_plugin_rejected_without_evidence() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.plugin.durable")
    env.memory_profile = _persistent_memory_profile()
    env.memory_profile = env.memory_profile.model_copy(
        update={"user_profile_store_plugin_id": "test.durable_qualified_user_profile"},
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        resolve_memory_platform_wiring(
            env,
            discover_entry_points=False,
            explicit_memory_plugins=(_DurableSelfCertifiedUserProfilePlugin,),
        )
    assert (
        exc_info.value.reason_code
        is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING
    )


def test_product_durable_external_plugin_passes_with_trusted_evidence() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.plugin.evidence")
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
    assert isinstance(wiring.user_profile_store, _SelfCertifiedDurableUserProfileStore)


def test_direct_custom_wiring_cannot_bypass_admission() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.bypass")
    env.memory_profile = _persistent_memory_profile()
    wiring = MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        organization_profile_store=None,
    )
    with pytest.raises(MemoryProviderAdmissionError):
        build_session_manager_from_environment(env, memory_wiring=wiring, tenant_id="tenant-a")


def test_unknown_durability_metadata_fails_closed_in_product() -> None:
    class _OpaqueUserProfileStore(UserProfileStore):
        async def get_profile(self, *, tenant_id: str, user_id: str) -> UserProfile:
            raise NotImplementedError

        async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
            raise NotImplementedError

        async def delete_profile(self, *, tenant_id: str, user_id: str) -> None:
            raise NotImplementedError

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.unknown")
    env.memory_profile = _persistent_memory_profile()
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        validate_memory_platform_wiring_admission(env, _OpaqueUserProfileStore())
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_IDENTITY_MISSING


def test_strict_lab_not_product_still_allows_reference() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.audit5a.strict.lab")
    env.integration_profile = IntegrationProfile()
    env.execution_mode = ExecutionMode.STRICT
    env.memory_profile = _persistent_memory_profile()
    wiring = resolve_memory_platform_wiring(env)
    assert isinstance(wiring.user_profile_store, InMemoryUserProfileStore)
    assert env.execution_mode is ExecutionMode.STRICT


def test_invalid_plugin_still_resolution_error() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.bad.plugin")
    env.memory_profile = _persistent_memory_profile()
    env.memory_profile = env.memory_profile.model_copy(
        update={"user_profile_store_plugin_id": "missing.plugin"},
    )
    with pytest.raises(MemoryStorePluginResolutionError):
        resolve_memory_platform_wiring(env, discover_entry_points=False)


def test_provider_spoof_self_certified_fails_without_registry() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar.spoof")
    env.memory_profile = _persistent_memory_profile()
    store = _SelfCertifiedDurableUserProfileStore()
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        validate_memory_platform_wiring_admission(env, store)
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_IDENTITY_MISSING


def test_wrong_provider_evidence_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar.wrong.provider")
    env.memory_profile = _persistent_memory_profile()
    registry = _evidence_for_provider("other.provider")
    identity = MemoryProviderIdentity(
        provider_id="evil.self.certified",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        validate_memory_platform_wiring_admission(
            env,
            _SelfCertifiedDurableUserProfileStore(),
            user_profile_store_identity=identity,
            qualification_evidence_registry=registry,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING


def test_not_qualified_evidence_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar.not.qualified")
    env.memory_profile = _persistent_memory_profile()
    registry = _evidence_for_provider(
        "evil.self.certified",
        status=MemoryProviderQualificationStatus.NOT_QUALIFIED,
    )
    identity = MemoryProviderIdentity(
        provider_id="evil.self.certified",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        source=MemoryProviderIdentitySource.DIRECT_INJECTION,
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        validate_memory_platform_wiring_admission(
            env,
            _SelfCertifiedDurableUserProfileStore(),
            user_profile_store_identity=identity,
            qualification_evidence_registry=registry,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_NOT_QUALIFIED


def test_admission_policy_source_does_not_trust_declared_qualification_field() -> None:
    from intergrax.applications._shared import memory_provider_admission as admission_module

    source = inspect.getsource(admission_module)
    assert "memory_provider_qualification_status" not in source


@pytest.mark.asyncio
async def test_runner_result_to_registry_to_admission_passes(tmp_path: Path) -> None:
    tmp = str(tmp_path)
    db_path = str(Path(tmp) / "qual.db")

    class _Factory(MemoryProviderInstanceFactory[SQLiteUserProfileStore]):
        async def create(self) -> SQLiteUserProfileStore:
            return SQLiteUserProfileStore(db_path)

        async def dispose(self, instance: SQLiteUserProfileStore) -> None:
            instance.close()

    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="sqlite.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=MemoryProviderQualificationContext(
            qualification_run_id="audit5ar-sqlite",
            tenant_qualification_id="t",
            user_qualification_id="u",
            workspace_qualification_id="w",
            reference_time_iso="2025-01-01T00:00:00+00:00",
        ),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_Factory()),
    )
    evidence = qualification_evidence_from_result(
        result,
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
    )
    assert evidence is not None
    assert evidence.status is MemoryProviderQualificationStatus.QUALIFIED

    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    registry.register(evidence)
    durability_registry = _durability_for_provider(
        "sqlite.user_profile",
        run_id="audit5ar-sqlite",
    )

    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5ar.runner")
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": tmp},
    }
    wiring = resolve_memory_platform_wiring(
        env,
        qualification_evidence_registry=registry,
        durability_evidence_registry=durability_registry,
    )
    assert isinstance(wiring.user_profile_store, SQLiteUserProfileStore)
    wiring.user_profile_store.close()


def test_product_sqlite_with_behavioral_only_fails_without_durability_evidence(
    tmp_path: Path,
) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(
        profile_id="mem.audit5ar3.sqlite.behavioral_only",
    )
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    registry = _evidence_for_provider("sqlite.user_profile")
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        resolve_memory_platform_wiring(
            env,
            qualification_evidence_registry=registry,
        )
    assert (
        exc_info.value.reason_code
        is MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING
    )
