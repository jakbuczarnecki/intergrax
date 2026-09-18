# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5A: production memory provider admission."""

from __future__ import annotations

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
from intergrax.memory.contracts.provider_qualification import MemoryProviderQualificationStatus
from intergrax.memory.resolver import MemoryStorePluginResolutionError
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
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


class _QualifiedDurableUserProfileStore(UserProfileStore):
    def __init__(self) -> None:
        self._profiles: dict[tuple[str, str], UserProfile] = {}

    @property
    def memory_provider_id(self) -> str:
        return "test.qualified_durable.user_profile"

    @property
    def memory_provider_durability(self) -> MemoryProviderDurability:
        return MemoryProviderDurability.DURABLE

    @property
    def memory_provider_reference_only(self) -> bool:
        return False

    @property
    def memory_provider_qualification_status(self) -> MemoryProviderQualificationStatus:
        return MemoryProviderQualificationStatus.QUALIFIED

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


class _DurableQualifiedUserProfilePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.durable_qualified_user_profile"

    @classmethod
    def create_user_profile_store(cls, **_kwargs: object) -> UserProfileStore:
        return _QualifiedDurableUserProfileStore()


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


def test_product_sqlite_qualified_store_admitted(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.sqlite")
    env.memory_profile = _persistent_memory_profile()
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    wiring = resolve_memory_platform_wiring(env)
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


def test_product_durable_external_plugin_admitted() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.audit5a.plugin.durable")
    env.memory_profile = _persistent_memory_profile()
    env.memory_profile = env.memory_profile.model_copy(
        update={"user_profile_store_plugin_id": "test.durable_qualified_user_profile"},
    )
    wiring = resolve_memory_platform_wiring(
        env,
        discover_entry_points=False,
        explicit_memory_plugins=(_DurableQualifiedUserProfilePlugin,),
    )
    assert isinstance(wiring.user_profile_store, _QualifiedDurableUserProfileStore)


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
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_NOT_DURABLE
    assert exc_info.value.durability is MemoryProviderDurability.UNKNOWN


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
