# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-B (PT-FC2) MEM-1..MEM-7: organization-store and session-storage admission."""

from __future__ import annotations

import pytest
from intergrax.applications._shared import memory_wiring as memory_wiring_module
from intergrax.applications._shared.memory_provider_admission import (
    MemoryOrgSessionAdmissionError,
    MemoryOrgSessionAdmissionReason,
    validate_organization_and_session_admission,
)
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.organization.stores.in_memory_organization_profile_store import (
    InMemoryOrganizationProfileStore,
)
from tests.unit.integrations.providers.document_store.test_mongodb import _collection_factory

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_Reason = MemoryOrgSessionAdmissionReason


def _org_only_memory() -> MemoryProfile:
    return MemoryProfile(enable_org_memory=True)


def _product(memory: MemoryProfile) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="fc2.mem")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = memory
    return env


def _mongo_profile() -> IntegrationProfile:
    from intergrax.integrations.core.binding import IntegrationBinding
    from intergrax.integrations.providers.document_store.mongodb.manifest import MANIFEST

    factory, _ = _collection_factory()
    profile = IntegrationProfile(document_store=IntegrationBinding.from_manifest(MANIFEST))
    profile.options = {
        **(profile.options or {}),
        "mongodb": {
            "uri": "mongodb://localhost:27017",
            "database": "intergrax_test",
            "collection_name": "user_profiles",
            "collection_factory": factory,
        },
    }
    return profile


def _lab_strict(memory: MemoryProfile) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc2.mem.strict")
    env.integration_profile = IntegrationProfile()
    env.execution_mode = ExecutionMode.STRICT
    env.memory_profile = memory
    return env


def test_product_org_memory_on_mongo_raises_unsupported_org_store() -> None:
    env = _product(_org_only_memory())
    env.integration_profile = _mongo_profile()
    with pytest.raises(MemoryOrgSessionAdmissionError) as refusal:
        resolve_memory_platform_wiring(env)
    assert refusal.value.reason is _Reason.ORG_STORE_UNSUPPORTED_BACKEND


def test_product_org_memory_without_backend_raises_instead_of_silent_skip() -> None:
    env = _product(_org_only_memory())
    with pytest.raises(MemoryOrgSessionAdmissionError) as refusal:
        resolve_memory_platform_wiring(env)
    assert refusal.value.reason is _Reason.ORG_STORE_MISSING


def test_product_durable_user_plugin_with_in_memory_session_storage_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The user-profile-store admission (trusted plugin evidence) is covered by the 5A tests;
    # here it is passed so that the remaining hole, the session storage, is isolated.
    monkeypatch.setattr(memory_wiring_module, "validate_memory_platform_wiring_admission", lambda *a, **k: None)
    env = _product(MemoryProfile(enable_user_memory=True, enable_long_term_memory=True))
    with pytest.raises(MemoryOrgSessionAdmissionError) as refusal:
        resolve_memory_platform_wiring(env)
    assert refusal.value.reason is _Reason.SESSION_STORAGE_NOT_DURABLE


def test_admission_classifies_in_memory_org_store_as_not_durable() -> None:
    env = _product(_org_only_memory())
    with pytest.raises(MemoryOrgSessionAdmissionError) as refusal:
        validate_organization_and_session_admission(
            env,
            session_storage=InMemorySessionStorage(),
            organization_profile_store=InMemoryOrganizationProfileStore(),
        )
    assert refusal.value.reason is _Reason.ORG_STORE_NOT_DURABLE


def test_strict_lab_org_memory_with_in_memory_org_store_raises() -> None:
    env = _lab_strict(_org_only_memory())
    assert env.application_profile is ApplicationProfile.LAB
    with pytest.raises(MemoryOrgSessionAdmissionError) as refusal:
        resolve_memory_platform_wiring(env)
    assert refusal.value.reason is _Reason.ORG_STORE_MISSING
    with pytest.raises(MemoryOrgSessionAdmissionError) as in_memory_refusal:
        validate_organization_and_session_admission(
            env,
            session_storage=InMemorySessionStorage(),
            organization_profile_store=InMemoryOrganizationProfileStore(),
        )
    assert in_memory_refusal.value.reason is _Reason.ORG_STORE_NOT_DURABLE


def test_lab_balanced_keeps_in_memory_org_store() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc2.mem.lab")
    env.integration_profile = _mongo_profile()
    env.memory_profile = MemoryProfile(enable_org_memory=True, enable_user_memory=True)
    wiring = resolve_memory_platform_wiring(env)
    assert isinstance(wiring.organization_profile_store, InMemoryOrganizationProfileStore)
    validate_organization_and_session_admission(
        env,
        session_storage=wiring.session_storage,
        organization_profile_store=wiring.organization_profile_store,
    )


def test_product_without_persistent_memory_flags_keeps_in_memory_baseline() -> None:
    env = _product(MemoryProfile())
    wiring = resolve_memory_platform_wiring(env)
    assert isinstance(wiring.session_storage, InMemorySessionStorage)
    assert wiring.organization_profile_store is None
