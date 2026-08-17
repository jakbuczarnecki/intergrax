# © Artur Czarnecki. All rights reserved.

"""AC-3 active registry projection resolver proofs."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentServingStore,
)
from intergrax.applications._shared.active_registry_projection import resolve_active_registry_projection
from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from dataclasses import replace

from intergrax.applications._shared.registry_projection import (
    InMemoryRuntimeRegistryProjectionStore,
    MaterializedRegistryProjection,
)
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection
from tests.unit.applications.test_registry_projection_ap10 import _APP, _ENV, _manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _projection_for_host() -> object:
    manifest = _manifest()
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    return build_test_registry_projection(manifest, env, revision_id="rev-active-1")


def test_resolve_active_projection_without_serving_record_fails_closed() -> None:
    state = AgentDistributionStoreState()
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    with pytest.raises(HarnessHostRegistryAuthorityError, match="no active traffic-serving"):
        resolve_active_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            serving_store=serving_store,
            projection_store=projection_store,
        )


def test_resolve_active_projection_missing_projection_fails_closed() -> None:
    state = AgentDistributionStoreState()
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    serving_store.atomic_swap_serving_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id="rev-missing",
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )
    with pytest.raises(HarnessHostRegistryAuthorityError, match="registry projection missing"):
        resolve_active_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            serving_store=serving_store,
            projection_store=projection_store,
        )


def test_resolve_active_projection_wrong_application_fails_closed() -> None:
    state = AgentDistributionStoreState()
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    projection = _projection_for_host()
    mismatched = replace(
        projection,
        evidence=projection.evidence.model_copy(update={"application_id": "other-app"}),
    )
    projection_store.put(mismatched)
    serving_store.atomic_swap_serving_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=mismatched.evidence.runtime_revision_id,
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )
    with pytest.raises(HarnessHostRegistryAuthorityError, match="application_id"):
        resolve_active_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            serving_store=serving_store,
            projection_store=projection_store,
        )


def test_resolve_active_projection_wrong_environment_fails_closed() -> None:
    state = AgentDistributionStoreState()
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    projection = _projection_for_host()
    mismatched = replace(
        projection,
        evidence=projection.evidence.model_copy(update={"application_environment_id": "env-other"}),
    )
    projection_store.put(mismatched)
    serving_store.atomic_swap_serving_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=mismatched.evidence.runtime_revision_id,
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )
    with pytest.raises(HarnessHostRegistryAuthorityError, match="application_environment_id"):
        resolve_active_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            serving_store=serving_store,
            projection_store=projection_store,
        )


def test_resolve_active_projection_success() -> None:
    state = AgentDistributionStoreState()
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    projection = _projection_for_host()
    projection_store.put(projection)
    serving_store.atomic_swap_serving_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=projection.evidence.runtime_revision_id,
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )
    resolved = resolve_active_registry_projection(
        application_id=_APP,
        application_environment_id=_ENV,
        serving_store=serving_store,
        projection_store=projection_store,
    )
    assert resolved.evidence.runtime_revision_id == "rev-active-1"
    assert resolved.agent_registry.list_agent_ids() == projection.agent_registry.list_agent_ids()
