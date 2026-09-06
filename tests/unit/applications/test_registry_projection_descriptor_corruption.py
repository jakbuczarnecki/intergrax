# © Artur Czarnecki. All rights reserved.

"""Fail-closed registry projection descriptor corruption tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.applications._shared.registry_projection import (
    InMemoryRuntimeRegistryProjectionStore,
)
from intergrax.applications._shared.registry_projection_rehydrator import (
    RegistryProjectionRehydrationError,
    RuntimeRegistryProjectionRehydrator,
)
from testing_support.canonical_agent_lifecycle_composition import (
    default_stage15_proof_config,
)
from testing_support.enterprise_agent_lifecycle_composition import (
    EnterpriseAgentLifecycleProofStack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _corrupt_descriptor_json(
    database,
    *,
    runtime_revision_id: str,
    payload: dict[str, object],
) -> None:
    connection_factory = getattr(database, "_connect")
    with connection_factory() as conn:
        conn.execute(
            "UPDATE projection_descriptors SET payload_json = ? WHERE runtime_revision_id = ?",
            (json.dumps(payload), runtime_revision_id),
        )
        conn.commit()


def test_descriptor_corruption_fail_closed(tmp_path: Path) -> None:
    config = default_stage15_proof_config()
    db_path = tmp_path / "corruption.db"
    shared_root = tmp_path / "shared-artifacts"
    stack = EnterpriseAgentLifecycleProofStack.build(shared_root, db_path=db_path)
    result = stack.run_happy_path()
    descriptor_store = stack.durable_runtime.projection_descriptor_store
    descriptor = descriptor_store.get_for_revision(
        config.application_id,
        config.environment_id,
        result.runtime_revision_id,
    )
    assert descriptor is not None
    rehydrator = RuntimeRegistryProjectionRehydrator(
        serving_store=stack.durable_runtime.distribution_store_bundle.serving_store,
        descriptor_store=descriptor_store,
        authority=stack.durable_runtime.registry_projection_authority,
        projection_store=InMemoryRuntimeRegistryProjectionStore(),
    )
    database = stack.durable_runtime.distribution_store_bundle.database

    wrong_digest = descriptor.model_dump()
    wrong_digest["materialization_artifact_digest"] = "sha256:" + ("f" * 64)
    _corrupt_descriptor_json(
        database,
        runtime_revision_id=result.runtime_revision_id,
        payload=wrong_digest,
    )
    with pytest.raises(RegistryProjectionRehydrationError):
        rehydrator.rehydrate_serving_registry_projection(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        )

    wrong_roster = descriptor.model_dump()
    wrong_roster["effective_roster_revision_id"] = "roster-corrupt"
    _corrupt_descriptor_json(
        database,
        runtime_revision_id=result.runtime_revision_id,
        payload=wrong_roster,
    )
    with pytest.raises(RegistryProjectionRehydrationError):
        rehydrator.rehydrate_serving_registry_projection(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        )

    wrong_revision = descriptor.model_dump()
    wrong_revision["runtime_revision_id"] = "rev-corrupt"
    _corrupt_descriptor_json(
        database,
        runtime_revision_id=result.runtime_revision_id,
        payload=wrong_revision,
    )
    with pytest.raises(RegistryProjectionRehydrationError):
        rehydrator.rehydrate_serving_registry_projection(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        )

    unsupported = descriptor.model_dump()
    unsupported["schema_version"] = "unsupported.v99"
    _corrupt_descriptor_json(
        database,
        runtime_revision_id=result.runtime_revision_id,
        payload=unsupported,
    )
    with pytest.raises((RegistryProjectionRehydrationError, ValueError)):
        rehydrator.rehydrate_serving_registry_projection(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        )

    connection_factory = getattr(database, "_connect")
    with connection_factory() as conn:
        conn.execute(
            "DELETE FROM projection_descriptors WHERE runtime_revision_id = ?",
            (result.runtime_revision_id,),
        )
        conn.commit()
    with pytest.raises(RegistryProjectionRehydrationError):
        rehydrator.rehydrate_serving_registry_projection(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        )
