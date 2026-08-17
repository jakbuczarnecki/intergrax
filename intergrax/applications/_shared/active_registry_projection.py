# © Artur Czarnecki. All rights reserved.

"""Resolve active revision-bound registry projection from AP-9/AP-10 stores (AC-3)."""

from __future__ import annotations

from intergrax.agent_distribution.stores import ApplicationEnvironmentServingStore
from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RuntimeRegistryProjectionStore,
)


def resolve_active_registry_projection(
    *,
    application_id: str,
    application_environment_id: str,
    serving_store: ApplicationEnvironmentServingStore,
    projection_store: RuntimeRegistryProjectionStore,
) -> MaterializedRegistryProjection:
    """Load the traffic-serving projection for one application environment."""
    serving = serving_store.get_serving_record(application_id, application_environment_id)
    if serving is None or serving.traffic_serving_revision_id is None:
        raise HarnessHostRegistryAuthorityError(
            "no active traffic-serving runtime revision for "
            f"{application_id!r}/{application_environment_id!r}"
        )

    revision_id = serving.traffic_serving_revision_id
    projection = projection_store.get(revision_id)
    if projection is None:
        raise HarnessHostRegistryAuthorityError(
            f"registry projection missing for active runtime revision {revision_id!r}"
        )

    evidence = projection.evidence
    if evidence.application_id != application_id:
        raise HarnessHostRegistryAuthorityError(
            "registry projection application_id "
            f"{evidence.application_id!r} does not match requested {application_id!r}"
        )
    if evidence.application_environment_id != application_environment_id:
        raise HarnessHostRegistryAuthorityError(
            "registry projection application_environment_id "
            f"{evidence.application_environment_id!r} does not match requested "
            f"{application_environment_id!r}"
        )
    return projection


__all__ = ["resolve_active_registry_projection"]
