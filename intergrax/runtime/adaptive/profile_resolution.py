# © Artur Czarnecki. All rights reserved.

"""Resolve active and candidate profile versions for runtime requests (Phase W-ADAPT-4.2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.host_profile_slices import AdaptiveProfile
from intergrax.runtime.adaptive.canary_traffic import should_route_canary_traffic
from intergrax.runtime.adaptive.contracts import ProfileArtifactType, ProfileVersionRecord, ProfileVersionStatus
from intergrax.runtime.adaptive.profile_pointer_store import ProfileActivePointerStore
from intergrax.runtime.adaptive.profile_version_store import ProfileVersionStore


class ResolvedProfileVersions(BaseModel):
    """Active and candidate profile versions selected for one request."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    use_candidate_for_traffic: bool = False
    active: dict[str, ProfileVersionRecord] = Field(default_factory=dict)
    candidate: dict[str, ProfileVersionRecord] = Field(default_factory=dict)


def resolve_profile_versions_for_request(
    *,
    tenant_id: str,
    task_class: str,
    routing_key: str,
    adaptive_profile: AdaptiveProfile,
    profile_store: ProfileVersionStore,
    pointer_store: ProfileActivePointerStore,
) -> ResolvedProfileVersions:
    """Resolve active pointers and latest shadow/canary candidates for a request."""
    active: dict[str, ProfileVersionRecord] = {}
    candidate: dict[str, ProfileVersionRecord] = {}

    for artifact_type in ProfileArtifactType:
        pointer = pointer_store.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        if pointer is not None:
            record = profile_store.get(pointer.active_version_id)
            if record is not None and record.status == ProfileVersionStatus.ACTIVE:
                active[artifact_type.value] = record

        versions = profile_store.list_versions(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            limit=20,
        )
        for record in reversed(versions):
            if record.status in {ProfileVersionStatus.SHADOW, ProfileVersionStatus.CANARY}:
                candidate[artifact_type.value] = record
                break

    use_candidate = False
    if adaptive_profile.enabled and adaptive_profile.mode in {"canary", "apply"}:
        use_candidate = should_route_canary_traffic(
            tenant_id=tenant_id,
            routing_key=routing_key,
            canary_tenant_allowlist=adaptive_profile.canary_tenant_allowlist,
            canary_traffic_percent=adaptive_profile.canary_traffic_percent,
        )

    return ResolvedProfileVersions(
        tenant_id=tenant_id,
        task_class=task_class,
        use_candidate_for_traffic=use_candidate,
        active=active,
        candidate=candidate,
    )
