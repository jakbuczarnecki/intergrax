# © Artur Czarnecki. All rights reserved.

"""Orchestration profile overrides from adaptive profile versions (Phase W-ADAPT-4.10)."""

from __future__ import annotations

from intergrax.contracts.host_profile_slices import OrchestrationProfile
from intergrax.runtime.adaptive.contracts import ProfileArtifactType, ProfileVersionRecord


def apply_orchestration_profile_version(
    base_profile: OrchestrationProfile,
    version: ProfileVersionRecord | None,
) -> OrchestrationProfile:
    """Merge active/candidate orchestration profile version payload into base profile."""
    if version is None or version.artifact_type != ProfileArtifactType.ORCHESTRATION:
        return base_profile

    payload = version.artifact_payload
    updates: dict[str, object] = {}

    max_parallel_nodes = payload.get("max_parallel_nodes")
    if isinstance(max_parallel_nodes, int) and max_parallel_nodes >= 1:
        updates["max_parallel_nodes"] = max_parallel_nodes

    retry_policy_name = payload.get("retry_policy_name")
    if isinstance(retry_policy_name, str) and retry_policy_name.strip():
        updates["retry_policy_name"] = retry_policy_name.strip()

    long_running_enabled = payload.get("long_running_enabled")
    if isinstance(long_running_enabled, bool):
        updates["long_running_enabled"] = long_running_enabled

    if not updates:
        return base_profile
    return base_profile.model_copy(update=updates)
