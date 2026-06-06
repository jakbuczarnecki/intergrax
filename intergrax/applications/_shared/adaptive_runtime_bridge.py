# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile adaptive fields to RuntimeConfig (Phase W-ADAPT-4.2, 4.10)."""

from __future__ import annotations

from intergrax.applications._shared.adaptive_wiring import ApplicationAdaptiveWiring
from intergrax.applications.contracts.environment_profile import (
    AdaptiveProfile,
    ApplicationEnvironmentProfile,
)
from intergrax.runtime.adaptive.contracts import ProfileArtifactType, ProfileVersionRecord
from intergrax.runtime.adaptive.profile_orchestration_resolver import apply_orchestration_profile_version
from intergrax.runtime.adaptive.profile_policy_resolver import apply_policy_fragment_version
from intergrax.runtime.adaptive.profile_rag_router import apply_rag_profile_version
from intergrax.runtime.adaptive.profile_resolution import ResolvedProfileVersions, resolve_profile_versions_for_request
from intergrax.runtime.nexus.config import RuntimeConfig


def apply_adaptive_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: AdaptiveProfile,
    *,
    wiring: ApplicationAdaptiveWiring | None = None,
    tenant_id: str | None = None,
    task_class: str = "",
    routing_key: str = "",
) -> RuntimeConfig:
    """Record adaptive posture and optionally resolve versioned profile overrides."""
    config.adaptive_profile = profile
    if wiring is not None and wiring.signal_collector is not None:
        config.signal_collector = wiring.signal_collector

    if (
        not profile.enabled
        or wiring is None
        or wiring.profile_version_store is None
        or wiring.pointer_store is None
        or tenant_id is None
    ):
        return config

    resolved = resolve_profile_versions_for_request(
        tenant_id=tenant_id,
        task_class=task_class,
        routing_key=routing_key or tenant_id,
        adaptive_profile=profile,
        profile_store=wiring.profile_version_store,
        pointer_store=wiring.pointer_store,
    )
    config.resolved_profile_versions = resolved
    return config


def apply_resolved_profile_versions_to_runtime_config(
    config: RuntimeConfig,
    *,
    env: ApplicationEnvironmentProfile,
    resolved: ResolvedProfileVersions,
) -> RuntimeConfig:
    """Apply resolved active/candidate profile versions to runtime config surfaces."""
    rag_version = _select_version(resolved, ProfileArtifactType.RAG)
    if config.rag_profile is not None:
        config.rag_profile = apply_rag_profile_version(config.rag_profile, rag_version)

    orch_version = _select_version(resolved, ProfileArtifactType.ORCHESTRATION)
    env.orchestration_profile = apply_orchestration_profile_version(
        env.orchestration_profile,
        orch_version,
    )

    policy_version = _select_version(resolved, ProfileArtifactType.POLICY_FRAGMENT)
    if config.policy_bundle is not None and policy_version is not None:
        config.policy_bundle = apply_policy_fragment_version(config.policy_bundle, policy_version)

    return config


def _select_version(
    resolved: ResolvedProfileVersions,
    artifact_type: ProfileArtifactType,
) -> ProfileVersionRecord | None:
    key = artifact_type.value
    if resolved.use_candidate_for_traffic and key in resolved.candidate:
        return resolved.candidate[key]
    if key in resolved.active:
        return resolved.active[key]
    if key in resolved.candidate:
        return resolved.candidate[key]
    return None


def apply_adaptive_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
    *,
    wiring: ApplicationAdaptiveWiring | None = None,
    tenant_id: str | None = None,
    task_class: str = "",
    routing_key: str = "",
) -> RuntimeConfig:
    """Apply environment-declared adaptive profile to runtime config."""
    updated = apply_adaptive_profile_to_runtime_config(
        config,
        env.adaptive_profile,
        wiring=wiring,
        tenant_id=tenant_id or config.tenant_id,
        task_class=task_class,
        routing_key=routing_key,
    )
    if updated.resolved_profile_versions is not None:
        updated = apply_resolved_profile_versions_to_runtime_config(
            updated,
            env=env,
            resolved=updated.resolved_profile_versions,
        )
    return updated
