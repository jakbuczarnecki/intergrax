# © Artur Czarnecki. All rights reserved.

"""Capture and wire :class:`EnvironmentSnapshot` on task intake (APP-EVOL-1)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    bundle_normalized_payload,
)
from intergrax.applications.contracts.environment_snapshot import (
    ENV_SNAPSHOT_RUNTIME_KEY,
    EnvironmentSnapshot,
    SnapshotCaptureSource,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
    from intergrax.runtime.nexus.nexus_loop import NexusLoop


def stable_digest_hex(payload: Any) -> str:
    """Return sha256 hex digest of canonical JSON."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_manifest_digest(manifest: ApplicationManifest) -> str:
    """Digest of deployable manifest identity (roster paths + version)."""
    agents: list[dict[str, Any]] = []
    for binding in manifest.agents:
        if not binding.enabled:
            continue
        agents.append(_binding_digest_payload(binding))
    agents.sort(key=lambda item: (item.get("import_path") or "", item.get("contract_id") or ""))
    payload = {
        "app_id": manifest.app_id,
        "version": manifest.version,
        "profile": manifest.profile.value,
        "agents": agents,
    }
    return stable_digest_hex(payload)


def compute_roster_digest(
    manifest: ApplicationManifest,
    *,
    registry_snapshot: HarnessRegistrySnapshot | None = None,
) -> str:
    """Digest of resolved agent roster — prefers live registry contract ids."""
    if registry_snapshot is not None:
        contract_ids = registry_snapshot.agent_contract_ids()
        if contract_ids:
            return stable_digest_hex({"agent_contract_ids": list(contract_ids)})
    bindings: list[dict[str, Any]] = []
    for binding in manifest.agents:
        if binding.enabled:
            bindings.append(_binding_digest_payload(binding))
    bindings.sort(key=lambda item: (item.get("import_path") or "", item.get("contract_id") or ""))
    return stable_digest_hex({"bindings": bindings})


_DEPLOY_SNAPSHOT_CACHE: dict[str, EnvironmentSnapshot] = {}


def compute_profile_snapshot_id(environment: ApplicationEnvironmentProfile) -> str:
    """Stable fingerprint of the resolved environment profile (bundle-normalized)."""
    payload = bundle_normalized_payload(environment.bundle_dump(mode="json"))
    return f"prof_{stable_digest_hex(payload)[:24]}"


def _deploy_cache_key(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    registry_snapshot: HarnessRegistrySnapshot | None = None,
) -> str:
    roster_digest = compute_roster_digest(manifest, registry_snapshot=registry_snapshot)
    return "|".join(
        (
            manifest.app_id,
            manifest.version,
            compute_profile_snapshot_id(environment),
            roster_digest,
        ),
    )


def cache_deploy_environment_snapshot(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    registry_snapshot: HarnessRegistrySnapshot | None = None,
) -> EnvironmentSnapshot:
    """Materialize and cache snapshot at host bootstrap (APP-EVOL-8 deploy cache)."""
    snapshot = capture_environment_snapshot(
        manifest,
        environment,
        registry_snapshot=registry_snapshot,
        captured_by=SnapshotCaptureSource.DEPLOY,
    )
    _DEPLOY_SNAPSHOT_CACHE[_deploy_cache_key(manifest, environment, registry_snapshot=registry_snapshot)] = (
        snapshot
    )
    return snapshot


def resolve_cached_environment_snapshot(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    registry_snapshot: HarnessRegistrySnapshot | None = None,
) -> EnvironmentSnapshot | None:
    """Return deploy-cached snapshot when manifest/profile/roster fingerprint matches."""
    return _DEPLOY_SNAPSHOT_CACHE.get(
        _deploy_cache_key(manifest, environment, registry_snapshot=registry_snapshot),
    )


def compute_snapshot_id(    *,
    manifest_digest: str,
    profile_snapshot_id: str,
    roster_digest: str,
    graph_spec_digest: str | None,
    org_envelope_digest: str | None,
) -> str:
    """Derive a stable snapshot id from component digests."""
    material = "|".join(
        (
            manifest_digest,
            profile_snapshot_id,
            roster_digest,
            graph_spec_digest or "",
            org_envelope_digest or "",
        )
    )
    return f"envsnap_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def capture_environment_snapshot(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    registry_snapshot: HarnessRegistrySnapshot | None = None,
    captured_by: SnapshotCaptureSource = SnapshotCaptureSource.INTAKE,
    use_deploy_cache: bool = True,
) -> EnvironmentSnapshot:
    """Materialize an :class:`EnvironmentSnapshot` from resolved host artifacts."""
    if captured_by == SnapshotCaptureSource.INTAKE and use_deploy_cache:
        cached = resolve_cached_environment_snapshot(
            manifest,
            environment,
            registry_snapshot=registry_snapshot,
        )
        if cached is not None:
            return cached.model_copy(update={"captured_by": SnapshotCaptureSource.INTAKE})
    manifest_digest = compute_manifest_digest(manifest)
    roster_digest = compute_roster_digest(manifest, registry_snapshot=registry_snapshot)
    profile_snapshot_id = compute_profile_snapshot_id(environment)
    graph_spec_digest = (
        stable_digest_hex(environment.graph_spec.model_dump(mode="json"))
        if environment.graph_spec is not None
        else None
    )
    org_envelope_digest = (
        stable_digest_hex(environment.organizational_policy.model_dump(mode="json"))
        if environment.organizational_policy is not None
        else None
    )
    snapshot_id = compute_snapshot_id(
        manifest_digest=manifest_digest,
        profile_snapshot_id=profile_snapshot_id,
        roster_digest=roster_digest,
        graph_spec_digest=graph_spec_digest,
        org_envelope_digest=org_envelope_digest,
    )
    return EnvironmentSnapshot(
        snapshot_id=snapshot_id,
        app_id=manifest.app_id,
        app_version=manifest.version,
        profile_snapshot_id=profile_snapshot_id,
        manifest_digest=manifest_digest,
        graph_spec_digest=graph_spec_digest,
        org_envelope_digest=org_envelope_digest,
        roster_digest=roster_digest,
        captured_at=SystemTimeProvider.utc_now().isoformat(),
        captured_by=captured_by,
    )


def _binding_digest_payload(binding: AgentBinding) -> dict[str, Any]:
    return {
        "import_path": binding.import_path,
        "contract_id": binding.contract_id,
        "capabilities": sorted(binding.capabilities),
        "config": binding.config,
        "enabled": binding.enabled,
        "requires_uaep": binding.requires_uaep,
    }


def apply_environment_snapshot_wiring(
    nexus: NexusLoop,
    *,
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    registry_snapshot: HarnessRegistrySnapshot | None = None,
) -> None:
    """Attach intake snapshot middleware (priority 35 — before env-state sync)."""
    from intergrax.applications._shared.application_host_wiring import _attach_middleware
    from intergrax.applications._shared.environment_snapshot_middleware import (
        EnvironmentSnapshotMiddleware,
    )

    _attach_middleware(
        nexus,
        EnvironmentSnapshotMiddleware(
            manifest=manifest,
            environment=environment,
            registry_snapshot=registry_snapshot,
        ),
    )


__all__ = [
    "ENV_SNAPSHOT_RUNTIME_KEY",
    "apply_environment_snapshot_wiring",
    "cache_deploy_environment_snapshot",
    "capture_environment_snapshot",
    "compute_profile_snapshot_id",
    "resolve_cached_environment_snapshot",
    "stable_digest_hex",
]
