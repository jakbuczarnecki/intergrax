# © Artur Czarnecki. All rights reserved.

"""ApplicationRegistry and EnvironmentRegistry sync, persistence, and queries (APP-OPS-4)."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.applications._shared.environment_snapshot_wiring import capture_environment_snapshot
from intergrax.applications._shared.health_score_wiring import (
    build_application_health_score,
    compute_environment_health_score,
)
from intergrax.applications._shared.ownership_wiring import standard_product_operational_ownership
from intergrax.applications._shared.package_wiring import (
    build_application_package,
    package_gate_environment,
)
from intergrax.applications._shared.product_manifest_registry import (
    iter_product_manifests,
    iter_strict_product_manifests,
)
from intergrax.applications.contracts.application_package import (
    ApplicationDistributionChannel,
    ApplicationPackage,
)
from intergrax.applications.contracts.application_registry import (
    ApplicationRegistry,
    ApplicationRegistryEntry,
    ApplicationRegistrySource,
    EnvironmentDeployment,
    EnvironmentDeploymentChannel,
    EnvironmentRegistry,
    EnvironmentRegistryEntry,
)
from intergrax.utils.time_provider import SystemTimeProvider

APPLICATION_REGISTRY_FILENAME = "application_registry.json"
ENVIRONMENT_REGISTRY_FILENAME = "environment_registry.json"


def application_registry_path(repo_root: Path) -> Path:
    """Default file path for the application registry artifact."""
    return repo_root / "build" / APPLICATION_REGISTRY_FILENAME


def environment_registry_path(repo_root: Path) -> Path:
    """Default file path for the environment registry artifact."""
    return repo_root / "build" / ENVIRONMENT_REGISTRY_FILENAME


def build_application_registry_entry(
    product_id: str,
    manifest,
    *,
    repo_root: Path,
    registered_at=None,
) -> ApplicationRegistryEntry:
    """Materialize one application registry row from a product manifest."""
    from intergrax.applications.contracts.manifest import ApplicationManifest

    assert isinstance(manifest, ApplicationManifest)
    gate_env = package_gate_environment(manifest.resolved_environment())
    package = build_application_package(
        manifest,
        gate_env,
        channel=ApplicationDistributionChannel.GIT,
    )
    health = build_application_health_score(product_id, manifest, repo_root=repo_root)
    ownership = manifest.ownership or standard_product_operational_ownership(manifest.app_id)
    return ApplicationRegistryEntry(
        app_id=manifest.app_id,
        name=manifest.name,
        current_version=manifest.version,
        package_ref=package,
        ownership=ownership,
        health=health,
        registered_at=registered_at or SystemTimeProvider.utc_now(),
        source=ApplicationRegistrySource.GIT,
    )


def build_environment_registry_entry(
    product_id: str,
    manifest,
    *,
    repo_root: Path,
    deployed_at=None,
) -> EnvironmentRegistryEntry:
    """Materialize one environment registry row for a STRICT product host."""
    from intergrax.applications.contracts.manifest import ApplicationManifest

    assert isinstance(manifest, ApplicationManifest)
    env = manifest.resolved_environment()
    snapshot = capture_environment_snapshot(manifest, env)
    health = compute_environment_health_score(
        product_id,
        manifest,
        repo_root=repo_root,
        environment_id=f"{manifest.app_id}-strict",
    )
    return EnvironmentRegistryEntry(
        environment_id=f"{manifest.app_id}-strict",
        app_id=manifest.app_id,
        app_version=manifest.version,
        profile_id=env.profile_id,
        execution_mode=env.execution_mode,
        deployment=EnvironmentDeployment(
            channel=EnvironmentDeploymentChannel.DOCKER,
            endpoint=manifest.route_prefix,
            image_tag=f"{manifest.app_id}:{manifest.version}",
            deployed_at=deployed_at or SystemTimeProvider.utc_now(),
            deployed_by="platform-registry-sync",
        ),
        snapshot_id=snapshot.snapshot_id,
        health=health,
    )


def build_application_registry(repo_root: Path) -> ApplicationRegistry:
    """Build application registry from canonical product manifests."""
    entries = [
        build_application_registry_entry(product_id, manifest, repo_root=repo_root)
        for product_id, manifest in iter_product_manifests()
    ]
    return ApplicationRegistry(entries=sorted(entries, key=lambda item: item.app_id))


def build_environment_registry(repo_root: Path) -> EnvironmentRegistry:
    """Build environment registry from STRICT product manifests."""
    entries = [
        build_environment_registry_entry(product_id, manifest, repo_root=repo_root)
        for product_id, manifest in iter_strict_product_manifests()
    ]
    return EnvironmentRegistry(entries=sorted(entries, key=lambda item: item.environment_id))


def save_application_registry(path: Path, registry: ApplicationRegistry) -> None:
    """Persist application registry JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def save_environment_registry(path: Path, registry: EnvironmentRegistry) -> None:
    """Persist environment registry JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _bootstrap_registry_validation() -> None:
    from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog

    bootstrap_application_integration_catalog()


def load_application_registry(path: Path) -> ApplicationRegistry:
    """Load application registry JSON."""
    _bootstrap_registry_validation()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ApplicationRegistry.model_validate(payload)


def load_environment_registry(path: Path) -> EnvironmentRegistry:
    """Load environment registry JSON."""
    _bootstrap_registry_validation()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EnvironmentRegistry.model_validate(payload)


def sync_platform_registries(repo_root: Path) -> tuple[ApplicationRegistry, EnvironmentRegistry]:
    """Rebuild and persist both ops registries."""
    app_registry = build_application_registry(repo_root)
    env_registry = build_environment_registry(repo_root)
    save_application_registry(application_registry_path(repo_root), app_registry)
    save_environment_registry(environment_registry_path(repo_root), env_registry)
    return app_registry, env_registry


def list_applications(repo_root: Path) -> list[ApplicationRegistryEntry]:
    """Return application registry entries, syncing when artifact is missing."""
    path = application_registry_path(repo_root)
    if path.is_file():
        return load_application_registry(path).entries
    return build_application_registry(repo_root).entries


def get_application(repo_root: Path, app_id: str) -> ApplicationRegistryEntry | None:
    """Return one application registry entry."""
    for entry in list_applications(repo_root):
        if entry.app_id == app_id:
            return entry
    return None


def list_environments(repo_root: Path, *, app_id: str | None = None) -> list[EnvironmentRegistryEntry]:
    """Return environment registry entries, optionally filtered by app."""
    path = environment_registry_path(repo_root)
    registry = (
        load_environment_registry(path)
        if path.is_file()
        else build_environment_registry(repo_root)
    )
    if app_id is None:
        return registry.entries
    return registry.list_for_app(app_id)


def get_environment(repo_root: Path, environment_id: str) -> EnvironmentRegistryEntry | None:
    """Return one environment registry entry."""
    path = environment_registry_path(repo_root)
    registry = (
        load_environment_registry(path)
        if path.is_file()
        else build_environment_registry(repo_root)
    )
    return registry.get(environment_id)


def register_application(repo_root: Path, package: ApplicationPackage) -> ApplicationRegistryEntry:
    """Upsert an application entry from a published package."""
    path = application_registry_path(repo_root)
    registry = load_application_registry(path) if path.is_file() else ApplicationRegistry()
    ownership = standard_product_operational_ownership(package.app_id)
    entry = ApplicationRegistryEntry(
        app_id=package.app_id,
        name=package.manifest.name,
        current_version=package.version,
        package_ref=package,
        ownership=ownership,
        health=None,
        registered_at=SystemTimeProvider.utc_now(),
        source=ApplicationRegistrySource.GIT,
    )
    remaining = [item for item in registry.entries if item.app_id != package.app_id]
    remaining.append(entry)
    updated = ApplicationRegistry(entries=sorted(remaining, key=lambda item: item.app_id))
    save_application_registry(path, updated)
    return entry


def format_application_entry(entry: ApplicationRegistryEntry) -> str:
    """Human-readable application summary."""
    health = "n/a"
    if entry.health is not None and entry.health.environments:
        health = f"{entry.health.environments[0].overall:.2f}"
    return (
        f"{entry.app_id:16} {entry.current_version:8} health={health:5} "
        f"owner={entry.ownership.owner.team}"
    )


def format_environment_entry(entry: EnvironmentRegistryEntry) -> str:
    """Human-readable environment summary."""
    health = f"{entry.health.overall:.2f}" if entry.health is not None else "n/a"
    return (
        f"{entry.environment_id:24} app={entry.app_id:12} mode={entry.execution_mode.value:8} "
        f"health={health:5} endpoint={entry.deployment.endpoint or 'n/a'}"
    )


def check_platform_registries(repo_root: Path) -> list[str]:
    """Validate synced registries cover all product manifests."""
    violations: list[str] = []
    app_registry, env_registry = sync_platform_registries(repo_root)

    expected_apps = {manifest.app_id for _, manifest in iter_product_manifests()}
    registered_apps = {entry.app_id for entry in app_registry.entries}
    missing_apps = sorted(expected_apps - registered_apps)
    if missing_apps:
        violations.append(f"application registry missing app_ids: {', '.join(missing_apps)}")

    expected_envs = {f"{manifest.app_id}-strict" for _, manifest in iter_strict_product_manifests()}
    registered_envs = {entry.environment_id for entry in env_registry.entries}
    missing_envs = sorted(expected_envs - registered_envs)
    if missing_envs:
        violations.append(f"environment registry missing environment_ids: {', '.join(missing_envs)}")

    for entry in app_registry.entries:
        if entry.package_ref is None:
            violations.append(f"{entry.app_id}: package_ref must be populated")
        if entry.health is None or not entry.health.production_ready:
            violations.append(f"{entry.app_id}: application health must be production_ready")

    app_path = application_registry_path(repo_root)
    env_path = environment_registry_path(repo_root)
    if not app_path.is_file():
        violations.append(f"missing {app_path}")
    if not env_path.is_file():
        violations.append(f"missing {env_path}")

    try:
        reloaded = load_application_registry(app_path)
        if len(reloaded.entries) != len(app_registry.entries):
            violations.append("application registry roundtrip entry count mismatch")
    except Exception as exc:
        violations.append(f"application registry roundtrip failed: {exc}")

    return violations
