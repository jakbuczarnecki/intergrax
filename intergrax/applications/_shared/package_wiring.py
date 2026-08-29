# © Artur Czarnecki. All rights reserved.

"""ApplicationPackage build, persistence, and dependency closure (APP-EVOL-7)."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.applications._shared.capability_graph_catalog import (
    resolve_binding_agent_contract_id,
)
from intergrax.applications._shared.capability_graph_wiring import EnvironmentCapabilityGraphView
from intergrax.applications._shared.environment_snapshot_wiring import stable_digest_hex
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.application_package import (
    ApplicationDependency,
    ApplicationDependencyKind,
    ApplicationDistribution,
    ApplicationDistributionChannel,
    ApplicationPackage,
    ApplicationPackageClosureError,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.architecture.capability_graph_applications import application_capability_node_id
from intergrax.skills.registry.factory import enabled_skill_ids_for_profile
from intergrax.tools.registry.factory import enabled_tool_ids_for_profile


def package_gate_environment(env: ApplicationEnvironmentProfile) -> ApplicationEnvironmentProfile:
    """Lab integration swap for CI package closure checks (no vendor drivers required)."""
    from intergrax.integrations.registry.profile import IntegrationProfile

    return env.model_copy(
        update={
            "integration_profile": IntegrationProfile.lab(),
            "application_profile": ApplicationProfile.LAB,
        },
    )


def package_id_for_app(app_id: str) -> str:
    """Return canonical reverse-DNS package id."""
    return f"com.intergrax.{app_id}"


def collect_application_dependencies(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
) -> list[ApplicationDependency]:
    """Collect direct dependencies from manifest and resolved environment."""
    dependencies: list[ApplicationDependency] = []

    for binding in manifest.enabled_agents():
        if binding.contract_id and binding.agent_type is None and binding.import_path is None:
            contract_ref = binding.contract_id
            version_constraint = "*"
        else:
            contract = binding.resolved_agent_type()().get_contract()
            contract_ref = contract.id
            version_constraint = f"={contract.version}"
        dependencies.append(
            ApplicationDependency(
                kind=ApplicationDependencyKind.AGENT,
                ref=contract_ref,
                version_constraint=version_constraint,
            ),
        )

    for skill_id in sorted(enabled_skill_ids_for_profile(env.skill_profile)):
        dependencies.append(
            ApplicationDependency(
                kind=ApplicationDependencyKind.SKILL,
                ref=skill_id,
                version_constraint="*",
            ),
        )

    for tool_id in sorted(enabled_tool_ids_for_profile(env.tool_profile)):
        dependencies.append(
            ApplicationDependency(
                kind=ApplicationDependencyKind.TOOL,
                ref=tool_id,
                version_constraint="*",
            ),
        )

    integration = env.integration_profile or manifest.integration_profile
    if integration is not None:
        dependencies.extend(_integration_dependencies(integration))

    dependencies.append(
        ApplicationDependency(
            kind=ApplicationDependencyKind.PROFILE_FRAGMENT,
            ref=env.profile_id,
            version_constraint=f"={env.spec_version}",
        ),
    )
    return dependencies


def _integration_dependencies(profile: IntegrationProfile) -> list[ApplicationDependency]:
    dependencies: list[ApplicationDependency] = []
    for field_name in IntegrationProfile._SLUG_FIELDS:
        binding = profile.binding_for_field(field_name)
        if binding is None:
            continue
        slug = binding.resolved_slug()
        if not slug:
            continue
        dependencies.append(
            ApplicationDependency(
                kind=ApplicationDependencyKind.INTEGRATION,
                ref=slug,
                version_constraint="*",
            ),
        )
    return dependencies


def compute_package_checksum(package: ApplicationPackage) -> str:
    """Return sha256 hex digest for an immutable package artifact."""
    payload = package.model_copy(
        update={
            "distribution": package.distribution.model_copy(update={"checksum": ""}),
        },
    )
    return stable_digest_hex(payload.model_dump(mode="json"))


def build_application_package(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    *,
    channel: ApplicationDistributionChannel = ApplicationDistributionChannel.GIT,
    artifact_uri: str | None = None,
) -> ApplicationPackage:
    """Materialize an ApplicationPackage from a live manifest and environment."""
    dependencies = collect_application_dependencies(manifest, env)
    package = ApplicationPackage(
        package_id=package_id_for_app(manifest.app_id),
        app_id=manifest.app_id,
        version=manifest.version,
        manifest=manifest,
        dependencies=dependencies,
        distribution=ApplicationDistribution(
            channel=channel,
            artifact_uri=artifact_uri,
        ),
    )
    checksum = compute_package_checksum(package)
    return package.model_copy(
        update={
            "distribution": package.distribution.model_copy(update={"checksum": checksum}),
        },
    )


def load_application_package(path: Path) -> ApplicationPackage:
    """Parse a package JSON document."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ApplicationPackage.model_validate(payload)


def write_application_package_json(path: Path, package: ApplicationPackage, *, force: bool = False) -> None:
    """Persist a package artifact as canonical JSON."""
    if path.exists() and not force:
        raise FileExistsError(f"Package file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(package.model_dump(mode="json"), indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def validate_application_package_closure(
    package: ApplicationPackage,
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    snapshot: HarnessRegistrySnapshot,
    *,
    capability_graph: EnvironmentCapabilityGraphView | None = None,
) -> list[str]:
    """Verify package dependencies against wired harness catalogs."""
    violations: list[str] = []

    if package.app_id != manifest.app_id:
        violations.append(
            f"package app_id {package.app_id!r} does not match manifest {manifest.app_id!r}",
        )
    if package.version != manifest.version:
        violations.append(
            f"package version {package.version!r} does not match manifest {manifest.version!r}",
        )

    expected = collect_application_dependencies(manifest, env)
    if _dependency_keys(package.dependencies) != _dependency_keys(expected):
        violations.append("package.dependencies drift from manifest/environment materialization")

    roster_contract_ids = {
        resolve_binding_agent_contract_id(binding) for binding in manifest.enabled_agents()
    }
    for dep in package.dependencies:
        if dep.kind is ApplicationDependencyKind.AGENT and dep.ref not in roster_contract_ids:
            violations.append(f"package agent dependency {dep.ref!r} not on enabled roster")

    snapshot_tools = frozenset(snapshot.tool_ids())
    snapshot_skills = frozenset(snapshot.skill_ids())
    for dep in package.dependencies:
        if dep.kind is ApplicationDependencyKind.TOOL and dep.ref not in snapshot_tools:
            violations.append(f"tool dependency {dep.ref!r} missing from wired tool registry")
        if dep.kind is ApplicationDependencyKind.SKILL and dep.ref not in snapshot_skills:
            violations.append(f"skill dependency {dep.ref!r} missing from wired skill registry")

    integration = env.integration_profile or manifest.integration_profile
    if integration is not None:
        integration_slugs = _integration_slugs(integration)
        for dep in package.dependencies:
            if dep.kind is not ApplicationDependencyKind.INTEGRATION:
                continue
            if dep.ref.strip().lower() not in integration_slugs:
                violations.append(f"integration dependency {dep.ref!r} not declared on profile")

    violations.extend(_validate_graph_spec_capabilities(manifest, env))

    if capability_graph is not None:
        app_node = application_capability_node_id(manifest.app_id)
        if not capability_graph.contains_node(app_node):
            violations.append(f"capability graph missing application node {app_node!r}")

    if package.distribution.checksum:
        expected_checksum = compute_package_checksum(
            package.model_copy(
                update={
                    "distribution": package.distribution.model_copy(update={"checksum": ""}),
                },
            ),
        )
        if package.distribution.checksum != expected_checksum:
            violations.append("package distribution checksum mismatch")

    return violations


def assert_manifest_package_closure(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    snapshot: HarnessRegistrySnapshot,
    *,
    capability_graph: EnvironmentCapabilityGraphView | None = None,
) -> None:
    """Build and validate package closure during environment wiring."""
    package = build_application_package(manifest, env)
    violations = validate_application_package_closure(
        package,
        manifest,
        env,
        snapshot,
        capability_graph=capability_graph,
    )
    if violations:
        raise ApplicationPackageClosureError(violations)


def build_scaffold_application_package(
    *,
    app_id: str,
    version: str,
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
) -> ApplicationPackage:
    """Build a local-channel package for scaffolded applications."""
    return build_application_package(
        manifest,
        env,
        channel=ApplicationDistributionChannel.LOCAL,
        artifact_uri=None,
    )


def _dependency_keys(dependencies: list[ApplicationDependency]) -> frozenset[tuple[str, str, str, bool]]:
    return frozenset(
        (dep.kind.value, dep.ref, dep.version_constraint, dep.optional) for dep in dependencies
    )


def _integration_slugs(profile: IntegrationProfile) -> frozenset[str]:
    slugs: set[str] = set()
    for field_name in IntegrationProfile._SLUG_FIELDS:
        binding = profile.binding_for_field(field_name)
        if binding is None:
            continue
        resolved = binding.resolved_slug()
        if resolved:
            slugs.add(resolved.strip().lower())
    return frozenset(slugs)


def _validate_graph_spec_capabilities(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
) -> list[str]:
    graph = env.graph_spec
    if graph is None:
        return []

    violations: list[str] = []
    try:
        graph.validate_against_roster(manifest.enabled_agents())
    except ValueError as exc:
        violations.append(str(exc))

    return violations
