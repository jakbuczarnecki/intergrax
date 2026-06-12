# © Artur Czarnecki. All rights reserved.

"""Emit ApplicationPackage JSON for scaffolded Tier-3 hosts (APP-EVOL-7)."""

from __future__ import annotations

import sys
from pathlib import Path

from intergrax.applications._shared.package_wiring import (
    build_scaffold_application_package,
    write_application_package_json,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.scaffold.agent_catalog import ScaffoldAgentSpec
from intergrax.scaffold.application_names import ScaffoldApplicationNames


def _bindings_from_specs(specs: list[ScaffoldAgentSpec]) -> list[AgentBinding]:
    bindings: list[AgentBinding] = []
    for index, spec in enumerate(specs):
        bindings.append(
            AgentBinding.deserialize(
                import_path=f"{spec.module}.{spec.class_name}",
                capabilities=list(spec.capabilities),
                default=index == 0,
            ),
        )
    return bindings


def build_scaffold_manifest_and_env(
    names: ScaffoldApplicationNames,
    specs: list[ScaffoldAgentSpec],
    profile: str,
) -> tuple[ApplicationManifest, ApplicationEnvironmentProfile]:
    """Build manifest and environment matching scaffold templates."""
    agents = _bindings_from_specs(specs)
    if profile == "product":
        environment = ApplicationEnvironmentProfile.product_defaults(
            profile_id=f"{names.short}.product",
        )
        manifest = ApplicationManifest.product(
            app_id=names.short,
            name=f"{names.display} API",
            route_prefix=names.route_prefix,
            env_prefix=names.env_prefix,
            default_port=names.port,
            integration_profile=IntegrationProfile.lab_stack(),
            environment=environment,
            agents=agents,
            description=f"Scaffolded Tier-3 product environment ({names.pkg})",
        )
        return manifest, environment

    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id=f"{names.short}.scaffold")
    manifest = ApplicationManifest.lab(
        app_id=names.short,
        name=f"{names.display} Lab Application",
        route_prefix=names.route_prefix,
        env_prefix=names.env_prefix,
        integration_profile=IntegrationProfile.lab_stack(),
        environment=environment,
        agents=agents,
        description="Scaffolded Tier-3 lab environment (Phase DX-1.5)",
    )
    return manifest, environment


def _ensure_scaffold_pythonpath(target: Path) -> None:
    """Add generated agent/application roots for roster resolution during packaging."""
    repo_root = target.parent.parent
    agents_root = repo_root / "agents"
    applications_root = repo_root / "applications"
    for path in (repo_root, agents_root, applications_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def write_scaffold_package_json(
    target: Path,
    names: ScaffoldApplicationNames,
    specs: list[ScaffoldAgentSpec],
    profile: str,
    *,
    force: bool,
) -> None:
    """Write ``package.json`` beside a scaffolded application tree."""
    _ensure_scaffold_pythonpath(target)
    manifest, environment = build_scaffold_manifest_and_env(names, specs, profile)
    package = build_scaffold_application_package(
        app_id=manifest.app_id,
        version=manifest.version,
        manifest=manifest,
        env=environment,
    )
    write_application_package_json(target / "package.json", package, force=force)
