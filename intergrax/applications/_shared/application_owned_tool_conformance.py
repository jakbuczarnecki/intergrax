# © Artur Czarnecki. All rights reserved.

"""Conformance checks for application-owned tool declarations (PLATFORM-5B)."""

from __future__ import annotations

from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.application_owned_tools import ApplicationOwnedToolDeclaration
from intergrax.applications.contracts.errors import ApplicationManifestConformanceError
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.tools.registry.catalog import list_catalog_tool_ids
from intergrax.tools.registry.factory import enabled_tool_ids_for_profile
from intergrax.tools.registry.runtime import ToolRegistry


def declared_application_owned_tool_ids(
    manifest: ApplicationManifest,
) -> frozenset[str]:
    """Return declared application-owned tool ids from the manifest."""
    return frozenset(declaration.tool_id for declaration in manifest.application_owned_tools)


def platform_reserved_tool_ids() -> frozenset[str]:
    """Return platform catalog tool ids that application declarations must not shadow."""
    return frozenset(list_catalog_tool_ids())


def allowed_tool_closure(
    manifest: ApplicationManifest,
    *,
    platform_tool_ids: frozenset[str] | None = None,
) -> frozenset[str]:
    """Compute the allowed tool-id closure for conformance checks."""
    reserved = platform_tool_ids if platform_tool_ids is not None else platform_reserved_tool_ids()
    return reserved | declared_application_owned_tool_ids(manifest)


def merge_application_owned_tool_registry(
    *,
    catalog_registry: ToolRegistry,
    application_registry: ToolRegistry,
    declared_tool_ids: frozenset[str],
) -> ToolRegistry:
    """Merge declared application-owned tools into the catalog-backed registry."""
    for tool_id in application_registry.tool_ids():
        if tool_id not in declared_tool_ids:
            raise ApplicationManifestConformanceError(
                f"application tool registry contains undeclared tool id {tool_id!r}",
            )
        if catalog_registry.has(tool_id):
            raise ApplicationManifestConformanceError(
                f"application-owned tool {tool_id!r} collides with platform catalog registration",
            )
        registered = application_registry.get(tool_id)
        if registered.contract.tool_id != tool_id:
            raise ApplicationManifestConformanceError(
                f"application-owned tool registry identity mismatch for {tool_id!r}",
            )
        catalog_registry.register(registered.contract, registered.handler)
    return catalog_registry


def validate_application_owned_tool_conformance(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    snapshot: HarnessRegistrySnapshot,
    *,
    platform_tool_ids: frozenset[str] | None = None,
) -> list[str]:
    """Validate declaration/profile/registry agreement for application-owned tools."""
    violations: list[str] = []
    reserved = platform_tool_ids if platform_tool_ids is not None else platform_reserved_tool_ids()
    declared = declared_application_owned_tool_ids(manifest)
    declaration_ids = [declaration.tool_id for declaration in manifest.application_owned_tools]
    if len(declaration_ids) != len(set(declaration_ids)):
        violations.append("duplicate application-owned tool declarations detected")

    collisions = sorted(declared & reserved)
    if collisions:
        violations.append(
            "application-owned tool declarations collide with platform catalog ids: "
            + ", ".join(repr(tool_id) for tool_id in collisions),
        )

    closure = reserved | declared
    enabled = frozenset(enabled_tool_ids_for_profile(env.tool_profile))
    registry_ids = frozenset(snapshot.tool_ids())

    for tool_id in sorted(enabled):
        if tool_id not in closure:
            violations.append(f"enabled tool {tool_id!r} outside allowed closure")
        if tool_id not in registry_ids:
            violations.append(f"enabled tool {tool_id!r} missing from wired tool registry")

    for tool_id in sorted(declared & enabled):
        if tool_id not in registry_ids:
            violations.append(
                f"declared application-owned tool {tool_id!r} enabled but not registered",
            )
        elif snapshot.tool_registry is not None:
            registered = snapshot.tool_registry.get(tool_id)
            if registered.contract.tool_id != tool_id:
                violations.append(
                    f"registered application-owned tool identity mismatch for {tool_id!r}",
                )

    return violations


def assert_application_owned_tool_conformance(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    snapshot: HarnessRegistrySnapshot,
    *,
    platform_tool_ids: frozenset[str] | None = None,
) -> None:
    """Raise when application-owned tool conformance validation fails."""
    violations = validate_application_owned_tool_conformance(
        manifest,
        env,
        snapshot,
        platform_tool_ids=platform_tool_ids,
    )
    if violations:
        raise ApplicationManifestConformanceError("; ".join(violations))


def application_owned_tool_declarations(
    tool_ids: list[str] | tuple[str, ...],
) -> list[ApplicationOwnedToolDeclaration]:
    """Build manifest declarations from canonical tool ids."""
    return [ApplicationOwnedToolDeclaration(tool_id=tool_id) for tool_id in tool_ids]
