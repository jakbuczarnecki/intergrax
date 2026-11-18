# © Artur Czarnecki. All rights reserved.

"""Permitted import sets for the importlib seams, derived from the registered product manifests.

T-IGX-051-I (A4-007 seams b1 and b2). The loaders (``resolve_agent_type``, ``load_callable``,
``_load_manifest_for_package``) accept an :class:`ImportPrefixAllowlist` and ``fail_closed``; this
module computes both for a caller. The permitted packages come from one registered source, the
product manifest registry, read on every call (it is mutable through ``register_product_manifest``);
no package name is typed here. The guard posture is decided only by ``wiring_fails_closed``.
In a guard posture an empty registry yields empty allowlists, which permit nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from intergrax.applications._shared.import_allowlist import ImportPrefixAllowlist
from intergrax.applications._shared.product_manifest_registry import iter_product_manifests
from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest


@dataclass(frozen=True)
class ImportPolicy:
    """Allowlists for the three importlib seams plus the fail-closed flag (``None`` = unrestricted)."""

    agent_modules: ImportPrefixAllowlist | None
    factory_modules: ImportPrefixAllowlist | None
    application_packages: ImportPrefixAllowlist | None
    fail_closed: bool


UNRESTRICTED = ImportPolicy(
    agent_modules=None,
    factory_modules=None,
    application_packages=None,
    fail_closed=False,
)


def _top_level_package(dotted_path: str) -> str:
    return dotted_path.split(".", 1)[0]


def _allowlist(packages: Iterable[str]) -> ImportPrefixAllowlist:
    return ImportPrefixAllowlist(prefixes=tuple(sorted(set(packages))))


def registered_agent_prefixes() -> ImportPrefixAllowlist:
    """Top-level packages of every ``import_path`` in the registered product manifests."""
    return _allowlist(
        _top_level_package(binding.import_path)
        for _, manifest in iter_product_manifests()
        for binding in manifest.agents
        if binding.import_path is not None
    )


def registered_factory_prefixes() -> ImportPrefixAllowlist:
    """Top-level packages of every ``factory_path`` in the registered product manifests."""
    return _allowlist(
        _top_level_package(binding.factory_path)
        for _, manifest in iter_product_manifests()
        for binding in manifest.agents
        if binding.factory_path is not None
    )


def registered_application_prefixes() -> ImportPrefixAllowlist:
    """``<product_id>_application`` for every registered product id that forms a valid identifier."""
    return _allowlist(
        package
        for package in (f"{product_id}_application" for product_id, _ in iter_product_manifests())
        if package.isidentifier()
    )


def guarded_import_policy() -> ImportPolicy:
    """Fail-closed policy with the three sets derived from the registry (used by the gate scripts)."""
    return ImportPolicy(
        agent_modules=registered_agent_prefixes(),
        factory_modules=registered_factory_prefixes(),
        application_packages=registered_application_prefixes(),
        fail_closed=True,
    )


def import_policy_for_environment(env: ApplicationEnvironmentProfile) -> ImportPolicy:
    """Guarded policy when ``wiring_fails_closed(env)``, else :data:`UNRESTRICTED`."""
    if wiring_fails_closed(env):
        return guarded_import_policy()
    return UNRESTRICTED


def import_policy_for_manifest(manifest: ApplicationManifest) -> ImportPolicy:
    """:func:`import_policy_for_environment` over ``manifest.resolved_environment()``."""
    return import_policy_for_environment(manifest.resolved_environment())


__all__ = [
    "UNRESTRICTED",
    "ImportPolicy",
    "guarded_import_policy",
    "import_policy_for_environment",
    "import_policy_for_manifest",
    "registered_agent_prefixes",
    "registered_application_prefixes",
    "registered_factory_prefixes",
]
