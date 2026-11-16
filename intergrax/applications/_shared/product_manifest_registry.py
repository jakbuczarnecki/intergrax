# © Artur Czarnecki. All rights reserved.

"""Canonical product host manifests for conformance gates (APP-PROD-7 · UC-A*)."""

from __future__ import annotations

from collections.abc import Iterator

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest

_REGISTERED: dict[str, ApplicationManifest] = {}


def register_product_manifest(product_id: str, manifest: ApplicationManifest) -> None:
    """Register one shipped product manifest (invoked from Tier-3 catalog wiring only)."""
    if product_id in _REGISTERED and _REGISTERED[product_id] is not manifest:
        raise ValueError(f"conflicting manifest registration for product_id={product_id!r}")
    _REGISTERED[product_id] = manifest


def iter_product_manifests() -> Iterator[tuple[str, ApplicationManifest]]:
    """Yield `(product_id, manifest)` for shipped Tier-3 product hosts."""
    for product_id in sorted(_REGISTERED):
        yield product_id, _REGISTERED[product_id]


def iter_strict_product_manifests() -> Iterator[tuple[str, ApplicationManifest]]:
    """Product manifests whose resolved environment runs in STRICT mode."""
    for product_id, manifest in iter_product_manifests():
        if manifest.profile is not ApplicationProfile.PRODUCT:
            continue
        env = manifest.resolved_environment()
        if env.execution_mode is ExecutionMode.STRICT:
            yield product_id, manifest
