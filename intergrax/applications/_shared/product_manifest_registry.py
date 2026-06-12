# © Artur Czarnecki. All rights reserved.

"""Canonical product host manifests for conformance gates (APP-PROD-7 · UC-A*)."""

from __future__ import annotations

from collections.abc import Iterator

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest


def iter_product_manifests() -> Iterator[tuple[str, ApplicationManifest]]:
    """Yield ``(product_id, manifest)`` for shipped Tier-3 product hosts."""
    from dispute_sim_application.manifest import DISPUTE_SIM_APPLICATION_MANIFEST
    from legal_application.manifest import LEGAL_APPLICATION_MANIFEST
    from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
    from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

    yield "legal", LEGAL_APPLICATION_MANIFEST
    yield "research", RESEARCH_APPLICATION_MANIFEST
    yield "dispute_sim", DISPUTE_SIM_APPLICATION_MANIFEST
    yield "local_workspace", LOCAL_WORKSPACE_APPLICATION_MANIFEST


def iter_strict_product_manifests() -> Iterator[tuple[str, ApplicationManifest]]:
    """Product manifests whose resolved environment runs in STRICT mode."""
    for product_id, manifest in iter_product_manifests():
        if manifest.profile is not ApplicationProfile.PRODUCT:
            continue
        env = manifest.resolved_environment()
        if env.execution_mode is ExecutionMode.STRICT:
            yield product_id, manifest
