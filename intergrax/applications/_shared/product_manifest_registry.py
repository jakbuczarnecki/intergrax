# © Artur Czarnecki. All rights reserved.

"""Canonical product host manifests for conformance gates (APP-PROD-7 · UC-A*)."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest

_PRODUCT_MANIFEST_MODULES: tuple[tuple[str, str, str], ...] = (
    ("legal", "legal_application.manifest", "LEGAL_APPLICATION_MANIFEST"),
    ("research", "research_application.manifest", "RESEARCH_APPLICATION_MANIFEST"),
    ("dispute_sim", "dispute_sim_application.manifest", "DISPUTE_SIM_APPLICATION_MANIFEST"),
    (
        "local_workspace",
        "local_workspace_application.manifest",
        "LOCAL_WORKSPACE_APPLICATION_MANIFEST",
    ),
)


def iter_product_manifests() -> Iterator[tuple[str, ApplicationManifest]]:
    """Yield `(product_id, manifest)` for shipped Tier-3 product hosts."""
    for product_id, module_name, attr in _PRODUCT_MANIFEST_MODULES:
        module = importlib.import_module(module_name)
        yield product_id, getattr(module, attr)


def iter_strict_product_manifests() -> Iterator[tuple[str, ApplicationManifest]]:
    """Product manifests whose resolved environment runs in STRICT mode."""
    for product_id, manifest in iter_product_manifests():
        if manifest.profile is not ApplicationProfile.PRODUCT:
            continue
        env = manifest.resolved_environment()
        if env.execution_mode is ExecutionMode.STRICT:
            yield product_id, manifest
