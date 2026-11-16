# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 catalog of shipped product manifests (registers into Tier-0 registry)."""

from __future__ import annotations

from intergrax.applications._shared.product_manifest_registry import register_product_manifest
from intergrax.applications.contracts.manifest import ApplicationManifest

from dispute_sim_application.manifest import DISPUTE_SIM_APPLICATION_MANIFEST
from legal_application.manifest import LEGAL_APPLICATION_MANIFEST
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

_SHIPPED: tuple[tuple[str, ApplicationManifest], ...] = (
    ("legal", LEGAL_APPLICATION_MANIFEST),
    ("research", RESEARCH_APPLICATION_MANIFEST),
    ("dispute_sim", DISPUTE_SIM_APPLICATION_MANIFEST),
    ("local_workspace", LOCAL_WORKSPACE_APPLICATION_MANIFEST),
)


def ensure_shipped_product_manifests_registered() -> None:
    """Idempotently register canonical shipped product manifests."""
    for product_id, manifest in _SHIPPED:
        register_product_manifest(product_id, manifest)


ensure_shipped_product_manifests_registered()
