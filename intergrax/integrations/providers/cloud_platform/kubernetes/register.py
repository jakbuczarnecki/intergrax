# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register kubernetes in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.cloud_platform.kubernetes.bundle import create_kubernetes_cloud_platform
from intergrax.integrations.providers.cloud_platform.kubernetes.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_kubernetes_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_kubernetes_cloud_platform, override=override)
