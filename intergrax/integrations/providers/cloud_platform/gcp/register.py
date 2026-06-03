# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register gcp in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.cloud_platform.gcp.bundle import create_gcp_cloud_platform
from intergrax.integrations.providers.cloud_platform.gcp.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_gcp_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_gcp_cloud_platform, override=override)
