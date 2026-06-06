# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register localstack in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.cloud_platform.localstack.bundle import create_localstack_cloud_platform
from intergrax.integrations.providers.cloud_platform.localstack.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_localstack_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_localstack_cloud_platform, override=override)
