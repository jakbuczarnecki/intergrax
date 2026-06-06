# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register launchdarkly in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.feature_flag.launchdarkly.bundle import create_launchdarkly_feature_flag
from intergrax.integrations.providers.feature_flag.launchdarkly.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_launchdarkly_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_launchdarkly_feature_flag, override=override)
