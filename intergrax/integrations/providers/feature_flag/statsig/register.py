# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register statsig in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.feature_flag.statsig.bundle import create_statsig_feature_flag
from intergrax.integrations.providers.feature_flag.statsig.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_statsig_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_statsig_feature_flag, override=override)
