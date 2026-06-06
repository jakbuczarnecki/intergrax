# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register unleash in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.feature_flag.unleash.bundle import create_unleash_feature_flag
from intergrax.integrations.providers.feature_flag.unleash.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_unleash_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_unleash_feature_flag, override=override)
