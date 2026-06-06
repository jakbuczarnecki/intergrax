# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register circleci in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.circleci.bundle import create_circleci_ci_cd
from intergrax.integrations.providers.ci_cd.circleci.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_circleci_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_circleci_ci_cd, override=override)
