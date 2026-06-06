# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register argocd in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.argocd.bundle import create_argocd_ci_cd
from intergrax.integrations.providers.ci_cd.argocd.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_argocd_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_argocd_ci_cd, override=override)
