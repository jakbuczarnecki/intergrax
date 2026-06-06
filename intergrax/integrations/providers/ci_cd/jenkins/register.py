# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register jenkins in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.jenkins.bundle import create_jenkins_ci_cd
from intergrax.integrations.providers.ci_cd.jenkins.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_jenkins_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_jenkins_ci_cd, override=override)
