# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register gitlab_ci in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.gitlab_ci.bundle import create_gitlab_ci_ci_cd
from intergrax.integrations.providers.ci_cd.gitlab_ci.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.ci_cd.gitlab_ci.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_gitlab_ci_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_gitlab_ci_ci_cd,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
