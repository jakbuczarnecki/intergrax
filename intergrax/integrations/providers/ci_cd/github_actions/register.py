# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register github_actions in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.github_actions.bundle import create_github_actions_ci_cd
from intergrax.integrations.providers.ci_cd.github_actions.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.ci_cd.github_actions.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_github_actions_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_github_actions_ci_cd,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
