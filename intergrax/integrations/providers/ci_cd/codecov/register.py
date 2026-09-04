# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register codecov in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.codecov.bundle import create_codecov_ci_cd
from intergrax.integrations.providers.ci_cd.codecov.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.ci_cd.codecov.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_codecov_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_codecov_ci_cd,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
