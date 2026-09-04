# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register buildkite in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.buildkite.bundle import create_buildkite_ci_cd
from intergrax.integrations.providers.ci_cd.buildkite.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.ci_cd.buildkite.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_buildkite_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_buildkite_ci_cd,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
