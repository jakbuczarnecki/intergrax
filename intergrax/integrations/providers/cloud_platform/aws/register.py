# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register aws in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.cloud_platform.aws.bundle import create_aws_cloud_platform
from intergrax.integrations.providers.cloud_platform.aws.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.cloud_platform.aws.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_aws_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_aws_cloud_platform,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
