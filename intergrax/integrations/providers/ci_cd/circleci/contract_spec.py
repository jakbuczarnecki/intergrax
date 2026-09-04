# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Circleci ci cd."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.circleci.bundle import (
    create_circleci_ci_cd_integration,
)
from intergrax.integrations.providers.ci_cd.circleci.integration import (
    CIRCLECI_CI_CD_PROVIDER_ID,
    CircleciCiCdIntegration,
    CircleciCiCdIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="ci_cd",
    provider_id=CIRCLECI_CI_CD_PROVIDER_ID,
    integration_class=CircleciCiCdIntegration,
    contract_class=CiCdIntegrationContract,
    contract_factory=create_circleci_ci_cd_integration,
    display_name="Circleci",
    config_class=CircleciCiCdIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
