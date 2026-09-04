# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Argocd ci cd."""

from __future__ import annotations

from intergrax.integrations.providers.ci_cd.argocd.bundle import (
    create_argocd_ci_cd_integration,
)
from intergrax.integrations.providers.ci_cd.argocd.integration import (
    ARGOCD_CI_CD_PROVIDER_ID,
    ArgocdCiCdIntegration,
    ArgocdCiCdIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="ci_cd",
    provider_id=ARGOCD_CI_CD_PROVIDER_ID,
    integration_class=ArgocdCiCdIntegration,
    contract_class=CiCdIntegrationContract,
    contract_factory=create_argocd_ci_cd_integration,
    display_name="Argocd",
    config_class=ArgocdCiCdIntegrationConfig,
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
