# © Artur Czarnecki. All rights reserved.

"""Production control-plane governance wiring (PROD-Q-R1-R7-D2).

Reference harness policy bundles are forbidden as production mutation authority.
Callers must inject an operator-approved ``ProductionControlPlaneMutationAuthorizationBoundary``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

ProductionControlPlaneMutationAuthorizationBoundary: TypeAlias = (
    ControlPlaneMutationAuthorizationBoundary
)


@dataclass(frozen=True, slots=True)
class ProductionControlPlaneGovernance:
    """Trusted principal, tenant resolver, and production mutation policy boundary."""

    principal: RequestIdentity
    mutation_authorization_boundary: ProductionControlPlaneMutationAuthorizationBoundary
    environment_tenant_resolver: ApplicationEnvironmentTenantResolver


__all__ = [
    "ProductionControlPlaneGovernance",
    "ProductionControlPlaneMutationAuthorizationBoundary",
]
