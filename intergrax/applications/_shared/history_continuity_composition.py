# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical history-continuity composition (D03-A-LOCAL-01).

``ProductionControlPlaneGovernance`` remains the semantic owner. This function
is the only supported constructor. Missing or rejected activation returns a
deny-only authority and does not wire the CREATE dispatch gate.
"""

from __future__ import annotations

from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.contracts.agent_run import canonical_principal_id_from_request_identity
from intergrax.contracts.governance.history_continuity import (
    HistoryContinuityActivation,
    HistoryContinuityAdapterKind,
    HistoryContinuityAuthority,
)
from intergrax.runtime.governance.local_history_continuity import (
    InactiveHistoryContinuityAuthority,
    _open_local_history_continuity,
)


def compose_history_continuity_authority(
    governance: ProductionControlPlaneGovernance,
    activation: HistoryContinuityActivation | None,
) -> HistoryContinuityAuthority:
    """Compose the local continuity authority, or a deny-only authority."""

    if activation is None or activation.enabled is not True:
        return InactiveHistoryContinuityAuthority()
    if activation.adapter is not HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC:
        return InactiveHistoryContinuityAuthority()
    try:
        issuer = canonical_principal_id_from_request_identity(governance.principal)
        resolved = governance.environment_tenant_resolver.resolve_tenant_id(
            application_id=activation.application_id,
            application_environment_id=activation.environment_id,
        )
    except Exception:
        return InactiveHistoryContinuityAuthority()
    if governance.principal.tenant_id != resolved:
        return InactiveHistoryContinuityAuthority()
    return _open_local_history_continuity(
        tenant_resolver=governance.environment_tenant_resolver,
        activation=activation,
        authorization_boundary=governance.mutation_authorization_boundary,
        principal=governance.principal,
        issuer_principal_id=issuer,
    )
