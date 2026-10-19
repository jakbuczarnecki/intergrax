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
    HistoryContinuityAuthority,
    HistoryContinuityProviderSelection,
    history_continuity_activation_is_admitted,
    history_continuity_offline_provider_execution_permitted,
)
from intergrax.runtime.governance.continuity_providers.credentials import (
    LiveCredentialRefusal,
    evaluate_continuity_credentials,
)
from intergrax.runtime.governance.continuity_providers.primary_register import (
    InMemoryPrimaryDatabase,
)
from intergrax.runtime.governance.continuity_providers.witness_log import (
    InMemoryAppendWitness,
)
from intergrax.runtime.governance.local_history_continuity import (
    InactiveHistoryContinuityAuthority,
    _open_local_history_continuity,
)
from intergrax.runtime.governance.offline_history_continuity import (
    OfflineProviderBundle,
    open_offline_external_history_continuity,
)


def compose_history_continuity_authority(
    governance: ProductionControlPlaneGovernance,
    activation: HistoryContinuityActivation | None,
    *,
    provider_selection: HistoryContinuityProviderSelection | None = None,
    offline_bundle: OfflineProviderBundle | None = None,
) -> HistoryContinuityAuthority:
    """Compose one continuity authority, or a deny-only authority.

    Local activation and the offline external pair are mutually exclusive.
    Recognizing the external pair does not admit it. A missing
    identity binding, a live credential, or a missing offline transport returns
    the deny-only authority. This function does not open a cloud session and
    does not call the CREATE dispatch gate.
    """

    if provider_selection is None and offline_bundle is None:
        return _compose_local(governance, activation)
    if activation is not None or provider_selection is None or offline_bundle is None:
        return InactiveHistoryContinuityAuthority()
    if not history_continuity_offline_provider_execution_permitted(provider_selection):
        return InactiveHistoryContinuityAuthority()
    if not _offline_bundle_is_injected(offline_bundle):
        return InactiveHistoryContinuityAuthority()
    decision = evaluate_continuity_credentials(provider_selection.identity_binding)
    if decision.token is not None or decision.refused is not LiveCredentialRefusal.QUALIFICATION_ABSENT:
        return InactiveHistoryContinuityAuthority()
    try:
        issuer = canonical_principal_id_from_request_identity(governance.principal)
        resolved = governance.environment_tenant_resolver.resolve_tenant_id(
            application_id=provider_selection.application_id,
            application_environment_id=provider_selection.environment_id,
        )
    except Exception:
        return InactiveHistoryContinuityAuthority()
    if governance.principal.tenant_id != resolved:
        return InactiveHistoryContinuityAuthority()
    return open_offline_external_history_continuity(
        tenant_resolver=governance.environment_tenant_resolver,
        selection=provider_selection,
        authorization_boundary=governance.mutation_authorization_boundary,
        principal=governance.principal,
        issuer_principal_id=issuer,
        bundle=offline_bundle,
    )


def _compose_local(
    governance: ProductionControlPlaneGovernance,
    activation: HistoryContinuityActivation | None,
) -> HistoryContinuityAuthority:
    if activation is None or not history_continuity_activation_is_admitted(activation):
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


def _offline_bundle_is_injected(bundle: object) -> bool:
    if not isinstance(bundle, OfflineProviderBundle) or type(bundle) is not OfflineProviderBundle:
        return False
    if type(bundle.primary) is not InMemoryPrimaryDatabase:
        return False
    if type(bundle.witness) is not InMemoryAppendWitness:
        return False
    if bundle.segment_block_limit < 3 or bundle.recovery_block_bound < 1:
        return False
    if bundle.witness_target.immutability_period_days is not None:
        return False
    if bundle.witness_target.protected_append != "unqualified":
        return False
    if bundle.witness_target.combined_hold != "unqualified":
        return False
    if bundle.primary_target.processing_units is not None:
        return False
    return True
