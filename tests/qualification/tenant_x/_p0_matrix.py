# © Artur Czarnecki. All rights reserved.

"""Aggregate TENANT-X-P0 inventory metrics."""

from __future__ import annotations

from tests.qualification.tenant_x._discovery import production_files_scanned_count
from tests.qualification.tenant_x._inventory import get_tenant_x_inventory, unknown_propagation_role_count
from tests.qualification.tenant_x._propagation import propagation_edge_count
from tests.qualification.tenant_x._types import (
    TenantBlockerClass,
    TenantDomain,
    TenantIdentitySource,
    TenantPropagationRole,
    TenantP0Matrix,
)

_STATE_DOMAINS = frozenset(
    {
        TenantDomain.DURABLE_STATE,
        TenantDomain.RUNTIME_STATE,
        TenantDomain.CACHE,
        TenantDomain.MEMORY,
        TenantDomain.CHECKPOINT,
    }
)

_PROVIDER_DOMAINS = frozenset(
    {
        TenantDomain.PROVIDER_INVOCATION,
        TenantDomain.INTEGRATION_RESOLUTION,
        TenantDomain.CONFIGURATION_PROFILES,
    }
)

_RECOVERY_DOMAINS = frozenset(
    {
        TenantDomain.RECOVERY,
        TenantDomain.RESUME,
        TenantDomain.REPLAY,
        TenantDomain.RETRY,
        TenantDomain.CHECKPOINT,
        TenantDomain.BACKUP_RESTORE,
    }
)

_EXTENSION_ROLES = frozenset(
    {
        TenantPropagationRole.TENANT_SCOPED_PROVIDER_SELECTION,
        TenantPropagationRole.TENANT_SCOPED_CONFIGURATION,
    }
)


def build_p0_matrix() -> TenantP0Matrix:
    inventory = get_tenant_x_inventory()
    canonical = sum(
        1 for r in inventory if "canonical_anchor_profile" in "".join(r.evidence_refs)
    )
    carriers = sum(1 for r in inventory if r.propagation_role == TenantPropagationRole.IDENTITY_CARRIER)
    derived = sum(
        1
        for r in inventory
        if r.propagation_role
        in {
            TenantPropagationRole.IDENTITY_DERIVATION,
            TenantPropagationRole.IDENTITY_CARRIER,
        }
        and r.identity_source != TenantIdentitySource.CANONICAL_CONTRACT
    )
    ambient = sum(
        1 for r in inventory if r.identity_source == TenantIdentitySource.AMBIENT_DEFAULT_LITERAL
    )
    unknown_identity = sum(1 for r in inventory if r.identity_source == TenantIdentitySource.UNKNOWN)
    global_non = sum(
        1 for r in inventory if r.propagation_role == TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE
    )
    legal_cross = sum(
        1 for r in inventory if r.propagation_role == TenantPropagationRole.LEGAL_CROSS_TENANT_BOUNDARY
    )
    blockers = sum(1 for r in inventory if r.blocker_class == TenantBlockerClass.IN_SCOPE_BLOCKER)

    return TenantP0Matrix(
        production_files_scanned=production_files_scanned_count(),
        tenant_related_candidates=len(inventory),
        tenant_required_surfaces=sum(1 for r in inventory if r.tenant_required),
        global_non_tenant_surfaces=global_non,
        canonical_identity_sources=canonical,
        tenant_carriers=carriers,
        derived_identity_carriers=derived,
        ambient_default_identity_sources=ambient,
        unknown_identity_sources=unknown_identity,
        propagation_edges=propagation_edge_count(),
        state_persistence_surfaces=sum(1 for r in inventory if r.domain in _STATE_DOMAINS),
        provider_config_surfaces=sum(1 for r in inventory if r.domain in _PROVIDER_DOMAINS),
        credential_secret_surfaces=sum(
            1 for r in inventory if r.domain == TenantDomain.CREDENTIALS_SECRETS
        ),
        evidence_trace_surfaces=sum(
            1 for r in inventory if r.domain in {TenantDomain.EVIDENCE, TenantDomain.TRACE}
        ),
        recovery_surfaces=sum(1 for r in inventory if r.domain in _RECOVERY_DOMAINS),
        extension_surfaces=sum(1 for r in inventory if r.propagation_role in _EXTENSION_ROLES),
        legal_cross_tenant_boundaries=legal_cross,
        implicit_default_candidates=ambient,
        unknown_propagation_roles=unknown_propagation_role_count(),
        in_scope_blockers=blockers,
        platform_escape_count=blockers + ambient,
    )
