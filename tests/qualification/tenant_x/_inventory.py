# © Artur Czarnecki. All rights reserved.

"""TENANT-X inventory SSOT — one typed record per discovered tenant surface."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.tenant_x._classifiers import classify_candidate
from tests.qualification.tenant_x._carrier_semantics import (
    TenantCarrierSemanticsReport,
    build_carrier_semantics_report,
)
from tests.qualification.tenant_x._discovery import raw_discovery_candidates, scanned_module_paths
from tests.qualification.tenant_x._types import (
    ClosedWorldDiscoveryReport,
    TenantIdentitySource,
    TenantPropagationRole,
    TenantSurfaceRecord,
)


@lru_cache(maxsize=1)
def tenant_x_inventory() -> tuple[TenantSurfaceRecord, ...]:
    records = [classify_candidate(c) for c in sorted(raw_discovery_candidates(), key=lambda x: x.surface_id)]
    return tuple(records)


def get_tenant_x_inventory() -> tuple[TenantSurfaceRecord, ...]:
    return tenant_x_inventory()


@lru_cache(maxsize=1)
def build_closed_world_discovery_report() -> ClosedWorldDiscoveryReport:
    raw = raw_discovery_candidates()
    classified_ids = inventory_surface_ids()
    unrecorded = frozenset(c.surface_id for c in raw if c.surface_id not in classified_ids)
    semantic_unknown = frozenset(
        r.surface_id
        for r in get_tenant_x_inventory()
        if r.propagation_role == TenantPropagationRole.UNKNOWN
        or r.identity_source == TenantIdentitySource.UNKNOWN
    )
    return ClosedWorldDiscoveryReport(
        module_paths_scanned=scanned_module_paths(),
        raw_candidates=raw,
        unrecorded_candidate_ids=unrecorded,
        semantic_unknown_surface_ids=semantic_unknown,
    )


def inventory_by_propagation_role(role: TenantPropagationRole) -> tuple[TenantSurfaceRecord, ...]:
    return tuple(r for r in get_tenant_x_inventory() if r.propagation_role == role)


def unknown_propagation_role_count() -> int:
    return sum(1 for r in get_tenant_x_inventory() if r.propagation_role == TenantPropagationRole.UNKNOWN)


def unknown_identity_source_count() -> int:
    from tests.qualification.tenant_x._types import TenantIdentitySource

    return sum(1 for r in get_tenant_x_inventory() if r.identity_source == TenantIdentitySource.UNKNOWN)


def inventory_surface_ids() -> frozenset[str]:
    return frozenset(r.surface_id for r in get_tenant_x_inventory())


def class_field_tenant_surface_ids() -> frozenset[str]:
    return frozenset(
        c.surface_id for c in raw_discovery_candidates() if c.discovery_kind == "class.field.tenant"
    )


def carrier_semantics_report() -> TenantCarrierSemanticsReport:
    return build_carrier_semantics_report(
        get_tenant_x_inventory(),
        class_field_surface_ids=class_field_tenant_surface_ids(),
    )
