# © Artur Czarnecki. All rights reserved.

"""TENANT-X inventory SSOT — one typed record per discovered tenant surface."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.tenant_x._classifiers import classify_candidate
from tests.qualification.tenant_x._discovery import discover_all_candidates
from tests.qualification.tenant_x._types import TenantPropagationRole, TenantSurfaceRecord


@lru_cache(maxsize=1)
def tenant_x_inventory() -> tuple[TenantSurfaceRecord, ...]:
    records = [classify_candidate(c) for c in sorted(discover_all_candidates(), key=lambda x: x.surface_id)]
    return tuple(records)


TENANT_X_INVENTORY: tuple[TenantSurfaceRecord, ...] = tenant_x_inventory()


def inventory_by_propagation_role(role: TenantPropagationRole) -> tuple[TenantSurfaceRecord, ...]:
    return tuple(r for r in TENANT_X_INVENTORY if r.propagation_role == role)


def unknown_propagation_role_count() -> int:
    return sum(1 for r in TENANT_X_INVENTORY if r.propagation_role == TenantPropagationRole.UNKNOWN)


def inventory_surface_ids() -> frozenset[str]:
    return frozenset(r.surface_id for r in TENANT_X_INVENTORY)
