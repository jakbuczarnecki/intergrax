# © Artur Czarnecki. All rights reserved.

"""Bounded cross-tenant-capable surface classification (inventory-derived)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._types import TenantPropagationRole, TenantSurfaceRecord


class R1CrossTenantClass(StrEnum):
    LEGAL_EXPLICIT_CROSS_TENANT = "LEGAL_EXPLICIT_CROSS_TENANT"
    SAME_TENANT_ONLY = "SAME_TENANT_ONLY"
    NON_TENANT = "NON_TENANT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class R1CrossTenantEntry:
    surface_id: str
    classification: R1CrossTenantClass
    evidence: tuple[str, ...]


def _classify_surface(record: TenantSurfaceRecord) -> R1CrossTenantEntry | None:
    sid = record.surface_id
    if record.propagation_role == TenantPropagationRole.LEGAL_CROSS_TENANT_BOUNDARY:
        return R1CrossTenantEntry(
            sid,
            R1CrossTenantClass.LEGAL_EXPLICIT_CROSS_TENANT,
            ("P0 propagation_role=LEGAL_CROSS_TENANT_BOUNDARY",),
        )
    if record.propagation_role == TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE:
        return R1CrossTenantEntry(
            sid,
            R1CrossTenantClass.NON_TENANT,
            ("P0 propagation_role=GLOBAL_NON_TENANT_SURFACE",),
        )
    joined = " ".join(record.evidence_refs)
    if "cross_tenant" in joined.lower() or "cross-tenant" in joined.lower():
        return R1CrossTenantEntry(
            sid,
            R1CrossTenantClass.UNKNOWN,
            ("cross-tenant marker without explicit legal contract evidence",),
        )
    return None


def build_r1_cross_tenant_inventory(
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> tuple[R1CrossTenantEntry, ...]:
    inv = inventory if inventory is not None else get_tenant_x_inventory()
    out: list[R1CrossTenantEntry] = []
    for record in inv:
        entry = _classify_surface(record)
        if entry is not None:
            out.append(entry)
    if not out:
        out.append(
            R1CrossTenantEntry(
                "inventory:zero_explicit_cross_tenant_surfaces",
                R1CrossTenantClass.SAME_TENANT_ONLY,
                ("no LEGAL_CROSS_TENANT_BOUNDARY surfaces in P0 inventory at current HEAD",),
            )
        )
    return tuple(out)


def legal_cross_tenant_mechanism_count(
    inventory: tuple[R1CrossTenantEntry, ...] | None = None,
) -> int:
    inv = inventory if inventory is not None else build_r1_cross_tenant_inventory()
    return sum(1 for e in inv if e.classification == R1CrossTenantClass.LEGAL_EXPLICIT_CROSS_TENANT)


def r1_cross_tenant_unknown_critical(
    inventory: tuple[R1CrossTenantEntry, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else build_r1_cross_tenant_inventory()
    return tuple(e.surface_id for e in inv if e.classification == R1CrossTenantClass.UNKNOWN)
