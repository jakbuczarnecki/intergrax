# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R1 qualification gates (post-remediation current HEAD)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.compat_x._compat_x_persisted_contract_authority import (
    build_contract_authority_durability_evidence,
)
from tests.qualification.compat_x._compat_x_persisted_gates import (
    build_persisted_r2_p0_qualification_report,
    persisted_r2_p0_inventory_gate_passes,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_types import PersistedVersionDisposition


@dataclass(frozen=True, slots=True)
class PersistedR2R1QualificationReport:
    missing_version_semantic_family_count: int
    opaque_carrier_family_count: int
    sqlite_contract_authority_blocker: bool
    r2_p0_inventory_gate: bool


def build_persisted_r2_r1_qualification_report() -> PersistedR2R1QualificationReport:
    missing = sum(
        1
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.version_disposition == PersistedVersionDisposition.MISSING_VERSION
    )
    opaque = sum(
        1
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.version_disposition
        == PersistedVersionDisposition.OPAQUE_CARRIER_NOT_SEMANTIC_PAYLOAD
    )
    authority = build_contract_authority_durability_evidence()
    return PersistedR2R1QualificationReport(
        missing_version_semantic_family_count=missing,
        opaque_carrier_family_count=opaque,
        sqlite_contract_authority_blocker=authority.in_scope_blocker,
        r2_p0_inventory_gate=persisted_r2_p0_inventory_gate_passes(),
    )


def persisted_r2_r1_inventory_gate_passes() -> bool:
    report = build_persisted_r2_r1_qualification_report()
    if not report.r2_p0_inventory_gate:
        return False
    if report.missing_version_semantic_family_count != 0:
        return False
    if report.sqlite_contract_authority_blocker:
        return False
    if report.opaque_carrier_family_count < 1:
        return False
    p0 = build_persisted_r2_p0_qualification_report()
    if p0.metrics.unknown_unclassified_count != 0:
        return False
    return True
