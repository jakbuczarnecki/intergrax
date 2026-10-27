# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P1 qualification selection metadata (deterministic L1/L2 batch only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.dist_scale_x.catalog import DIST_SCALE_X_P1_SELECTION_BATCH


class DistScaleXQualificationLayer(StrEnum):
    L1_CATALOG = "L1"
    L2_EVIDENCE_CONTRACT = "L2"
    L3_PHYSICAL = "L3"
    L4_STRESS = "L4"


@dataclass(frozen=True, slots=True)
class DistScaleXSelectionEntry:
    layer: DistScaleXQualificationLayer
    label: str
    pytest_targets: tuple[str, ...]
    requires_threshold_lock: bool = False
    requires_qual_infra: bool = False


DIST_SCALE_X_P2_L3_L4_PYTEST_TARGETS: tuple[str, ...] = (
    "tests/qualification/dist_scale_x/test_dist_scale_x_p2_e01_lkw_trace.py",
    "tests/qualification/dist_scale_x/test_dist_scale_x_p2_l3_l4_qualification.py",
)

DIST_SCALE_X_P2_SELECTION_BATCH: tuple[str, ...] = (
    *DIST_SCALE_X_P1_SELECTION_BATCH,
    *DIST_SCALE_X_P2_L3_L4_PYTEST_TARGETS,
)

DIST_SCALE_X_P3_L3_L4_PYTEST_TARGETS: tuple[str, ...] = (
    "tests/qualification/dist_scale_x/wl_02_hot_tenant_p3.py",
    "tests/qualification/dist_scale_x/wl_03_hot_key_contention.py",
    "tests/qualification/dist_scale_x/wl_04_http_physical_qualification.py",
    "tests/qualification/dist_scale_x/wl_06_retry_storm.py",
    "tests/qualification/dist_scale_x/wl_06_rabbit_broker_storm.py",
)

DIST_SCALE_X_P3_SELECTION_BATCH: tuple[str, ...] = DIST_SCALE_X_P3_L3_L4_PYTEST_TARGETS

DIST_SCALE_X_P4_L3_L4_PYTEST_TARGETS: tuple[str, ...] = (
    "tests/qualification/dist_scale_x/p4_physical_faults.py",
    "tests/qualification/dist_scale_x/p4_physical_crash.py",
    "tests/qualification/dist_scale_x/p4_physical_restart.py",
    "tests/qualification/dist_scale_x/wl_05_persistence_heavy.py",
)

DIST_SCALE_X_P4_SELECTION_BATCH: tuple[str, ...] = DIST_SCALE_X_P4_L3_L4_PYTEST_TARGETS

DIST_SCALE_X_P1_SELECTION_CATALOG: tuple[DistScaleXSelectionEntry, ...] = (
    DistScaleXSelectionEntry(
        DistScaleXQualificationLayer.L1_CATALOG,
        "P1 catalog integrity and mapping gates",
        ("tests/qualification/dist_scale_x/test_dist_scale_x_l1_catalog_integrity.py",),
    ),
    DistScaleXSelectionEntry(
        DistScaleXQualificationLayer.L2_EVIDENCE_CONTRACT,
        "P1 evidence schema, manifest, and negative contract gates",
        ("tests/qualification/dist_scale_x/test_dist_scale_x_l2_evidence_contract_gates.py",),
    ),
)

DIST_SCALE_X_P2_SELECTION_CATALOG: tuple[DistScaleXSelectionEntry, ...] = (
    DistScaleXSelectionEntry(
        DistScaleXQualificationLayer.L3_PHYSICAL,
        "P2 WL-1/WL-2 physical L3/L4 qualification",
        DIST_SCALE_X_P2_L3_L4_PYTEST_TARGETS,
        requires_qual_infra=True,
    ),
)


def dist_scale_x_p1_pytest_argv() -> list[str]:
    return ["-m", "pytest", "-q", "--tb=short", *DIST_SCALE_X_P1_SELECTION_BATCH]
