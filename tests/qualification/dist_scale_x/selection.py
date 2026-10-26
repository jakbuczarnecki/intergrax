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


def dist_scale_x_p1_pytest_argv() -> list[str]:
    return ["-m", "pytest", "-q", "--tb=short", *DIST_SCALE_X_P1_SELECTION_BATCH]
