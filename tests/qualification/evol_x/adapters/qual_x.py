# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.qualification.qual_x._p5_freeze_selection_catalog import (
    QUAL_X_P5_FREEZE_SELECTION_MANIFEST,
    QualXFreezeMandatoryStatus,
    QualXFreezeQualificationDimension,
    qual_x_p5_entry_pytest_node_ids,
    qual_x_p5_mandatory_collectable_node_ids,
)
from tests.qualification.qual_x._p5_freeze_selection_support import missing_modules_for_manifest

_REPO = Path(__file__).resolve().parents[4]


class QualXSuiteCatalogAdapter:
    """QUAL-X P5 catalog adapter: T1 ``MANDATORY_MIN``, T2 ``QUAL_POP_A_COLLECTABLE``, DIM-B exclusion (II-A/II-C)."""

    def collectable_node_ids(self) -> frozenset[str]:
        ids: set[str] = set()
        for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST:
            ids.update(qual_x_p5_entry_pytest_node_ids(entry))
        missing = missing_modules_for_manifest(_REPO, QUAL_X_P5_FREEZE_SELECTION_MANIFEST)
        if missing:
            raise RuntimeError(f"QUAL-X manifest modules missing: {missing}")
        return frozenset(ids)

    def mandatory_node_ids(self) -> frozenset[str]:
        return frozenset(qual_x_p5_mandatory_collectable_node_ids())

    def mandatory_min_node_ids(self) -> frozenset[str]:
        """T1."""
        return self.mandatory_node_ids()

    def pop_a_collectable_node_ids(self) -> frozenset[str]:
        """T2: Dimension-A (non DIM-B) node ids of every entry that is mandatory at HEAD or at QUAL-X final."""
        ids: set[str] = set()
        for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST:
            if entry.qualification_dimension is not QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A:
                continue
            if entry.mandatory_status is QualXFreezeMandatoryStatus.EXCLUDED_FROM_DIMENSION_A:
                continue
            ids.update(qual_x_p5_entry_pytest_node_ids(entry))
        return frozenset(ids)

    def dim_b_excluded_modules(self) -> frozenset[str]:
        out: set[str] = set()
        for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST:
            if entry.mandatory_status is QualXFreezeMandatoryStatus.EXCLUDED_FROM_DIMENSION_A:
                out.update(entry.pytest_modules)
        return frozenset(out)


def verify_node_ids_collectable(node_ids: frozenset[str]) -> frozenset[str]:
    """Return node ids that ``pytest --collect-only`` cannot resolve (one pytest invocation, module-level targets)."""
    if not node_ids:
        return frozenset()
    modules = sorted({n.split("::", 1)[0] for n in node_ids})
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *modules, "--collect-only", "-q", "-m", "not no_ci"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    collected = {line.strip().split("[", 1)[0] for line in proc.stdout.splitlines() if "::" in line}
    base_collected = {c.split("::", 1)[0] + "::" + c.split("::", 1)[1] for c in collected}
    return frozenset(n for n in node_ids if n.split("[", 1)[0] not in base_collected)
