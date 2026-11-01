# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

from tests.qualification.qual_x._p5_freeze_selection_catalog import (
    QUAL_X_P5_FREEZE_SELECTION_MANIFEST,
    qual_x_p5_entry_pytest_node_ids,
    qual_x_p5_mandatory_collectable_node_ids,
)
from tests.qualification.qual_x._p5_freeze_selection_support import missing_modules_for_manifest

_REPO = Path(__file__).resolve().parents[4]


class QualXSuiteCatalogAdapter:
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
