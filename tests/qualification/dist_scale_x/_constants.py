# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X mechanical constants (P1)."""

from __future__ import annotations

from typing import Final

DIST_SCALE_X_P1_START_HEAD: Final[str] = "ce5700cb8895109f1735c940a3191271a885ac88"

DIST_SCALE_X_P2_EXECUTION_BASELINE_HEAD: Final[str] = "3cbd8bea0864a785c10cd39260345834a6e11dca"

DIST_SCALE_X_ROADMAP_SECTION: Final[str] = "PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md §3.0.6"

DIST_SCALE_X_PROOF_ROOT_REL: Final[str] = ".tmp/proof/dist-scale-x"

DIST_SCALE_X_MANIFEST_FILENAME: Final[str] = "dist_scale_x_evidence_manifest.json"

DIST_SCALE_X_MANIFEST_SCHEMA_VERSION: Final[int] = 1

MANDATORY_BACKEND_IDS: Final[tuple[str, ...]] = (
    "MB-1",
    "MB-2",
    "MB-3",
    "MB-4",
    "MB-5",
)

DIMENSION_IDS: Final[tuple[str, ...]] = tuple(chr(ord("A") + i) for i in range(14))

WORKLOAD_IDS: Final[tuple[str, ...]] = tuple(f"WL-{i}" for i in range(1, 8))

EXIT_CRITERION_IDS: Final[tuple[int, ...]] = tuple(range(1, 17))
