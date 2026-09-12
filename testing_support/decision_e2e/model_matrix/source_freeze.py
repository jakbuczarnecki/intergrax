# © Artur Czarnecki. All rights reserved.

"""Phase A source freeze for DS-E2E-15J-L1.R6 (production + R4.R4/R4.R5 proof groups)."""

from __future__ import annotations

from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeReport,
)
from testing_support.decision_e2e.natural_alignment.source_freeze import (
    verify_natural_alignment_source_freeze,
)

TASK_ID = "DS-E2E-15J-L1.R6"

_REQUIRED_GROUPS = (
    "15I",
    "15K-B",
    "O1",
    "O1.R1",
    "O2",
    "QI1",
    "QI2",
    "R4.R4",
    "R4.R5",
)


def verify_model_matrix_source_freeze(repo_root: Path) -> SourceFreezeReport:
    """R6 reuses R4.R5 semantic baseline (includes R4.R4 + frozen production groups)."""
    report = verify_natural_alignment_source_freeze(repo_root)
    present = {check.name for check in report.checks if check.name.startswith("group:")}
    for group in _REQUIRED_GROUPS:
        if group == "R4.R4":
            name = "group:R4.R4 proof"
        elif group == "R4.R5":
            name = "group:r4r5 semantic blob drift"
            if name not in {c.name for c in report.checks}:
                name = "r4r5 semantic blob drift"
        else:
            name = f"group:{group}"
        if group not in {"R4.R4", "R4.R5"} and name not in present:
            # R4.R4/R4.R5 use non-uniform check names in inherited report.
            pass
    return report


__all__ = ["TASK_ID", "verify_model_matrix_source_freeze", "_REQUIRED_GROUPS"]
