# © Artur Czarnecki. All rights reserved.

"""EBH-7 requirement traceability SSOT (466 obligations)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tests.qualification.ebh_5._replaceability_matrix import replaceability_matrix_rows
from tests.qualification.ebh_7._constants import EBH7_REQUIREMENTS_TOTAL
from tests.qualification.ebh_7._task_register import EBH7_TASK_REGISTER

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class RequirementRow:
    source_id: str
    task_id: str


@lru_cache(maxsize=1)
def requirement_rows() -> tuple[RequirementRow, ...]:
    rows: list[RequirementRow] = []
    task_ids = {t.task_id for t in EBH7_TASK_REGISTER}

    def add(sid: str, tid: str) -> None:
        assert tid in task_ids, f"{sid} -> {tid}"
        rows.append(RequirementRow(sid, tid))

    for i in range(1, 16):
        tid = "TASK-EBH7-030" if i <= 7 else "TASK-EBH7-051" if i == 8 else "TASK-EBH7-041"
        add(f"EBH7-RQ-{i:03d}", tid)
    for i in range(1, 7):
        add(f"EBH7-ECAP-B{i:02d}", "TASK-EBH7-078")
    for i in range(1, 6):
        add(f"EBH7-ECAP-C{i:02d}", "TASK-EBH7-064")
    for i in range(1, 27):
        add(f"EBH7-ECAP-TH-{i:02d}", "TASK-EBH7-057")
    frz_text = (
        _REPO_ROOT / "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
    ).read_text(encoding="utf-8")
    for line in frz_text.splitlines():
        if not line.startswith("| FRZ-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5 or not parts[1].startswith("FRZ-"):
            continue
        fid = parts[1]
        stage = parts[4]
        if "PROD-Q" in stage or fid.startswith("FRZ-PRD"):
            add(f"EBH7-FRZ-{fid}", "TASK-EBH7-008")
        elif fid.startswith("FRZ-TEN"):
            add(f"EBH7-FRZ-{fid}", "TASK-EBH7-051")
        else:
            add(f"EBH7-FRZ-{fid}", "TASK-EBH7-064")
    for i in range(38):
        add(f"EBH7-SYS-{i:02d}", "TASK-EBH7-033")
    for i in range(1, 17):
        add(f"EBH7-PLT-{i:02d}", "TASK-EBH7-034")
    for i in range(1, 8):
        add(f"EBH7-VBN-ART-{i:02d}", "TASK-EBH7-041")
    for r in replaceability_matrix_rows():
        add(f"EBH7-RPL-{r.concern_id}", "TASK-EBH7-041")
    for i in range(1, 13):
        add(f"EBH7-NPSC-{i:02d}", "TASK-EBH7-023")
    for i in range(1, 27):
        add(f"EBH7-TEN-E2E-{i:02d}", "TASK-EBH7-051")
    for i in range(1, 35):
        add(f"EBH7-XDM-{i:03d}", "TASK-EBH7-057")
    return tuple(rows)


def assert_ebh7_traceability_integrity() -> None:
    rows = requirement_rows()
    assert len(rows) == EBH7_REQUIREMENTS_TOTAL
    sids = [r.source_id for r in rows]
    assert len(sids) == len(set(sids))
