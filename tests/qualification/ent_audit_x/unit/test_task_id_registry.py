from __future__ import annotations

import re
from pathlib import Path

_TASK_ROW = re.compile(r"\|\s*ENT-T-(C\d{3})\s*\|")
_APPENDIX_B = re.compile(r"\|\s*(C0(?:6[5-9]|7[0-5]))\s*\|")


def _task_ids_from_corrective(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        match = _TASK_ROW.search(line)
        if not match:
            continue
        task_id = match.group(1)
        num = int(task_id[1:])
        if 1 <= num <= 64:
            rows[task_id] = line
    return rows


def _task_ids_from_closure_appendix_b(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        match = _APPENDIX_B.search(line)
        if match:
            rows[match.group(1)] = line
    return rows


def test_seventy_five_unique_task_ids_and_canonical_c055_c057_c058() -> None:
    root = Path(__file__).resolve().parents[4]
    qual = root / "docs/project/maintainers/qualification/ent_audit_x"
    corrective = (qual / "ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md").read_text(
        encoding="utf-8"
    )
    closure = (qual / "ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md").read_text(encoding="utf-8")

    c001_c064 = _task_ids_from_corrective(corrective)
    assert len(c001_c064) == 64, f"expected 64 corrective rows, got {len(c001_c064)}"
    netnew = _task_ids_from_closure_appendix_b(closure)
    assert len(netnew) == 11, f"expected C065..C075 in Appendix B, got {len(netnew)}"

    all_ids = set(c001_c064) | set(netnew)
    assert len(all_ids) == 75

    assert "Annex F" in c001_c064["C055"]
    assert "OD-ENT matrix" in c001_c064["C057"]
    assert "b6-handoff" in c001_c064["C058"].lower()


def test_closure_master_plan_exists() -> None:
    root = Path(__file__).resolve().parents[4]
    qual = root / "docs/project/maintainers/qualification/ent_audit_x"
    path = qual / "ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md"
    assert path.is_file()
