#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Export W0 C075 matrix JSON from master plan SSOT + wbs_reconciliation_lib."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from master_plan_ssot_lib import _split_md_row, master_plan_path, parse_matrix_c, repo_root
from wbs_reconciliation_lib import (
    CI_BLOCKED,
    GAP_RC_EVIDENCE,
    GAP_RC_STATUS,
    IMPLEMENTED_PARTIAL_DEFAULT,
    OPERATOR_OR_EXTERNAL,
    POST_TES_DELTA,
    QUALIFIED_TES,
    TES,
    classify_task,
    closure_condition,
    evidence_path_for,
    remaining_delta,
)


def tier_class(task_id: str) -> str:
    if task_id in QUALIFIED_TES:
        return "T1"
    if task_id in OPERATOR_OR_EXTERNAL:
        return "T5"
    if task_id in CI_BLOCKED:
        return "T2_CI"
    if task_id in IMPLEMENTED_PARTIAL_DEFAULT:
        return "T2_NEG"
    if task_id in POST_TES_DELTA or task_id in {"C054", "C066"}:
        return "T4"
    return "T3"


def git_head(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def build_matrix_rows(text: str) -> list[dict]:
    matrix_c = parse_matrix_c(text)
    rows: list[dict] = []
    for tid in sorted(matrix_c.keys(), key=lambda x: int(x[1:])):
        cells = _split_md_row(matrix_c[tid].line)
        impl, qual = classify_task(tid)
        rows.append(
            {
                "task_id": tid,
                "wave_batch": matrix_c[tid].batch,
                "impl_status": impl,
                "qual_status": qual,
                "evidence_path": evidence_path_for(tid),
                "remaining_delta": remaining_delta(tid),
                "positive_test": cells[7] if len(cells) > 7 else "",
                "negative_test": cells[8] if len(cells) > 8 else "",
                "closure_condition": closure_condition(tid),
                "tier_class": tier_class(tid),
                "deps": cells[6] if len(cells) > 6 else "",
            }
        )
    return rows


def gap_rc_rows() -> list[dict]:
    out: list[dict] = []
    for gap_id in sorted(GAP_RC_STATUS.keys()):
        out.append(
            {
                "gap_id": gap_id,
                "reconcile_status": GAP_RC_STATUS[gap_id],
                "evidence": GAP_RC_EVIDENCE.get(gap_id, ""),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo_root() / ".tmp/session/ent-audit-x-closure/W0_C075_matrix.v1.json",
    )
    args = parser.parse_args()
    root = repo_root()
    mp = master_plan_path()
    text = mp.read_text(encoding="utf-8")
    payload = {
        "schema": "ent_audit_x.w0_c075_matrix.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "head_sha": git_head(root),
        "test_execution_sha": TES,
        "task_rows": build_matrix_rows(text),
        "task_count": 75,
        "gap_rc": gap_rc_rows(),
        "gap_rc_count": len(GAP_RC_STATUS),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(payload['task_rows'])} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
