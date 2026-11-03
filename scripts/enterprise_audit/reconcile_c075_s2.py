#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Apply C075-S2 WBS reconciliation to post-implementation closure master plan SSOT."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from master_plan_ssot_lib import (
    TECHNICAL_PARTIAL_DISPOSITION,
    _split_md_row,
    build_matrix_e_table,
    extract_section,
    master_plan_path,
    parse_impl_qual_counts,
    parse_matrix_c,
    repo_root,
    validate_master_plan,
)
from wbs_reconciliation_lib import (
    EVID,
    GAP_RC_EVIDENCE,
    GAP_RC_STATUS,
    TES,
    classify_task,
    closure_condition,
    evidence_path_for,
    load_b5_outcomes,
    matrix_e_qual_label,
    remaining_delta,
)


def _collect_only_count(root: Path) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/qualification/ent_audit_x", "--collect-only", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        match = re.search(r"(\d+) tests collected", line)
        if match:
            return int(match.group(1))
    return 0


def _replace_section(text: str, heading: str, new_body: str) -> str:
    idx = text.find(heading)
    if idx < 0:
        raise ValueError(f"missing heading {heading}")
    after = text[idx + len(heading) :]
    next_idx = after.find("\n## ")
    if next_idx < 0:
        return text[:idx] + new_body
    return text[:idx] + new_body + after[next_idx:]


def rebuild_matrix_c(text: str) -> str:
    section = extract_section(text, "## Matrix C — 75 TaskID")
    old_by_id: dict[str, list[str]] = {}
    for line in section.splitlines():
        if not line.startswith("| C"):
            continue
        cells = _split_md_row(line)
        if len(cells) < 10 or not cells[0].startswith("C"):
            continue
        old_by_id[cells[0]] = cells
    lines = [
        "## Matrix C — 75 TaskID",
        "",
        "| TaskID | Batch | Impl | Qual | Evidence @ TES/B5 | Remaining delta | "  # noqa: E501
        "Owner | Deps | Positive test | Negative test | Closure |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tid in sorted(old_by_id.keys(), key=lambda x: int(x[1:])):
        cells = old_by_id[tid]
        batch = cells[1]
        impl, qual = classify_task(tid)
        evid = evidence_path_for(tid)
        delta = remaining_delta(tid)
        owner = cells[5]
        deps = cells[6]
        pos = cells[7]
        neg = cells[8]
        lines.append(
            "| "
            + " | ".join(
                [
                    tid,
                    batch,
                    impl,
                    qual,
                    evid,
                    delta,
                    owner,
                    deps,
                    pos,
                    neg,
                    closure_condition(tid),
                ]
            )
            + " |"
        )
    new_section = "\n".join(lines) + "\n\n## Matrix D"
    return text.split("## Matrix C — 75 TaskID", 1)[0] + new_section + text.split("\n## Matrix D", 1)[1]


def rebuild_matrix_e(root: Path) -> str:
    outcomes = load_b5_outcomes(root)

    def qual_fn(cid: str) -> str:
        return matrix_e_qual_label(cid, outcomes)

    return build_matrix_e_table(root, qual_fn=qual_fn)


def rebuild_gap_rc(text: str) -> str:
    section = text.split("### GAP-RC (33)", 1)[1].split("\n### GAP-B", 1)[0]
    lines = ["### GAP-RC (33)", ""]
    header_done = False
    for line in section.splitlines():
        if not line.startswith("| GAP-RC-"):
            if line.startswith("| GAP-ID") or line.startswith("| ---"):
                if not header_done:
                    lines.append(
                        "| GAP-ID | Source | Root cause | Sev | Reconcile status | Evidence @ SHA | "
                        "TaskID | Pos | Neg | Remaining | Closure |"
                    )
                    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                    header_done = True
                continue
            if line.strip():
                lines.append(line)
            continue
        cells = _split_md_row(line)
        gid = cells[0]
        status = GAP_RC_STATUS.get(gid, "OPEN")
        evid = GAP_RC_EVIDENCE.get(gid, "—")
        if status == "CLOSED" and evid == "—" and len(cells) > 8:
            evid = cells[8]
        task_ids = cells[5] if len(cells) > 5 else ""
        pos = cells[6] if len(cells) > 6 else ""
        neg = cells[7] if len(cells) > 7 else ""
        remaining = cells[4] if len(cells) > 4 else ""
        closure = cells[9] if len(cells) > 9 else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    gid,
                    cells[1],
                    cells[2],
                    cells[3],
                    status,
                    evid,
                    task_ids,
                    pos,
                    neg,
                    remaining,
                    closure,
                ]
            )
            + " |"
        )
    lines.append("")
    return text.split("### GAP-RC (33)", 1)[0] + "\n".join(lines) + "\n### GAP-B" + text.split("\n### GAP-B", 1)[1]


def patch_header(text: str, root: Path, head: str, collect_n: int) -> str:
    impl_counts, qual_counts = parse_impl_qual_counts(parse_matrix_c(text))
    text = re.sub(
        r"\*\*Disposition:\*\* `[^`]+`",
        f"**Disposition:** `{TECHNICAL_PARTIAL_DISPOSITION}`",
        text,
        count=1,
    )
    text = re.sub(
        r"\*\*Audited implementation baseline \(IMPLEMENTATION_SHA\):\*\* `[0-9a-f]{40}`",
        f"**Audited implementation baseline (IMPLEMENTATION_SHA):** `{head}`",
        text,
        count=1,
    )
    text = re.sub(
        r"\*\*Current runtime qualification:\*\* OPEN[^\n]*",
        f"**TEST_EXECUTION_SHA:** `{TES}` (certifying PASS — preserved). "
        f"**EVIDENCE_PUBLICATION_SHA:** `{EVID}`. "
        f"Post-TES changes: docs/scripts/CI only unless new TES pinned.",
        text,
        count=1,
    )
    text = re.sub(
        r"\| FP-04 \| Impl [0-9·]+; Qual [0-9·]+ @ `[^`]+`\.? \|",
        f"| FP-04 | Impl {impl_counts.get('IMPLEMENTED', 0)}·{impl_counts.get('PARTIAL', 0)}·"
        f"{impl_counts.get('NOT_STARTED', 0)}; Qual {qual_counts.get('QUALIFIED', 0)}·"
        f"{qual_counts.get('NOT_QUALIFIED', 0)} @ `{TES}` |",
        text,
        count=1,
    )
    pl07 = (
        f"**{collect_n}** tests collect-only in `tests/qualification/ent_audit_x` @ "
        f"`{head}` (collect-only SHA = documentation HEAD; **not** TEST_EXECUTION_SHA)."
    )
    text = re.sub(
        r"\*\*[0-9]+\*\* tests collect-only in `tests/qualification/ent_audit_x` @ `[0-9a-f]{40}`[^\n]*",
        pl07,
        text,
        count=1,
    )
    text = re.sub(
        r"\| PL-07 collect-only [0-9]+ @ audited baseline \| PASS \|",
        f"| PL-07 collect-only {collect_n} @ `{head}` | PASS |",
        text,
        count=1,
    )
    text = re.sub(
        r"Runtime qualification @ TEST_EXECUTION_SHA: OPEN \(not publication gate\)\.",
        f"Runtime qualification @ `{TES}`: CERTIFYING PASS (B5 `{EVID[:8]}…`); WBS/CI gaps: see disposition.",
        text,
        count=1,
    )
    return text


def apply(root: Path) -> tuple[str, list[str]]:
    path = master_plan_path(root)
    text = path.read_text(encoding="utf-8")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    collect_n = _collect_only_count(root)
    text = rebuild_matrix_c(text)
    text = _replace_section(text, "## Matrix E — 85 semantic oracles", rebuild_matrix_e(root))
    text = rebuild_gap_rc(text)
    text = patch_header(text, root, head, collect_n)
    errors = validate_master_plan(text, root)
    return text, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write master plan if validation passes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    text, errors = apply(root)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        print(f"FAIL ({len(errors)} reconciliation/validation errors)", file=sys.stderr)
        return 1
    print("PASS reconcile preview")
    if args.write and not args.dry_run:
        master_plan_path(root).write_text(text, encoding="utf-8")
        print(f"Wrote {master_plan_path(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
