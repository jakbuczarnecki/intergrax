#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501

"""Apply current-HEAD SSOT repair to post-implementation closure master plan."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from master_plan_ssot_lib import (
    READY_DISPOSITION,
    build_matrix_e_table,
    master_plan_path,
    parse_dag_table,
    parse_impl_qual_counts,
    parse_matrix_c,
    topological_sort,
    validate_master_plan,
)

AUDITED_IMPLEMENTATION_SHA = "d4974efc255bf6ff6ca96e00fb084293ea685d66"
CURRENT_DOCUMENTATION_HEAD = "c6d600df56f3fedb85bcdce12ba508ea8274ccd4"
PRIOR_PLAN_PUBLICATION_SHA = "87e185dd3bcce45f107149b4b3691c3ede614c8a"
HISTORICAL_TEST_EXECUTION_SHA = "0cee41a88274a53d14bb3558752d55bdde0d0b01"


def _replace_section(text: str, heading: str, new_body: str) -> str:
    idx = text.find(heading)
    if idx < 0:
        raise ValueError(f"missing heading {heading}")
    after = text[idx + len(heading) :]
    next_idx = after.find("\n## ")
    if next_idx < 0:
        return text[:idx] + new_body
    return text[:idx] + new_body + after[next_idx:]


def _patch_matrix_c_line(
    line: str, batch: str | None = None, impl: str | None = None, qual: str | None = None, delta: str | None = None
) -> str:
    cells = [c.strip() for c in line.strip().split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    if batch is not None:
        cells[1] = batch
    if impl is not None:
        cells[2] = impl
    if qual is not None:
        cells[3] = qual
    if delta is not None:
        cells[4] = delta
    return "| " + " | ".join(cells) + " |"


def patch_matrix_c(text: str) -> str:
    section = text.split("## Matrix C — 75 TaskID", 1)[1].split("\n## Matrix D", 1)[0]
    lines = section.splitlines()
    out_lines: list[str] = []
    patches: dict[str, dict] = {
        "C015": {"impl": "PARTIAL", "delta": "Extend redaction patterns (v2); core `_redact` @ audited SHA"},
        "C016": {"impl": "PARTIAL", "delta": "`run_status` completed/interrupted wired; interrupt path qual @ TES"},
        "C065": {
            "batch": "W-D",
            "impl": "PARTIAL",
            "delta": "Wire promote + operator_records; verify closed bundle (after C026)",
        },
        "C072": {
            "batch": "W-B",
            "impl": "PARTIAL",
            "delta": "Impact map JSON + `select_checks_for_changed_since`; qual mapping tests @ TES",
        },
        "C073": {"impl": "PARTIAL", "delta": "Catalog profile versions; strict verify @ TES"},
        "C075": {
            "batch": "W-G",
            "impl": "PARTIAL",
            "delta": "C075-S2 final reconcile (S1 @ W-0 done in this repair wave)",
        },
    }
    for line in lines:
        if line.startswith("| C"):
            tid = line.split("|")[1].strip()
            if tid in patches:
                p = patches[tid]
                line = _patch_matrix_c_line(line, **p)
        out_lines.append(line)
    new_section = "## Matrix C — 75 TaskID\n" + "\n".join(out_lines) + "\n## Matrix D"
    return text.split("## Matrix C — 75 TaskID", 1)[0] + new_section + text.split("\n## Matrix D", 1)[1]


def patch_dag_batches(text: str) -> str:
    replacements = {
        "| C065 | C003,C026 | W-A |": "| C065 | C003,C026 | W-D |",
        "| C072 | C012,C024 | W-A |": "| C072 | C012,C024 | W-B |",
        "| C075 | C041,C048,C055 | W-0 |": "| C075 | C041,C048,C055 | W-G |",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def rebuild_topo_section(text: str) -> str:
    dag = parse_dag_table(text)
    order = topological_sort(dag)
    chain = " → ".join(order)
    pattern = r"(\*\*Acyclic:\*\* YES · \*\*Nodes:\*\* 75\n\n)(C001 →.*?)(\n\n\*\*Wave F infra)"
    repl = rf"\1{chain}\3"
    return re.sub(pattern, repl, text, count=1, flags=re.DOTALL)


def build_header() -> str:
    return f"""# ENT-AUDIT-X — post-implementation final closure master plan

**Disposition:** `{READY_DISPOSITION}`

**CURRENT_DOCUMENTATION_HEAD:** `{CURRENT_DOCUMENTATION_HEAD}`

**Audited implementation baseline (IMPLEMENTATION_SHA):** `{AUDITED_IMPLEMENTATION_SHA}`

**Prior PLAN_PUBLICATION_SHA:** `{PRIOR_PLAN_PUBLICATION_SHA}`

**Historical TEST_EXECUTION_SHA (not current qualification):** `{HISTORICAL_TEST_EXECUTION_SHA}` — log @ older wave; **do not** treat as proof @ `{AUDITED_IMPLEMENTATION_SHA}`.

**Current runtime qualification:** OPEN — pin new **TEST_EXECUTION_SHA** only after certifying FULL/L3 @ implementation baseline.

**Mode:** execution SSOT — no B6 PASS / FRZ CLOSED / ARCH-FREEZE / ENT stage CLOSED / OD APPROVED by agent.

Canonical scope: [corrective plan](./ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md) C001–C064; C065–C075 Appendix B.

### W-0 substeps (documentation)

| Substep | Wave | Scope | Deps |
| --- | --- | --- | --- |
| C075-S1 | W-0 | Current-HEAD inventory: git ancestry, PL-07 collect-only, ladder append-only row, disposition | — |
| C075-S2 | W-G | Final SSOT reconcile: Matrix C/E, §14 DAG, §18 validator PASS, annex sync | C041,C048,C055 |

"""


def build_fp_section(impl_counts: dict[str, int], qual_counts: dict[str, int]) -> str:
    impl_line = (
        f"{impl_counts.get('IMPLEMENTED', 0)}·{impl_counts.get('PARTIAL', 0)}·{impl_counts.get('NOT_STARTED', 0)}"
    )
    qual_line = f"{qual_counts.get('QUALIFIED', 0)}·{qual_counts.get('NOT_QUALIFIED', 0)}"
    return f"""### FP-01..FP-10

| FP | Resolution @ audited implementation baseline |
| --- | --- |
| FP-01 | Single execution SSOT; waves W-0..W-G + substeps. |
| FP-02 | C051+C074 in W-D before Wave F qual. |
| FP-03 | Matrix C 75/75 with owner, deps, pos/neg tests. |
| FP-04 | Impl {impl_line}; Qual {qual_line} @ `{AUDITED_IMPLEMENTATION_SHA}`. |
| FP-05 | §14 dependency table + topological order (wave-safe). |
| FP-06 | §13 agent commit+push; operator OD; admin OD-ENT-02; B6 external. |
| FP-07 | `development` protection HTTP 404 @ documentation HEAD. |
| FP-08 | Matrices E/F/G/D/A complete; Matrix E from manifest+Annex C. |
| FP-09 | C066 after TES artifacts; C051/C074 infra pre Wave F. |
| FP-10 | §18 PASS only when `validate_post_implementation_master_plan.py` exit 0. |

### PL-08 — ECAP (canonical)

Implementation waves (W-A..W-E code) → agent **commit+push** → clean checkout → pin **TEST_EXECUTION_SHA** → L1/L2 → real **85/85** semantic gates (zero skip) → certifying **FULL** → **ADV31** @ TES → first **FREEZE candidate** → **EVIDENCE_PUBLICATION_SHA (B5)** → independent **Protocol 03 B6** → operator **OD-ENT-01** → **ACCEPTED_BASELINE_SHA** → **normal FREEZE** → **OD-ENT-08** / **FRZ-AUD-01..10** → roadmap §5 → **ARCH-FREEZE** handoff.

### PL-07

**282** tests collect-only in `tests/qualification/ent_audit_x` @ `{AUDITED_IMPLEMENTATION_SHA}` (collect-only SHA = documentation HEAD; **not** TEST_EXECUTION_SHA).

"""


def build_section_9_13() -> str:
    return """## §9 Qualification and Wave F

1. Complete remaining W-A..W-E implementation deltas @ audited baseline.
2. Commit/push per wave; no production changes in SSOT-only repair sessions.
3. Clean checkout @ pin candidate; operator selects **TEST_EXECUTION_SHA**.
4. **L1/L2** pytest subsets (`ent_l1`, `ent_l2`, markers per C040).
5. **C051/C074** orchestrator must exist before final Wave F qual.
6. Real **85/85** semantic execution (Matrix E wrappers + manifest outcomes).
7. Certifying **FULL** (0 skipped, exit 0, FULL coverage label).
8. **ADV31** fault-injection suite @ TES.
9. First **FREEZE candidate** (no accepted baseline without OD).
10. **B5** evidence publication tree @ descendant of TES.
11. Independent **B6** (agent prepares dossier only).
12. **OD-ENT-01** → accepted baseline → normal FREEZE → **OD-ENT-08**.

## §13 Agent instruction

Execute **§14.2 topological order** within wave constraints (no backward cross-wave edges). **W-0:** C075-S1 inventory only. **W-A..W-E:** implementation deltas from Matrix C; **git commit and push** each wave. Pytest logs under `.tmp/session/ent-audit-x-*`. Before Wave F: **C031,C032,C051,C074** must be IMPLEMENTED. **C066** readiness script runs only after TES artifacts exist (fail-closed). **Operator only:** OD-ENT-00, OD-ENT-01, OD-ENT-02, OD-ENT-08. **GitHub admin:** branch protection (BLK-CI-ENFORCE). **Independent auditor:** Protocol 03 B6 (BLK-ENT-B6). Agent does **not** declare FRZ/B6/CLOSED/OD APPROVED.

"""


def build_section_18(errors: list[str]) -> str:
    if errors:
        gate = "FAIL"
        overall = "FAIL"
        validator_row = f"Validator exit code: 1 ({len(errors)} errors)"
    else:
        gate = "PASS"
        overall = "PASS"
        validator_row = "Validator exit code: 0"
    rows = [
        ("Matrix C 75 rows", gate),
        ("No backward cross-wave edges", gate),
        ("Matrix E 85 non-generic oracles", gate),
        ("GAP-RC 33 set equality", gate),
        ("GAP-B 6 external blockers listed", gate),
        ("ADV 31 / Matrix G 31 / FRZ-AUD 10", gate),
        ("FP-04 matches Matrix C impl counts", gate),
        ("C075-S1/S2 documented", gate),
        ("PL-07 collect-only 282 @ audited baseline", gate),
        ("ECAP B5 before B6 in PL-08", gate),
        (validator_row, gate),
    ]
    lines = ["## §18 Mechanical publication self-test", "", "| Check | Result |", "| --- | --- |"]
    for name, result in rows:
        lines.append(f"| {name} | {result} |")
    lines.append("")
    lines.append(f"**Overall publication gate:** {overall}")
    lines.append("")
    lines.append("Runtime qualification @ TEST_EXECUTION_SHA: OPEN (not publication gate).")
    lines.append("")
    return "\n".join(lines)


def patch_gap_b(text: str) -> str:
    new = """### GAP-B (6)

| GAP-B-ID | Status | Owner | Required evidence | Closure |
| --- | --- | --- | --- | --- |
| BLK-EVOL-B6 | EXTERNAL | Independent auditor + operator | EVOL Protocol 03 PASS @ EVOL TES/FINAL; ladder B6 SHA | Before ENT stage CLOSED (OD-ENT-00) |
| BLK-EVOL-FRZ | EXTERNAL | Operator | FRZ-EVOL-01..10 promoted after EVOL B6 | ARCH-FREEZE EVOL criteria |
| BLK-EVOL-HANDOFF | EXTERNAL | Operator | W7-06 handoff pin + EVOL formal closure SHA | ENT EVOL digest sync |
| BLK-CI-ENFORCE | EXTERNAL | OD-ENT-02 + GitHub admin | Branch protection + required checks @ TES | FRZ-AUD-10 |
| BLK-ROADMAP-ENT-DEP | EXTERNAL | Operator OD-ENT-00 | Written decision: EVOL-X formally CLOSED or documented exception | Roadmap §5 ENT-AUDIT-X row |
| BLK-ENT-B6 | EXTERNAL | Independent Protocol 03 | B5 bundle + 85/85 @ TES + ADV31; dossier PASS | OD-ENT-01 / FRZ-AUD PASS |

"""
    return _replace_section(text, "### GAP-B (6)", new)


def apply_repair(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    body_start = text.find("## Matrix C — 75 TaskID")
    if body_start < 0:
        raise ValueError("Matrix C not found")
    body = text[body_start:]
    text = build_header() + body
    text = patch_matrix_c(text)
    text = patch_dag_batches(text)
    text = rebuild_topo_section(text)
    text = _replace_section(text, "## Matrix E — 85 semantic oracles", build_matrix_e_table())
    text = patch_gap_b(text)
    sec9_idx = text.find("## §9 Wave F")
    if sec9_idx < 0:
        sec9_idx = text.find("## §9 Qualification")
    sec18_end = text.find("## Appendix B")
    if sec9_idx >= 0 and sec18_end >= 0:
        text = text[:sec9_idx] + build_section_9_13() + text[sec18_end:]
    matrix_c = parse_matrix_c(text)
    impl_counts, qual_counts = parse_impl_qual_counts(matrix_c)
    fp_block = build_fp_section(impl_counts, qual_counts)
    marker = "### W-0 substeps (documentation)"
    w0_end = text.find("## Matrix C — 75 TaskID")
    if marker in text and w0_end > 0:
        text = text[:w0_end] + fp_block + text[w0_end:]
    errors = validate_master_plan(text)
    sec18 = build_section_18(errors)
    if "## §18 Mechanical publication self-test" in text:
        text = _replace_section(text, "## §18 Mechanical publication self-test", sec18)
    else:
        appendix_idx = text.find("## Appendix B")
        if appendix_idx < 0:
            text = text + "\n" + sec18
        else:
            text = text[:appendix_idx] + sec18 + "\n" + text[appendix_idx:]
    if errors:
        text = text.replace(READY_DISPOSITION, "ENT-AUDIT-X — PLAN PARTIAL / EXACT BLOCKERS")
    else:
        partial = "ENT-AUDIT-X — PLAN PARTIAL / EXACT BLOCKERS"
        if partial in text:
            text = text.replace(partial, READY_DISPOSITION)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    path = master_plan_path()
    apply_repair(path)
    text = path.read_text(encoding="utf-8")
    errors = validate_master_plan(text)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("SSOT repair applied; validator PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
