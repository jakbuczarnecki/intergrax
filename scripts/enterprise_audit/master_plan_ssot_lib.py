# © Artur Czarnecki. All rights reserved.

"""Parse and validate ENT-AUDIT-X post-implementation closure master plan SSOT."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

WAVE_RANK = {
    "W-0": 0,
    "W-A": 1,
    "W-B": 2,
    "W-C": 3,
    "W-D": 4,
    "W-E": 5,
    "W-G": 6,
    "W-H": 7,
}

FORBIDDEN_ORACLE_SUBSTRINGS = (
    "Annex C architecture @ TES",
    "subprocess gate exit 0 @ TES",
    "inject → non-zero",
    "Tenant fail-closed @ TES",
    "VBN @ HEAD replay",
)

EXPECTED_GAP_RC_IDS = {
    "GAP-RC-OD-TRUST",
    "GAP-RC-MECH-READY",
    "GAP-RC-EVIDENCE-TRUST",
    "GAP-RC-L3-FULL",
    "GAP-RC-DRY-RUN",
    "GAP-RC-VBN-31",
    "GAP-RC-CHANGED-IMPACT",
    "GAP-RC-GIT-ANCESTRY",
    "GAP-RC-RESOLVED-BLOCK",
    "GAP-RC-DRIFT-SCHEMA",
    "GAP-RC-JSON-SCHEMA",
    "GAP-RC-ATOMIC-PUB",
    "GAP-RC-SANDBOX",
    "GAP-RC-REDACTION",
    "GAP-RC-SEMANTIC-GATES",
    "GAP-RC-ADV-INJECT",
    "GAP-RC-PR-PATH-SKIP",
    "GAP-RC-NIGHTLY-FREEZE",
    "GAP-RC-LAUNCHER-BAT",
    "GAP-RC-LAUNCHER-SH",
    "GAP-RC-LAUNCHER-CWD",
    "GAP-RC-OUTPUT-COLLISION",
    "GAP-RC-DOSSIER",
    "GAP-RC-SHA-LADDER",
    "GAP-RC-SSOT-DRIFT",
    "GAP-RC-FRZ-MAP",
    "GAP-RC-RUN-STATUS",
    "GAP-RC-REQUAL-DIGEST",
    "GAP-RC-AUDIT-OF-AUDITOR",
    "GAP-RC-EVID-PUBLISH",
    "GAP-RC-CI-PROFILE",
    "GAP-RC-CERTIFY-SPLIT",
    "GAP-RC-CANDIDATE-FSM",
}

EXPECTED_GAP_B_IDS = {
    "BLK-EVOL-B6",
    "BLK-EVOL-FRZ",
    "BLK-EVOL-HANDOFF",
    "BLK-CI-ENFORCE",
    "BLK-ROADMAP-ENT-DEP",
    "BLK-ENT-B6",
}

DOC_ONLY_NEGATIVE_WHITELIST = frozenset(
    {
        "C041",
        "C052",
        "C053",
        "C055",
        "C056",
        "C057",
        "C064",
    }
)

READY_DISPOSITION = "ENT-AUDIT-X — CURRENT-HEAD COMPLETE CLOSURE EXECUTION PLAN / READY FOR AGENT IMPLEMENTATION"

PARTIAL_DISPOSITION = "ENT-AUDIT-X — PLAN PARTIAL / EXACT BLOCKERS"

TASK_ID_RE = re.compile(r"^C0(?:0[1-9]|[1-6][0-9]|7[0-5])$")

FORBIDDEN_SELF_CERT = (
    "B6 PASS",
    "FRZ CLOSED",
    "OD APPROVED",
    "ENT-AUDIT-X CLOSED",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def master_plan_path(root: Path | None = None) -> Path:
    root = root or repo_root()
    return (
        root
        / "docs/project/maintainers/qualification/ent_audit_x"
        / "ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md"
    )


def annex_c_path(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / "docs/project/maintainers/qualification/ent_audit_x/ANNEX_C_CONTROL_MATRIX.md"


def load_manifest() -> dict[str, Any]:
    pkg = resources.files("intergrax.platform.enterprise_audit")
    raw = (pkg / "data" / "enterprise_audit_check_manifest.v1.json").read_text(encoding="utf-8")
    return json.loads(raw)


@dataclass
class MatrixCRow:
    task_id: str
    batch: str
    impl: str
    qual: str
    line: str


@dataclass
class MatrixERow:
    check_id: str
    line: str
    cells: list[str]


@dataclass
class DagRow:
    task_id: str
    deps: list[str]
    batch: str


def _split_md_row(line: str) -> list[str]:
    parts = [p.strip() for p in line.strip().split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def extract_section(text: str, heading: str) -> str:
    pattern = re.compile(rf"^{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"\n## ", text[start:])
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end]


def parse_matrix_c(text: str) -> dict[str, MatrixCRow]:
    section = extract_section(text, "## Matrix C — 75 TaskID")
    rows: dict[str, MatrixCRow] = {}
    for line in section.splitlines():
        if not line.startswith("| C"):
            continue
        cells = _split_md_row(line)
        if len(cells) < 10:
            continue
        task_id = cells[0]
        if not TASK_ID_RE.match(task_id):
            continue
        rows[task_id] = MatrixCRow(
            task_id=task_id,
            batch=cells[1],
            impl=cells[2],
            qual=cells[3],
            line=line,
        )
    return rows


def parse_dag_table(text: str) -> dict[str, DagRow]:
    section = extract_section(text, "### 14.1 Dependencies")
    rows: dict[str, DagRow] = {}
    for line in section.splitlines():
        if not line.startswith("| C"):
            continue
        cells = _split_md_row(line)
        if len(cells) < 3:
            continue
        task_id = cells[0]
        if not TASK_ID_RE.match(task_id):
            continue
        dep_raw = cells[1].strip()
        if dep_raw in ("—", "-", ""):
            deps: list[str] = []
        else:
            deps = [d.strip() for d in dep_raw.split(",") if d.strip()]
        rows[task_id] = DagRow(task_id=task_id, deps=deps, batch=cells[2])
    return rows


def count_dag_table_lines(text: str) -> int:
    section = extract_section(text, "### 14.1 Dependencies")
    total = 0
    for line in section.splitlines():
        if not line.startswith("| C"):
            continue
        cells = _split_md_row(line)
        if cells and TASK_ID_RE.match(cells[0]):
            total += 1
    return total


def parse_matrix_e(text: str) -> dict[str, MatrixERow]:
    section = extract_section(text, "## Matrix E — 85 semantic oracles")
    rows: dict[str, MatrixERow] = {}
    for line in section.splitlines():
        if not line.startswith("| ENT-A-") and not line.startswith("| TEN-") and not line.startswith("| VBN-"):
            continue
        cells = _split_md_row(line)
        if len(cells) < 11:
            continue
        check_id = cells[0]
        rows[check_id] = MatrixERow(check_id=check_id, line=line, cells=cells)
    return rows


def parse_gap_rc_ids(text: str) -> set[str]:
    section = extract_section(text, "### GAP-RC (33)")
    found: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("| GAP-RC-"):
            continue
        cells = _split_md_row(line)
        if cells:
            found.add(cells[0])
    return found


def parse_gap_b_ids(text: str) -> set[str]:
    section = extract_section(text, "### GAP-B (6)")
    found: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("| BLK-"):
            continue
        cells = _split_md_row(line)
        if cells:
            found.add(cells[0])
    return found


def parse_adv_ids(text: str) -> set[str]:
    section = extract_section(text, "## Matrix F — ADV-01..31")
    found: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("| ADV-"):
            continue
        cells = _split_md_row(line)
        if cells and cells[0].startswith("ADV-"):
            found.add(cells[0])
    return found


def parse_matrix_g_domains(text: str) -> set[str]:
    section = extract_section(text, "## Matrix G — D01–D31")
    found: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("| D"):
            continue
        cells = _split_md_row(line)
        if cells and re.match(r"^D\d{2}$", cells[0]):
            found.add(cells[0])
    return found


def parse_frz_aud_rows(text: str) -> int:
    section = extract_section(text, "## Matrix A — FRZ-AUD-01..10")
    count = 0
    for line in section.splitlines():
        if line.startswith("| FRZ-AUD-"):
            count += 1
    return count


def parse_impl_qual_counts(matrix_c: dict[str, MatrixCRow]) -> tuple[dict[str, int], dict[str, int]]:
    impl_counts: dict[str, int] = {}
    qual_counts: dict[str, int] = {}
    for row in matrix_c.values():
        impl_counts[row.impl] = impl_counts.get(row.impl, 0) + 1
        qual_counts[row.qual] = qual_counts.get(row.qual, 0) + 1
    return impl_counts, qual_counts


def topological_sort(dag: dict[str, DagRow]) -> list[str]:
    indeg: dict[str, int] = {tid: 0 for tid in dag}
    adj: dict[str, list[str]] = {tid: [] for tid in dag}
    for tid, row in dag.items():
        for dep in row.deps:
            if dep not in dag:
                raise ValueError(f"orphan prerequisite {dep} for {tid}")
            adj[dep].append(tid)
            indeg[tid] += 1
    queue = sorted([t for t, d in indeg.items() if d == 0])
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in sorted(adj[node]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(dag):
        raise ValueError("cycle detected in task DAG")
    return order


def parse_annex_c_requirements(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = _split_md_row(line)
        if len(cells) < 2:
            continue
        cid = cells[0]
        if not re.match(r"^(ENT-A-|TEN-|VBN-)", cid):
            continue
        if cid in ("Check ID",):
            continue
        req = cells[1]
        if cid.startswith("ENT-A-"):
            frz = cells[4] if len(cells) >= 5 else "FRZ-AUD-02"
            negative = f"adversarial fault inject for {cid} → non-zero exit"
        else:
            frz = cells[9] if len(cells) >= 10 else "FRZ-AUD-02"
            negative = cells[8] if len(cells) >= 9 else f"adversarial inject for {cid}"
        out[cid] = {"requirement": req, "frz": frz, "negative": negative}
    return out


def acceptance_wrapper_nodeid(check_id: str) -> str:
    slug = check_id.lower().replace("-", "_")
    rel = f"tests/qualification/ent_audit_x/acceptance/test_ent_a_{slug}.py"
    func = f"test_acceptance_{slug}_semantic_gate"
    return f"{rel}::{func}"


def matrix_e_task_id(check_id: str) -> str:
    if check_id.startswith("TEN-"):
        return "C035"
    if check_id.startswith("VBN-"):
        return "C036"
    return "C034"


def build_matrix_e_table(root: Path | None = None) -> str:
    root = root or repo_root()
    manifest = load_manifest()
    annex = parse_annex_c_requirements(annex_c_path(root))
    lines = [
        "## Matrix E — 85 semantic oracles",
        "",
        "| CheckID | kind | Entrypoint | Invariant | Semantic oracle @ TES | "
        "Negative mutation | FRZ | GAP-COV | TaskID | Evidence | Qual @ HEAD |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(manifest["checks"], key=lambda r: r["check_id"]):
        cid = row["check_id"]
        meta = annex.get(cid, {})
        req = meta.get("requirement", cid)
        frz = meta.get("frz", "FRZ-AUD-02")
        neg = meta.get("negative", "fault inject → pytest exit != 0")
        kind = row.get("entrypoint_kind", "pytest_nodeid")
        entry = row["entrypoint"]
        wrapper = acceptance_wrapper_nodeid(cid)
        oracle = (
            f"Wrapper `{wrapper}` executes manifest entrypoint via sandbox; "
            f"assert outcome PASS for check_id={cid} @ TES"
        )
        if kind == "pytest_file":
            oracle += f" (shared file — wrapper scopes assertion to {cid})"
        lines.append(
            "| "
            + " | ".join(
                [
                    cid,
                    kind,
                    f"`{entry}`",
                    req,
                    oracle,
                    neg,
                    frz,
                    f"GAP-COV-{cid}",
                    matrix_e_task_id(cid),
                    "manifest outcomes + junit @ TES",
                    "OPEN until certifying FULL @ TEST_EXECUTION_SHA",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _line_has_negated_cert(line: str, phrase: str) -> bool:
    lower = line.lower()
    if phrase.lower() not in lower:
        return False
    return any(x in lower for x in ("no ", "not ", "without ", "≠", "— no", "does not"))


def validate_master_plan(text: str, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    root = root or repo_root()

    if READY_DISPOSITION not in text and PARTIAL_DISPOSITION not in text:
        errors.append("disposition must declare READY or PLAN PARTIAL constant")

    for line in text.splitlines():
        for phrase in FORBIDDEN_SELF_CERT:
            if phrase in line and not _line_has_negated_cert(line, phrase):
                if line.strip().startswith("**Mode:**") or "execution SSOT" in line:
                    continue
                errors.append(f"forbidden self-certification phrase: {phrase!r} in: {line[:80]}")

    matrix_c = parse_matrix_c(text)
    if len(matrix_c) != 75:
        errors.append(f"Matrix C expected 75 TaskID rows, got {len(matrix_c)}")

    dag = parse_dag_table(text)
    if len(dag) != 75:
        errors.append(f"§14.1 expected 75 DAG rows, got {len(dag)}")
    if count_dag_table_lines(text) != len(dag):
        errors.append("§14.1 duplicate TaskID rows detected")

    if set(matrix_c) != set(dag):
        errors.append("Matrix C TaskID set != §14.1 TaskID set")

    for tid, row in matrix_c.items():
        if row.impl == "QUALIFIED":
            errors.append(f"{tid}: Impl column cannot be QUALIFIED")
        if row.qual == "QUALIFIED" and row.impl != "IMPLEMENTED":
            errors.append(f"{tid}: QUALIFIED requires IMPLEMENTED impl status")
        cells = _split_md_row(row.line)
        if len(cells) > 8 and tid not in DOC_ONLY_NEGATIVE_WHITELIST:
            neg = cells[8]
            if neg.strip().upper() == "N/A":
                errors.append(f"{tid}: negative test cannot be N/A")

    gap_rc = parse_gap_rc_ids(text)
    if gap_rc != EXPECTED_GAP_RC_IDS:
        missing = EXPECTED_GAP_RC_IDS - gap_rc
        extra = gap_rc - EXPECTED_GAP_RC_IDS
        if missing:
            errors.append(f"GAP-RC missing ids: {sorted(missing)}")
        if extra:
            errors.append(f"GAP-RC extra ids: {sorted(extra)}")

    gap_b = parse_gap_b_ids(text)
    if gap_b != EXPECTED_GAP_B_IDS:
        errors.append(f"GAP-B set mismatch: got {sorted(gap_b)}")

    if len(parse_adv_ids(text)) != 31:
        errors.append("Matrix F expected 31 ADV rows")

    if len(parse_matrix_g_domains(text)) != 31:
        errors.append("Matrix G expected 31 domain rows")

    if parse_frz_aud_rows(text) != 10:
        errors.append("Matrix A expected 10 FRZ-AUD rows")

    matrix_e = parse_matrix_e(text)
    manifest_ids = {r["check_id"] for r in load_manifest()["checks"]}
    if set(matrix_e) != manifest_ids:
        errors.append("Matrix E CheckID set != manifest check_id set")

    for cid, row in matrix_e.items():
        for bad in FORBIDDEN_ORACLE_SUBSTRINGS:
            if bad in row.line:
                errors.append(f"Matrix E {cid} contains forbidden generic oracle: {bad}")
        if len(row.cells[3]) < 8:
            errors.append(f"Matrix E {cid} invariant text too short")
        oracle = row.cells[4]
        if "Wrapper `" not in oracle and "wrapper `" not in oracle:
            errors.append(f"Matrix E {cid} must use acceptance wrapper semantic oracle")

    try:
        order = topological_sort(dag)
    except ValueError as exc:
        errors.append(str(exc))
        order = []

    for tid, row in dag.items():
        batch = row.batch
        if batch not in WAVE_RANK:
            errors.append(f"{tid}: unknown batch {batch}")
            continue
        rank = WAVE_RANK[batch]
        for dep in row.deps:
            dep_row = dag.get(dep)
            if not dep_row:
                errors.append(f"{tid}: orphan dep {dep}")
                continue
            dep_rank = WAVE_RANK.get(dep_row.batch, -1)
            if dep_rank > rank:
                errors.append(f"backward cross-wave: {tid} ({batch}) depends on {dep} ({dep_row.batch})")

    c065 = dag.get("C065")
    if c065 and c065.batch in ("W-A", "W-B"):
        errors.append("C065 batch must be W-D or later (after C026)")

    if dag.get("C072") and dag["C072"].batch == "W-A":
        errors.append("C072 must not remain in W-A")

    if dag.get("C075") and dag["C075"].batch == "W-0":
        errors.append("C075 final reconcile must be W-G (C075-S1 is W-0 substep only)")

    if not re.search(
        r"PL-07[\s\S]{0,200}\*\*282\*\* tests collect-only",
        text,
    ):
        errors.append("PL-07 must declare **282** tests collect-only @ audited implementation baseline")

    pl08 = extract_section(text, "### PL-08 — ECAP (canonical)")
    if pl08 and "B5" in pl08 and "B6" in pl08 and pl08.index("B5") > pl08.index("B6"):
        errors.append("PL-08 ECAP must order B5 before B6")

    if "C075-S1" not in text or "C075-S2" not in text:
        errors.append("C075-S1 and C075-S2 substeps must be documented")

    impl_counts, qual_counts = parse_impl_qual_counts(matrix_c)
    fp04 = re.search(r"FP-04 \| Impl ([0-9·]+)", text)
    if fp04:
        declared = fp04.group(1).strip()
        actual = (
            f"{impl_counts.get('IMPLEMENTED', 0)}·{impl_counts.get('PARTIAL', 0)}·{impl_counts.get('NOT_STARTED', 0)}"
        )
        if declared != actual:
            errors.append(f"FP-04 impl balance {declared} != Matrix C {actual}")

    if qual_counts.get("QUALIFIED", 0) + qual_counts.get("NOT_QUALIFIED", 0) != 75:
        errors.append("Matrix C qual statuses must sum to 75")

    sec18 = extract_section(text, "## §18 Mechanical publication self-test")
    if "**Overall publication gate:** PASS" in sec18 and "Validator exit code: 0" not in sec18:
        errors.append("§18 Overall PASS requires Validator exit code: 0 row")

    if READY_DISPOSITION in text and errors:
        errors.append("READY disposition incompatible with validation errors")

    if order:
        if order.index("C066") < order.index("C051"):
            errors.append("C066 must come after C051 in topological order")

    return errors
