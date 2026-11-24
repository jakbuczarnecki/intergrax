#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Build canonical T-AF2-RW-PLAN artefacts from FRZ matrix + ticket plans (W2-RECOVERY-01)."""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MATRIX = _REPO / "docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md"
_PLANS = _REPO / "governance/tickets/plans"
_RW_PLAN_MD = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md"
_RW_PLAN_JSON = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json"

APPENDIX_A_MARK = "## Appendix A."
APPENDIX_END_MARKERS = ("## Appendix B", "## Appendix C", "## Appendix D")
_TICKET_RE = re.compile(r"T-IGX-\d{3}-[A-Z0-9]+")
_RW_TICKET_RE = re.compile(r"RW ticket (T-IGX-\d{3}-[A-Z0-9]+)")
_NT_RE = re.compile(r"(NT-[A-Z0-9]+-\d+|NT-[A-Z]{3}-\d+|NT-ROW-\d+)")
_FRZ_ROW_RE = re.compile(r"^\|\s*`(FRZ-[A-Z]+-\d+)`\s*\|")

AUDIT_FLAG_CONFLICTS = [
    {"finding_id": "A4-005", "owner_ticket": "T-IGX-051-H", "disposition": "T-AF2-DEPS-BATCH; PT-WF overlap"},
    {
        "finding_id": "A4-018",
        "owner_ticket": "T-IGX-051-D",
        "disposition": "T-AF2-RW-RUNTIME; shares PT-WF / FRZ-CTR rows",
    },
    {"finding_id": "A6-005", "owner_ticket": "T-IGX-043-I", "disposition": "PT-WD; ENT manifest engine scope"},
    {"finding_id": "A4-006", "owner_ticket": "T-IGX-044-A", "disposition": "PT-T01 gate emitter / tenant propagation"},
    {"finding_id": "A3-018", "owner_ticket": "T-IGX-046-G", "disposition": "PT-G02 STATE-X citation repair"},
    {"finding_id": "A4-016", "owner_ticket": "T-IGX-043-H", "disposition": "PT-WB dual host LLM resolver"},
]

LANE_ORDER = [
    ("lane_1", "T-IGX-043-H", "PT-WB"),
    ("lane_2", "T-IGX-043-G", "PT-WA"),
    ("lane_3", ["T-IGX-043-B", "T-IGX-043-C", "T-IGX-043-A", "T-IGX-051-D", "T-IGX-051-E", "T-IGX-051-F"]),
    ("lane_4", ["T-IGX-043-D", "T-IGX-043-E", "T-IGX-043-F", "T-IGX-043-I", "T-IGX-043-J", "T-IGX-043-M"]),
    ("lane_5", ["T-IGX-051-B", "T-IGX-051-C", "T-IGX-051-G", "T-IGX-051-N"]),
    ("lane_6", ["T-IGX-043-N", "T-IGX-051-A"]),
    ("lane_7", "T-IGX-051-A", "T-AF2-RW-TENANT"),
]

SERIALIZED_PAIRS = [
    ("T-IGX-043-B", "T-IGX-043-H", "applications/_shared/*_wiring.py via wiring_posture"),
    ("T-IGX-043-B", "T-IGX-051-E", "queue_worker_wiring.py"),
    ("T-IGX-043-H", "T-IGX-051-B", "package_b_strict_production_serving_command.py Part G"),
    ("T-IGX-043-G", "T-IGX-051-A", "harness_host_runtime.py"),
    ("T-IGX-043-J", "T-IGX-043-I", "split PT-WD shared files"),
]


@dataclass
class CodeFixRow:
    frz_id: str
    summary: str
    negative_test: str | None
    remediation_track: str
    owner_ticket: str | None
    delta_findings: list[str] = field(default_factory=list)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()


def _parse_appendix_a() -> list[CodeFixRow]:
    text = _MATRIX.read_text(encoding="utf-8")
    start = text.find(APPENDIX_A_MARK)
    if start < 0:
        raise SystemExit(f"missing {APPENDIX_A_MARK} in matrix")
    section = text[start:]
    for marker in APPENDIX_END_MARKERS:
        end = section.find(marker)
        if end > 0:
            section = section[:end]
            break
    seen: dict[str, CodeFixRow] = {}
    for line in section.splitlines():
        m = _FRZ_ROW_RE.match(line)
        if not m:
            continue
        frz_id = m.group(1)
        if "CODE_FIX" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        summary = parts[1] if len(parts) > 1 else ""
        nt = None
        nt_m = _NT_RE.search(line)
        if nt_m:
            nt = nt_m.group(1)
        elif "NEG-0" in line:
            nt = "NEG-0"
        elif "NEG-P" in line:
            nt = "NEG-P"
        ticket = None
        rw_m = _RW_TICKET_RE.search(line)
        if rw_m:
            ticket = rw_m.group(1)
        elif "Appendix C," in line:
            tm = _TICKET_RE.search(line.split("Appendix C,")[-1])
            if tm:
                ticket = tm.group(0)
        if not ticket:
            tm = _TICKET_RE.search(line)
            if tm:
                ticket = tm.group(0)
        findings = re.findall(r"A\d+-\d+", parts[-2] if len(parts) > 2 else "")
        track = "CODE_FIX"
        if "CODE_FIX (" in line:
            track = line.split("CODE_FIX", 1)[1].split("|", 1)[0].strip(" ): ")
        row = CodeFixRow(
            frz_id=frz_id,
            summary=summary[:200],
            negative_test=nt,
            remediation_track=track,
            owner_ticket=ticket or "T-IGX-050-D",
            delta_findings=findings,
        )
        if frz_id not in seen or (seen[frz_id].negative_test is None and row.negative_test):
            seen[frz_id] = row
    return list(seen.values())


def _scan_plan_paths(ticket: str) -> list[str]:
    plan = _PLANS / f"PLAN_{ticket}.md"
    if not plan.is_file():
        return []
    text = plan.read_text(encoding="utf-8", errors="replace")
    paths = set(re.findall(r"(?<![\w/.])(?:intergrax|agents|applications|scripts|tests)/[\w./\-\*]+", text))
    return sorted(p for p in paths if not p.endswith("*") or p.count("*") == 1)


def _reconcile_35_vs_45(rows: list[CodeFixRow]) -> dict[str, object]:
    """MEP cites 35 unplanned WP12-Q12 rows; matrix Appendix C lists 45 CODE_FIX rows @ T-IGX-050-D."""
    return {
        "mep_historical_unplanned_count": 35,
        "matrix_appendix_c_code_fix_count": len(rows),
        "delta_explanation": (
            "T-IGX-050-D reconciled 225 FRZ rows; the matrix file contains 45 CODE_FIX tokens "
            "(incl. summary/checklist duplicates). Appendix A table parse yields the unique inventory "
            "below (subset bounded by appendix-A end marker). MEP historical 35-pack = WP12-Q12 scope; "
            "no count forced to 35."
        ),
        "find_af2_23_note": (
            "FIND-AF2-23 (18 CODE_FIX without explicit negative tests) is historical WP-12 phase-check-3; "
            f"current matrix CODE_FIX rows with NT-* or NEG-P: "
            f"{sum(1 for r in rows if r.negative_test)} / {len(rows)}."
        ),
    }


def build() -> dict[str, object]:
    head = _git_head()
    code_fix_rows = _parse_appendix_a()
    by_ticket: dict[str, list[CodeFixRow]] = defaultdict(list)
    for row in code_fix_rows:
        if row.owner_ticket:
            by_ticket[row.owner_ticket].append(row)

    ticket_writes: dict[str, list[str]] = {}
    for ticket in sorted(by_ticket):
        ticket_writes[ticket] = _scan_plan_paths(ticket)[:80]

    payload: dict[str, object] = {
        "schema": "rw_plan.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "execution_sha": head,
        "node": "T-AF2-RW-PLAN",
        "reconciliation": _reconcile_35_vs_45(code_fix_rows),
        "code_fix_inventory": [
            {
                "frz_id": r.frz_id,
                "summary": r.summary,
                "owner_ticket": r.owner_ticket,
                "negative_test_id": r.negative_test,
                "red_first_prefix_sha": head,
                "red_first_status": "EVIDENCE_MISSING",
                "regression_scope": r.owner_ticket or "UNASSIGNED",
                "delta_findings": r.delta_findings,
                "remediation_track": r.remediation_track,
            }
            for r in code_fix_rows
        ],
        "audit_flag_conflicts": AUDIT_FLAG_CONFLICTS,
        "lane_order": LANE_ORDER,
        "serialization_pairs": [{"first": a, "second": b, "reason": c} for a, b, c in SERIALIZED_PAIRS],
        "ticket_write_sets": ticket_writes,
        "overlap_hotspots": {
            "applications/_shared/harness_host_runtime.py": ["T-IGX-043-J", "T-IGX-043-G", "T-IGX-051-A"],
            "applications/_shared/*_wiring.py": ["T-IGX-043-B", "T-IGX-043-H", "T-IGX-051-B"],
            "intergrax/applications/_shared/queue_worker_wiring.py": ["T-IGX-043-B", "T-IGX-051-E"],
        },
        "path_b_invariants": [
            "ReplayX path_b=B on FRZ matrix L66 rows is non-authoritative (DP-34, FRZ_RR Path B).",
            "Tenant isolation: cross-tenant NEG tests on T-IGX-051-A scope.",
        ],
        "g2_entry_conditions": [
            "SCOPE hard predecessors satisfied",
            "RW train head CI green except known-red register",
            "PT-Q02 wave audit after G2 qualification",
        ],
        "dependencies": {
            "W2": ["T-AF2-FRZ-MATRIX", "T-AF2-DS-EXC-PACKAGE", "T-AF2-BND-CLOSURE"],
            "SCOPE": ["T-AF2-RW-PLAN", "T-AF2-RW-PLAN-AUDIT", "PT-V04", "PT-B01", "PT-B02", "REV-MASTER"],
            "W3": ["SCOPE", "PT-WA"],
        },
        "implementation_disposition": {
            "published_on_development": ["FRZ matrix", "MEP", "WP12 DAG", "traceability validator"],
            "published_on_rw_train_only": "Most product remediation (PT-WA..WF, T-IGX-051-*)",
            "qualified_at_head": "Not claimed — diagnostic W2 outputs only",
        },
    }
    return payload


def _render_md(payload: dict[str, object]) -> str:
    head = payload["execution_sha"]
    rows = payload["code_fix_inventory"]
    recon = payload["reconciliation"]
    lines = [
        "# RW-PLAN — T-AF2-RW-PLAN (canonical)",
        "",
        f"**Generated:** {payload['generated_at']}",
        f"**Execution SHA:** `{head}`",
        "**Node:** T-AF2-RW-PLAN (W2, DOC)",
        "",
        "## 0. Reconciliation (35 vs 45 CODE_FIX)",
        "",
        f"- MEP historical unplanned pack: **{recon['mep_historical_unplanned_count']}**",
        f"- Matrix Appendix C CODE_FIX rows: **{recon['matrix_appendix_c_code_fix_count']}**",
        f"- {recon['delta_explanation']}",
        f"- {recon['find_af2_23_note']}",
        "",
        "## 1. CODE_FIX inventory",
        "",
        "| FRZ | Owner ticket | Negative test | Red-first | Regression |",
        "|-----|--------------|---------------|-----------|------------|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['frz_id']}` | {r.get('owner_ticket') or 'UNASSIGNED'} | "
            f"{r.get('negative_test_id') or '—'} | {r['red_first_status']} | {r['regression_scope']} |"
        )
    lines.extend(
        [
            "",
            "## 2. Audit-flag conflicts (6)",
            "",
            "| Finding | Owner ticket | Disposition |",
            "|---------|--------------|-------------|",
        ]
    )
    for c in payload["audit_flag_conflicts"]:
        lines.append(f"| {c['finding_id']} | {c['owner_ticket']} | {c['disposition']} |")
    lines.extend(
        [
            "",
            "## 3. Lane order and serialization",
            "",
            "See `rw_plan.v1.json` → `lane_order`, `serialization_pairs`, `overlap_hotspots`.",
            "",
            "## 4. Path B and tenant isolation",
            "",
        ]
    )
    for inv in payload["path_b_invariants"]:
        lines.append(f"- {inv}")
    lines.extend(["", "## 5. G2 entry conditions", ""])
    for g in payload["g2_entry_conditions"]:
        lines.append(f"- {g}")
    lines.extend(
        [
            "",
            "## 6. Dependencies (SCOPE / W3)",
            "",
            f"W2 preds: {payload['dependencies']['W2']}",
            f"SCOPE preds (partial): {payload['dependencies']['SCOPE']}",
            f"W3: {payload['dependencies']['W3']}",
            "",
            "## 7. Write sets (from approved plans; mechanical scan)",
            "",
        ]
    )
    for ticket, paths in sorted(payload["ticket_write_sets"].items()):
        lines.append(f"### {ticket}")
        if not paths:
            lines.append("_No plan paths extracted (plan missing or paths only in A1 blocks)._")
        else:
            for p in paths[:25]:
                lines.append(f"- `{p}`")
            if len(paths) > 25:
                lines.append(f"- … +{len(paths) - 25} more (see JSON)")
        lines.append("")
    lines.append("## 8. Machine contract")
    lines.append("")
    lines.append("Strict phase check 3: `scripts/governance/verify_rw_plan_phase_check3.py`")
    lines.append("")
    lines.append(f"JSON SSOT: `rw_plan.v1.json` (schema `{payload['schema']}`)")
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = build()
    _RW_PLAN_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _RW_PLAN_MD.write_text(_render_md(payload), encoding="utf-8")
    print(f"[OK] Wrote {_RW_PLAN_MD.relative_to(_REPO)}")
    print(f"[OK] Wrote {_RW_PLAN_JSON.relative_to(_REPO)} ({len(payload['code_fix_inventory'])} CODE_FIX rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
