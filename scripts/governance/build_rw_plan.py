#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Build canonical T-AF2-RW-PLAN artefacts from FRZ matrix + ticket plans (W2-RECOVERY-02)."""

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
_RW_TICKET_RE = re.compile(r"RW ticket (T-IGX-\d{3}-[A-Z0-9]+)")
_NT_RE = re.compile(r"(NT-[A-Z0-9]+-\d+|NT-[A-Z]{3}-\d+|NT-ROW-\d+)")
_FRZ_ROW_RE = re.compile(r"^\|\s*`(FRZ-[A-Z]+-\d+)`\s*\|")
APPENDIX_A_COL = {
    "frz_id": 1,
    "summary": 2,
    "tests": 10,
    "negative_tests": 11,
    "required_fix": 12,
    "required_certification": 13,
}

AUDIT_FLAG_CONFLICTS = [
    {"finding_id": "A4-005", "owner_ticket": "T-IGX-051-H", "disposition": "T-AF2-DEPS-BATCH; PT-WF overlap"},
    {
        "finding_id": "A4-018",
        "owner_ticket": "T-IGX-051-D",
        "disposition": "T-AF2-RW-RUNTIME; shares PT-WF / FRZ-CTR rows",
    },
    {"finding_id": "A6-005", "owner_ticket": "T-IGX-043-I", "disposition": "PT-WD; ENT manifest engine scope"},
    {
        "finding_id": "A4-006",
        "owner_ticket": None,
        "owner_node": "PT-T01",
        "owner_wave": "W3",
        "owner_source": "WP12 DAG / PT-T01 / A4-006",
        "disposition": "PT-T01 gate emitter / tenant propagation",
    },
    {"finding_id": "A3-018", "owner_ticket": "T-IGX-046-G", "disposition": "PT-G02 STATE-X citation repair"},
    {"finding_id": "A4-016", "owner_ticket": "T-IGX-043-H", "disposition": "PT-WB dual host LLM resolver"},
]

SCOPE_HARD_PREDECESSORS = [
    "M1",
    "DS-1",
    "DS-2",
    "DS-3",
    "DS-4",
    "DS-EXC",
    "PT-V04",
    "PT-B01",
    "PT-B02",
    "REV-MASTER",
    "T-AF2-RW-PLAN",
    "T-AF2-RW-PLAN-AUDIT",
]

LANE_ORDER = [
    ("lane_1", "T-IGX-043-G", "PT-WA"),
    ("lane_2", "T-IGX-043-H", "PT-WB"),
    ("lane_3", ["T-IGX-043-B", "T-IGX-043-C", "T-IGX-043-A", "T-IGX-051-D", "T-IGX-051-E", "T-IGX-051-F"]),
    ("lane_4", ["T-IGX-043-D", "T-IGX-043-E", "T-IGX-043-F", "T-IGX-043-I", "T-IGX-043-J", "T-IGX-043-M"]),
    ("lane_5", ["T-IGX-051-B", "T-IGX-051-C", "T-IGX-051-G", "T-IGX-051-N"]),
    ("lane_6", ["T-IGX-043-N"]),
    ("lane_7", "T-IGX-051-A", "T-AF2-RW-TENANT"),
]

SERIALIZED_PAIRS = [
    ("T-IGX-043-B", "T-IGX-043-H", "applications/_shared/*_wiring.py via wiring_posture"),
    ("T-IGX-043-B", "T-IGX-051-E", "queue_worker_wiring.py"),
    ("T-IGX-043-H", "T-IGX-051-B", "package_b_strict_production_serving_command.py Part G"),
    ("T-IGX-043-G", "T-IGX-051-A", "harness_host_runtime.py"),
    ("T-IGX-043-J", "T-IGX-043-I", "split PT-WD shared files"),
]

# Source-backed CODE_FIX ownership (MEP-AF2 v2.1 / WP12 DAG); exact FRZ ids only.
EXPLICIT_CODE_FIX_OWNERSHIP: dict[str, dict[str, object]] = {
    "FRZ-TYP-01": {
        "owner_ticket": "T-IGX-051-G",
        "owner_node": "T-AF2-RW-TYP",
        "owner_wave": "W3",
        "owner_source": "MEP-AF2 v2.1 / section G / DP-36",
    },
    "FRZ-TYP-03": {
        "owner_ticket": "T-IGX-051-G",
        "owner_node": "T-AF2-RW-TYP",
        "owner_wave": "W3",
        "owner_source": "MEP-AF2 v2.1 / section G / DP-36",
    },
    "FRZ-RPL-04": {
        "owner_ticket": "T-IGX-051-G",
        "owner_node": "T-AF2-RW-TYP",
        "owner_wave": "W3",
        "owner_source": "MEP-AF2 v2.1 / section G / DP-36",
    },
    "FRZ-TEN-02": {
        "owner_ticket": None,
        "owner_node": "PT-T01",
        "owner_wave": "W3",
        "owner_source": "WP12 DAG / PT-T01 / A4-006",
    },
    "FRZ-TEN-07": {
        "owner_ticket": None,
        "owner_node": "PT-T01",
        "owner_wave": "W3",
        "owner_source": "WP12 DAG / PT-T01 / A4-006",
    },
    "FRZ-FRZ-10": {
        "owner_ticket": None,
        "owner_node": "T-B7-PFP-ACTIVATE",
        "owner_wave": "W5",
        "verification_node": "T-B7-PFP-VERIFY",
        "verification_wave": "W8",
        "owner_source": "MEP-AF2 v2.1 / W5 and W8",
        "owner_ticket_family": "T-IGX-052-*",
    },
}


@dataclass
class CodeFixRow:
    frz_id: str
    summary: str
    negative_test: str | None
    remediation_track: str
    owner_ticket: str | None
    regression_scope: str | None
    delta_findings: list[str] = field(default_factory=list)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()


def _extract_appendix_a_section() -> str:
    text = _MATRIX.read_text(encoding="utf-8")
    start = text.find(APPENDIX_A_MARK)
    if start < 0:
        raise SystemExit(f"missing {APPENDIX_A_MARK} in matrix")
    section = text[start:]
    for marker in APPENDIX_END_MARKERS:
        end = section.find(marker)
        if end > 0:
            return section[:end]
    return section


def _row_parts(line: str) -> list[str]:
    return [p.strip() for p in line.split("|")]


def _extract_negative_test(neg_col: str) -> str | None:
    nt_m = _NT_RE.search(neg_col)
    if nt_m:
        return nt_m.group(1)
    if "NEG-0" in neg_col:
        return "NEG-0"
    if "NEG-P" in neg_col:
        return "NEG-P"
    return None


def _extract_owner_ticket(cert_col: str, fix_col: str) -> str | None:
    rw_m = _RW_TICKET_RE.search(cert_col)
    if rw_m:
        return rw_m.group(1)
    for pattern in (
        r"Appendix C,?\s*(T-IGX-\d{3}-[A-Z0-9]+)",
        r";\s*(T-IGX-\d{3}-[A-Z0-9]+)\)",
        r";\s*(T-IGX-\d{3}-[A-Z0-9]+)\s*\+",
    ):
        m = re.search(pattern, fix_col)
        if m and "*" not in m.group(1):
            return m.group(1)
    return None


def apply_explicit_ownership(frz_id: str, row: CodeFixRow) -> dict[str, object]:
    """Merge matrix parse with canonical explicit ownership for six FRZ rows."""
    explicit = EXPLICIT_CODE_FIX_OWNERSHIP.get(frz_id)
    if explicit is None:
        return {"owner_ticket": row.owner_ticket}
    out: dict[str, object] = {"owner_provenance": "explicit_mep_wp12"}
    ticket = explicit.get("owner_ticket")
    if ticket is not None:
        out["owner_ticket"] = ticket
    else:
        out["owner_ticket"] = None
    for key in (
        "owner_node",
        "owner_wave",
        "owner_source",
        "verification_node",
        "verification_wave",
        "owner_ticket_family",
    ):
        if key in explicit:
            out[key] = explicit[key]
    if row.owner_ticket and row.owner_ticket != out.get("owner_ticket"):
        out["matrix_owner_ticket_parse"] = row.owner_ticket
    return out


def _code_fix_row_key(row: CodeFixRow) -> tuple:
    return (
        row.summary,
        row.negative_test,
        row.remediation_track,
        row.owner_ticket,
        row.regression_scope,
        tuple(row.delta_findings),
    )


def parse_appendix_a_code_fix_inventory(section: str | None = None) -> list[CodeFixRow]:
    """Parse CODE_FIX rows from Appendix A required_fix column only."""
    if section is None:
        section = _extract_appendix_a_section()
    seen: dict[str, CodeFixRow] = {}
    for line in section.splitlines():
        if not _FRZ_ROW_RE.match(line):
            continue
        parts = _row_parts(line)
        if len(parts) <= max(APPENDIX_A_COL.values()):
            raise SystemExit(f"malformed Appendix A row (columns): {line[:120]}")
        required_fix = parts[APPENDIX_A_COL["required_fix"]]
        if not required_fix.startswith("CODE_FIX"):
            continue
        frz_id = parts[APPENDIX_A_COL["frz_id"]].strip("`")
        summary = parts[APPENDIX_A_COL["summary"]]
        neg_col = parts[APPENDIX_A_COL["negative_tests"]]
        tests_col = parts[APPENDIX_A_COL["tests"]]
        cert_col = parts[APPENDIX_A_COL["required_certification"]]
        delta_findings = re.findall(r"A\d+-\d+", parts[14] if len(parts) > 14 else "")
        regression = tests_col.strip()
        if not regression or regression in {"—", "-", "none"}:
            regression = None
        row = CodeFixRow(
            frz_id=frz_id,
            summary=summary[:500],
            negative_test=_extract_negative_test(neg_col),
            remediation_track=required_fix,
            owner_ticket=_extract_owner_ticket(cert_col, required_fix),
            regression_scope=regression,
            delta_findings=delta_findings,
        )
        if frz_id in seen and _code_fix_row_key(seen[frz_id]) != _code_fix_row_key(row):
            raise SystemExit(f"conflicting CODE_FIX records for {frz_id}")
        seen[frz_id] = row
    return list(seen.values())


def _load_prior_red_first() -> dict[str, dict[str, object]]:
    if not _RW_PLAN_JSON.is_file():
        return {}
    try:
        data = json.loads(_RW_PLAN_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, dict[str, object]] = {}
    for row in data.get("code_fix_inventory") or []:
        if not isinstance(row, dict):
            continue
        frz = row.get("frz_id")
        status = row.get("red_first_status")
        sha = row.get("red_first_prefix_sha")
        ref = row.get("red_first_evidence_ref")
        if frz and status == "RED_AT_PREFIX" and isinstance(sha, str) and len(sha) == 40 and ref:
            out[str(frz)] = {
                "red_first_prefix_sha": sha,
                "red_first_status": status,
                "red_first_evidence_ref": ref,
            }
    return out


def _scan_plan_paths(ticket: str) -> list[str]:
    plan = _PLANS / f"PLAN_{ticket}.md"
    if not plan.is_file():
        return []
    text = plan.read_text(encoding="utf-8", errors="replace")
    paths = set(re.findall(r"(?<![\w/.])(?:intergrax|agents|applications|scripts|tests)/[\w./\-\*]+", text))
    return sorted(p for p in paths if not p.endswith("*") or p.count("*") == 1)


def _reconcile_35_vs_45(rows: list[CodeFixRow]) -> dict[str, object]:
    return {
        "mep_historical_unplanned_count": 35,
        "matrix_appendix_c_code_fix_count": len(rows),
        "delta_explanation": (
            "T-IGX-050-D reconciled 225 FRZ rows; CODE_FIX inventory is parsed from Appendix A "
            "required_fix column only (bounded by appendix-A end marker). MEP historical 35-pack = "
            "WP12-Q12 scope; matrix row count is authoritative (not hardcoded)."
        ),
        "find_af2_23_note": (
            "FIND-AF2-23 (18 CODE_FIX without explicit negative tests) is historical WP-12 phase-check-3; "
            f"current matrix CODE_FIX rows with NT-* or NEG-P/NEG-0: "
            f"{sum(1 for r in rows if r.negative_test)} / {len(rows)}."
        ),
    }


def _load_known_red_fix_inventory() -> list[dict[str, object]]:
    import importlib.util

    path = Path(__file__).with_name("known_red_fix_rows.py")
    spec = importlib.util.spec_from_file_location("known_red_fix_rows", path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load known_red_fix_rows")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(getattr(mod, "KNOWN_RED_FIX_INVENTORY"))


def build() -> dict[str, object]:
    head = _git_head()
    prior_rf = _load_prior_red_first()
    code_fix_rows = sorted(parse_appendix_a_code_fix_inventory(), key=lambda r: r.frz_id)
    by_ticket: dict[str, list[CodeFixRow]] = defaultdict(list)
    for row in code_fix_rows:
        if row.owner_ticket:
            by_ticket[row.owner_ticket].append(row)

    ticket_writes: dict[str, list[str]] = {}
    for ticket in sorted(by_ticket):
        ticket_writes[ticket] = _scan_plan_paths(ticket)[:80]

    inventory: list[dict[str, object]] = []
    for r in code_fix_rows:
        prior = prior_rf.get(r.frz_id)
        if prior:
            rf_sha = prior["red_first_prefix_sha"]
            rf_status = prior["red_first_status"]
            rf_ref = prior.get("red_first_evidence_ref")
        else:
            rf_sha = None
            rf_status = "EVIDENCE_MISSING"
            rf_ref = None
        reg_scope = r.regression_scope if r.regression_scope else "MAPPING_MISSING"
        ownership = apply_explicit_ownership(r.frz_id, r)
        entry: dict[str, object] = {
            "frz_id": r.frz_id,
            "summary": r.summary,
            "owner_ticket": ownership.get("owner_ticket", r.owner_ticket),
            "negative_test_id": r.negative_test,
            "red_first_prefix_sha": rf_sha,
            "red_first_status": rf_status,
            "red_first_evidence_ref": rf_ref,
            "regression_scope": reg_scope,
            "regression_scope_source": "matrix_tests_column" if r.regression_scope else "MAPPING_MISSING",
            "delta_findings": r.delta_findings,
            "remediation_track": r.remediation_track,
        }
        for key, val in ownership.items():
            if key != "owner_ticket":
                entry[key] = val
        inventory.append(entry)

    payload: dict[str, object] = {
        "schema": "rw_plan.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "execution_sha": head,
        "node": "T-AF2-RW-PLAN",
        "reconciliation": _reconcile_35_vs_45(code_fix_rows),
        "code_fix_inventory": inventory,
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
            "SCOPE": list(SCOPE_HARD_PREDECESSORS),
            "W3": ["SCOPE", "PT-WA"],
        },
        "implementation_disposition": {
            "published_on_development": ["FRZ matrix", "MEP", "WP12 DAG", "traceability validator"],
            "published_on_rw_train_only": "Most product remediation (PT-WA..WF, T-IGX-051-*)",
            "qualified_at_head": "Not claimed — diagnostic W2 outputs only",
        },
        "known_red_fix_inventory": _load_known_red_fix_inventory(),
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
        f"- Matrix Appendix A CODE_FIX rows: **{recon['matrix_appendix_c_code_fix_count']}**",
        f"- {recon['delta_explanation']}",
        f"- {recon['find_af2_23_note']}",
        "",
        "## 1. CODE_FIX inventory",
        "",
        "| FRZ | Owner ticket | Negative test | Red-first | Regression |",
        "|-----|--------------|---------------|-----------|------------|",
    ]
    for r in rows:
        sha_disp = r.get("red_first_prefix_sha") or "—"
        owner = r.get("owner_ticket")
        if not owner and r.get("owner_node"):
            owner = str(r["owner_node"])
        owner_disp = owner or "UNASSIGNED"
        lines.append(
            f"| `{r['frz_id']}` | {owner_disp} | "
            f"{r.get('negative_test_id') or '—'} | {r['red_first_status']} @ {sha_disp} | {r['regression_scope']} |"
        )
    kr_rows = payload.get("known_red_fix_inventory") or []
    if kr_rows:
        lines.extend(
            [
                "",
                "## 1b. Known-red FIX inventory (20)",
                "",
                "| KR | Owner node | Negative test | File scope (summary) |",
                "|----|------------|---------------|----------------------|",
            ]
        )
        for r in kr_rows:
            scope = r.get("file_scope") or []
            scope_disp = "; ".join(scope[:2])
            if len(scope) > 2:
                scope_disp += f" (+{len(scope) - 2})"
            lines.append(f"| `{r['kr_id']}` | {r['owner_node']} | {r['negative_test_id']} | {scope_disp} |")
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
            f"SCOPE preds: {payload['dependencies']['SCOPE']}",
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
