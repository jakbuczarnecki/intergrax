#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501
"""Mechanical traceability validator for ARCH-FREEZE planning dossier artifacts."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_QUAL = _REPO / "docs/project/maintainers/qualification"
_CHECKLIST = _QUAL / "PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
_MATRIX = _QUAL / "ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md"
_WBS = _QUAL / "ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md"
_GAPS = _QUAL / "ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md"
_QUAL_PLAN = _QUAL / "ARCH_FREEZE_QUALIFICATION_AND_EVIDENCE_PLAN.md"
_IIAC_PROTOCOL = _QUAL / "INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md"
_IIAC_ACTIVATION = _QUAL / "IIAC_POLICY_ACTIVATION_RECORD.json"

PATH_B_FULL = frozenset({"FRZ-PRD-01", "FRZ-PRD-02"})
BLOCKER_FINDINGS = frozenset(
    {
        "FIND-PROD-REACH-DEFERRED",
        "FIND-PROD-REACH-CLASSIFICATION",
        "FIND-VBN-PARTIAL",
        "FIND-FORMAL-B6",
        "FIND-ACCEPTED-BASELINE",
    }
)
EXPECTED_WBS_TASKS = 54
EXPECTED_QUAL_RUNS = 16
AUDIT_BASELINE_SHA = "a0624e28ff77b061e83f3e10aab4c2a9083ee1b4"

PROMOTE_FAMS = ("BND", "CTR", "OBS", "OWN", "PLG", "PRD", "REG", "REL", "SEC", "TYP")
PROMOTE_TASKS = tuple(f"T-B7-PROMOTE-{fam}" for fam in PROMOTE_FAMS)
FRZ_FRZ02_PREREQ_TASKS = (
    "T-B5-EVOL-FRZ",
    "T-B6-FRZ-AUD",
    "T-B2-DEBT",
    "T-B3-DOC",
    "T-B7-TEN-REVAL",
    "T-PATHB-EXCL",
    "T-B7-CMP-REVAL",
    "T-B7-PROMOTE-RPL",
    "T-B7-PROMOTE-DEBT",
    "T-B7-PROMOTE-DOC",
) + PROMOTE_TASKS
INDEPENDENT_AUDITOR_TASKS = frozenset(
    {
        "T-B5-EVOL-B6",
        "T-B6-ENT-B6",
        "T-B7-ECAP-2C",
        "T-B7-TEN-REVAL",
        "T-B7-CMP-REVAL",
        "T-B7-VBN-INDEPENDENT-ACCEPT",
    }
)


def _parse_checklist() -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in _CHECKLIST.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (FRZ-[A-Z]+-\d+) \|", line)
        if not m:
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 5:
            statuses[m.group(1)] = parts[4]
    return statuses


def remediation_task(frz_id: str, status: str) -> str:
    if status != "OPEN":
        return "T-B7-TEN-REVAL" if frz_id.startswith("FRZ-TEN-") else "—"
    if frz_id in PATH_B_FULL:
        return "T-PATHB-EXCL"
    if frz_id == "FRZ-BND-03":
        return "T-B2-BND03-RECONCILE"
    fam = frz_id.split("-")[1]
    mapping = {
        "EVOL": "T-B5-EVOL-FRZ",
        "AUD": "T-B6-FRZ-AUD",
        "FRZ": "T-B7-FRZ-FRZ",
        "DEBT": "T-B2-DEBT",
        "DOC": "T-B3-DOC",
        "BND": "T-B7-PROMOTE-BND",
        "OWN": "T-B7-PROMOTE-OWN",
        "CTR": "T-B7-PROMOTE-CTR",
        "TYP": "T-B7-PROMOTE-TYP",
        "PLG": "T-B7-PROMOTE-PLG",
        "OBS": "T-B7-PROMOTE-OBS",
        "SEC": "T-B7-PROMOTE-SEC",
        "REL": "T-B7-PROMOTE-REL",
        "PRD": "T-B7-PROMOTE-PRD",
        "REG": "T-B7-PROMOTE-REG",
        "RPL": "T-B2-VBN",
    }
    return mapping.get(fam, f"T-B7-PROMOTE-{fam}")


def _coord_arch_frz_deps() -> str:
    inner = ",".join(FRZ_FRZ02_PREREQ_TASKS)
    return f"[{inner}]"


def _ancestors(graph: dict[str, list[str]], node: str) -> set[str]:
    seen: set[str] = set()
    stack = list(graph.get(node, []))
    while stack:
        nxt = stack.pop()
        if nxt in seen:
            continue
        seen.add(nxt)
        stack.extend(graph.get(nxt, []))
    return seen


def _validate_semantic_dag(graph: dict[str, list[str]], task_ids: set[str], errors: list[str]) -> None:
    required = {
        "T-B7-COORD-ARCH-FRZ",
        "T-B7-FRZ-FRZ",
        "T-B7-PFP-ACTIVATE",
        "T-B7-PFP-VERIFY",
        "T-PATHB-EXCL",
        "T-B8-ROADMAP",
        "T-B8-ECAP-16",
    }
    missing = required - task_ids
    if missing:
        errors.append(f"semantic DAG missing tasks: {sorted(missing)}")
        return

    coord_anc = _ancestors(graph, "T-B7-COORD-ARCH-FRZ")
    for prereq in FRZ_FRZ02_PREREQ_TASKS:
        if prereq not in coord_anc:
            errors.append(f"FRZ-FRZ-02: T-B7-COORD-ARCH-FRZ must follow {prereq} (reachability)")

    frz_frz_anc = _ancestors(graph, "T-B7-FRZ-FRZ")
    for gate in ("T-B7-COORD-ARCH-FRZ", "T-B7-ECAP-2C", "T-B7-PFP-VERIFY"):
        if gate not in frz_frz_anc:
            errors.append(f"T-B7-FRZ-FRZ must follow {gate} (FRZ-FRZ suite gate)")

    if "T-B7-FRZ-FRZ" in _ancestors(graph, "T-PATHB-EXCL"):
        errors.append("Path B: T-PATHB-EXCL must not depend on T-B7-FRZ-FRZ")

    pfp_anc = _ancestors(graph, "T-B7-PFP-VERIFY")
    if "T-B7-PFP-ACTIVATE" not in pfp_anc:
        errors.append("FRZ-FRZ-10: T-B7-PFP-VERIFY must follow T-B7-PFP-ACTIVATE")

    roadmap_anc = _ancestors(graph, "T-B8-ROADMAP")
    for gate in ("T-B7-FRZ-FRZ", "T-B7-PFP-VERIFY", "T-B7-COORD-ARCH-FRZ"):
        if gate not in roadmap_anc:
            errors.append(f"Formal closure T-B8-ROADMAP must follow {gate}")

    if "T-B7-PFP-VERIFY" not in roadmap_anc:
        errors.append("FRZ-FRZ-10: post-freeze policy must be verified before T-B8-ROADMAP")

    ecap16_anc = _ancestors(graph, "T-B8-ECAP-16")
    if "T-B8-ROADMAP" not in ecap16_anc:
        errors.append("T-B8-ECAP-16 must follow T-B8-ROADMAP (post-closure handoff only)")

    if "T-B7-PFP-ACTIVATE" in ecap16_anc and "T-B8-ROADMAP" not in _ancestors(graph, "T-B7-PFP-ACTIVATE"):
        pass  # PFP activate is pre-roadmap by design

    auditor_before: tuple[tuple[str, str, str], ...] = (
        ("T-B5-EVOL-B6", "T-B5-EVOL-FRZ", "EVOL B6 before FRZ-EVOL promotion"),
        ("T-B6-ENT-B6", "T-B6-FRZ-AUD", "ENT B6 before FRZ-AUD promotion"),
        ("T-B7-ECAP-2C", "T-B7-FRZ-FRZ", "ECAP §2.C before FRZ-FRZ admission"),
    )
    for aud_task, dependent, label in auditor_before:
        if aud_task not in _ancestors(graph, dependent):
            errors.append(f"{label}: {aud_task} must precede {dependent}")

    for aud_task in INDEPENDENT_AUDITOR_TASKS:
        if aud_task not in task_ids:
            errors.append(f"missing independent auditor task {aud_task}")

    if "T-B5-EVOL-FORMAL-CLOSURE" not in _ancestors(graph, "T-B6-OD-ENT-00"):
        errors.append("OD-ENT-00 must follow T-B5-EVOL-FORMAL-CLOSURE (EVOL formal closure)")
    if "T-B5-EVOL-FRZ" not in _ancestors(graph, "T-B5-EVOL-FORMAL-CLOSURE"):
        errors.append("T-B5-EVOL-FORMAL-CLOSURE must follow T-B5-EVOL-FRZ")
    if "T-B6-OD-ENT-02" not in _ancestors(graph, "T-B6-OD-ENT-08"):
        errors.append("OD-ENT-08 must follow T-B6-OD-ENT-02 (branch protection)")
    if "T-B7-VBN-INDEPENDENT-ACCEPT" not in _ancestors(graph, "T-B7-PROMOTE-RPL"):
        errors.append("T-B7-PROMOTE-RPL must follow T-B7-VBN-INDEPENDENT-ACCEPT (not engineering-only)")
    if "T-B7-LOCK-FREEZE-SHA" not in _ancestors(graph, "T-B7-L3-FREEZE"):
        errors.append("T-B7-L3-FREEZE must follow T-B7-LOCK-FREEZE-SHA")
    if "T-B6-OD-ENT-08" not in _ancestors(graph, "T-B7-PFP-ACTIVATE"):
        errors.append("T-B7-PFP-ACTIVATE must follow T-B6-OD-ENT-08")
    if "T-B7-PFP-ACTIVATE" not in _ancestors(graph, "T-B7-LOCK-FREEZE-SHA"):
        errors.append("T-B7-LOCK-FREEZE-SHA must follow T-B7-PFP-ACTIVATE (post-freeze policy before lock)")
    if "T-B7-L3-FREEZE" in _ancestors(graph, "T-B7-PFP-ACTIVATE"):
        errors.append("T-B7-PFP-ACTIVATE must precede T-B7-L3-FREEZE (pre-lock activation)")
    if "T-B7-LOCK-FREEZE-SHA" not in _ancestors(graph, "T-B7-VBN-INDEPENDENT-ACCEPT"):
        errors.append("T-B7-VBN-INDEPENDENT-ACCEPT must follow T-B7-LOCK-FREEZE-SHA (certify @ FREEZE_SHA)")
    if "T-B7-LOCK-FREEZE-SHA" in _ancestors(graph, "T-B7-VBN-HEAD-DOSSIER"):
        errors.append("T-B7-VBN-HEAD-DOSSIER must not require T-B7-LOCK-FREEZE-SHA (HEAD prep allowed pre-lock)")
    if "T-B7-L3-FREEZE" not in _ancestors(graph, "T-B7-PFP-VERIFY"):
        errors.append("T-B7-PFP-VERIFY must follow T-B7-L3-FREEZE (verify @ FREEZE_SHA)")


def sync_matrix() -> None:
    lines = _MATRIX.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        if not line.startswith("| FRZ-"):
            out.append(line)
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 12:
            out.append(line)
            continue
        fid = parts[1]
        st = parts[2]
        parts[9] = remediation_task(fid, st)
        out.append("| " + " | ".join(parts[1:-1]) + " |")
    _MATRIX.write_text("\n".join(out) + "\n", encoding="utf-8")


def _open_frz_by_family(statuses: dict[str, str]) -> dict[str, list[str]]:
    by_fam: dict[str, list[str]] = defaultdict(list)
    for fid, st in sorted(statuses.items()):
        if st == "OPEN":
            by_fam[fid.split("-")[1]].append(fid)
    return by_fam


def _frz_csv(ids: list[str]) -> str:
    return ",".join(ids)


def sync_wbs() -> None:
    statuses = _parse_checklist()
    by_fam = _open_frz_by_family(statuses)
    evol = _frz_csv(by_fam["EVOL"])
    aud = _frz_csv(by_fam["AUD"])
    frz = _frz_csv(by_fam["FRZ"])
    debt = _frz_csv(by_fam["DEBT"])
    doc = _frz_csv(by_fam["DOC"])
    rpl = _frz_csv(by_fam["RPL"])
    ten = _frz_csv([f"FRZ-TEN-{i:02d}" for i in range(1, 13)])

    def promote(fam: str) -> str:
        return _frz_csv(by_fam[fam])

    rows: list[tuple] = [
        (
            "T-R0-WAVE1-SSOT-RECONCILE",
            "R0",
            "FRZ-FRZ-04",
            "FIND-ARTIFACT-CONTRADICTION,FIND-VALIDATOR-WEAK",
            "implementer",
            "docs/project/maintainers/qualification/ARCH_FREEZE_*",
            "[]",
            "Reconcile SSOT: G6 ACCEPTED @ 950097ea; 54 tasks; SHA ladder; remove AUDIT_PENDING drift",
            "uv run python scripts/governance/verify_arch_freeze_plan_traceability.py",
            "ARCH_FREEZE dossier @ 950097ea",
            "WAVE1 disposition aligned; validator WBS=54",
            "MED",
            "no",
            "COMPLETE @ 950097ea",
        ),
        (
            "T-R0-BASELINE-CLOSURE-HEAD",
            "R0",
            "",
            "FIND-PRECHECKLIST-HEAD",
            "implementer",
            "ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md",
            "[T-R0-WAVE1-SSOT-RECONCILE]",
            "Close T-B0-001: refresh pre-admission pointer @ CLOSURE_DOC_HEAD; immutable AUDIT_BASELINE_SHA unchanged",
            "git rev-parse HEAD; doc pointer refresh",
            "pre-admission dossier @ 950097ea",
            "pointer current; AUDIT_BASELINE_SHA=a0624e28 locked",
            "LOW",
            "yes",
            "COMPLETE @ 950097ea",
        ),
        (
            "T-B0-001",
            "B0",
            "",
            "FIND-PRECHECKLIST-HEAD",
            "implementer",
            "repo-wide",
            "[]",
            "Lock baseline; refresh pre-admission dossier pointer",
            "git merge-base; FRZ mechanical counts",
            "ARCH_FREEZE_* @ baseline",
            "225 FRZ / 26 fam verified",
            "LOW",
            "yes",
            "COMPLETE @ 950097ea (superseded ongoing pointer: T-R0-BASELINE-CLOSURE-HEAD)",
        ),
        (
            "T-B0-SSOT-REFRESH",
            "B0",
            "FRZ-FRZ-04",
            "FIND-PRECHECKLIST-HEAD,FIND-BASELINE-SHA-DRIFT,FIND-ARTIFACT-CONTRADICTION,FIND-VALIDATOR-WEAK",
            "implementer",
            "docs/project/maintainers/qualification/ARCH_FREEZE_*; scripts/governance/verify_arch_freeze_plan_traceability.py",
            "[T-B0-001]",
            f"Lock AUDIT_BASELINE_SHA={AUDIT_BASELINE_SHA}; reconcile SSOT; harden traceability validator",
            "uv run python scripts/governance/verify_arch_freeze_plan_traceability.py",
            f"ARCH_FREEZE dossier @ {AUDIT_BASELINE_SHA[:8]}",
            "SSOT cites HEAD; validator extended",
            "MED",
            "no",
            "PLANNED",
        ),
        (
            "T-B0-TRACE-VALIDATE",
            "B0",
            "",
            "",
            "implementer",
            "scripts/governance/verify_arch_freeze_plan_traceability.py",
            "[T-B0-SSOT-REFRESH]",
            "Run mechanical dossier traceability validator",
            "uv run python scripts/governance/verify_arch_freeze_plan_traceability.py",
            "validator stdout",
            "exit 0",
            "LOW",
            "yes",
            "PLANNED",
        ),
        (
            "T-B1-INV",
            "B1",
            "",
            "",
            "implementer",
            "intergrax/; agents/; applications/",
            "[T-B0-SSOT-REFRESH]",
            "Planning-phase ENT manifest 85 checks + 8 anchors",
            "manifest load; dry-run",
            "ARCH_FREEZE_ARCHITECTURE_AND_PRODUCTION_SURFACE_INVENTORY.md",
            "manifest + anchors documented",
            "MED",
            "yes",
            "PLANNED",
        ),
        (
            "T-B1-PROD-REACH-SCAN",
            "B1",
            "FRZ-BND-03,FRZ-BND-04,FRZ-CTR-02",
            "FIND-PROD-REACH-DEFERRED",
            "implementer",
            "production Python tree",
            "[T-B1-INV]",
            "Full production import/reachability scan (NOT EXECUTED in planning)",
            "COMPAT-X/PROD-Q inventory patterns",
            "extended surface inventory",
            "UNCLASSIFIED=0",
            "HIGH",
            "no",
            "SUPERSEDED @ 18a0f68",
        ),
        (
            "T-B1-ENGINE",
            "B1",
            "FRZ-BND-04,FRZ-CTR-02,FRZ-CMP-01",
            "FIND-PROD-REACH-CLASSIFICATION,FIND-UNKNOWN-HARDCODE,FIND-CW-GATE-FAIL,FIND-B1-EXIT-CODE,FIND-B1-SHA-PAYLOAD",
            "implementer",
            "scripts/governance/arch_freeze_b1_closed_world_inventory.py",
            "[T-B1-INV,T-B0-SSOT-REFRESH]",
            "Fix classification, UNKNOWN accounting, CW inputs, SHA fields, strict/report CLI",
            "tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py",
            "inventory script diff",
            "B1 engine complete",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B1-MANDATORY-DISPOSITION",
            "B1",
            "FRZ-BND-03,FRZ-CTR-02",
            "FIND-CW-EXTRACTOR-UNKNOWNS,FIND-CW-UNRESOLVED-IMPORTS",
            "implementer",
            "arch_freeze_b1_mandatory_dispositions.json",
            "[T-B1-ENGINE]",
            "Disposition mandatory unknowns and unresolved imports",
            "uv run pytest tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py -k cw04",
            "dispositions JSON",
            "mandatory_undisposed_count=0",
            "HIGH",
            "no",
            "PLANNED",
        ),
        (
            "T-B1-STRICT-QUALIFY",
            "B1",
            "FRZ-CMP-01,FRZ-CMP-06",
            "FIND-CW-GATE-FAIL",
            "tester",
            "scripts/governance/arch_freeze_b1_closed_world_inventory.py",
            "[T-B1-MANDATORY-DISPOSITION]",
            "B1 pytest + certifying --strict-certify inventory run",
            "uv run pytest tests/unit/governance/test_arch_freeze_b1_closed_world_inventory.py",
            ".tmp/session/arch-freeze-b1-cert/",
            "strict_pass=true",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B1-INDEP-AUDIT-PACKAGE",
            "B1",
            "",
            "",
            "implementer",
            "ARCH_FREEZE_B1_INDEPENDENT_AUDIT_HANDOFF.md",
            "[T-B1-STRICT-QUALIFY]",
            "Refresh B1 independent audit handoff; external STOP",
            "uv run python scripts/governance/verify_arch_freeze_plan_traceability.py",
            "B1 handoff dossier",
            "replay @ HEAD documented",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B2-BND03-RECONCILE",
            "B2",
            "FRZ-BND-03,FRZ-DEBT-03",
            "FIND-BND03-DEBT-STALE",
            "implementer",
            "debug_lab_nexus_loop.py; checklist row",
            "[T-B1-STRICT-QUALIFY]",
            "Reconcile checklist vs HEAD; no FRZ status promotion",
            "test_hardening_3_layer_boundary_gate.py",
            "gate log",
            "evidence reconciled",
            "HIGH",
            "no",
            "PLANNED",
        ),
        (
            "T-B2-VBN",
            "B2",
            rpl,
            "FIND-VBN-PARTIAL",
            "implementer",
            "enterprise_audit/vbn",
            "[T-B1-STRICT-QUALIFY]",
            "Complete VBN artifacts @ current HEAD (roadmap §3.0.8 + VBN §2.1-2.7)",
            "ENT VBN domain checks",
            "VBN §2.1-2.7",
            "prohibited hardcoding=0",
            "HIGH",
            "no",
            "PLANNED",
        ),
        (
            "T-B2-DEBT",
            "B2",
            debt,
            "FIND-DEBT-REG",
            "implementer",
            "debt registers",
            "[T-B2-BND03-RECONCILE]",
            "Zero unresolved architecture debt in frozen scope",
            "ENT debt checks",
            "gap register",
            "FRZ-DEBT promotion path",
            "HIGH",
            "no",
            "PLANNED",
        ),
        (
            "T-B3-DOC",
            "B3",
            doc,
            "",
            "implementer",
            "docs/project manifests",
            "[T-B1-STRICT-QUALIFY]",
            "Doc/manifest alignment",
            "test_replay_x_mandatory_docs_present; check_manifest_annex_c_alignment.py",
            "ENT-A-DOC outcomes",
            "FRZ-DOC promotion path",
            "MED",
            "yes",
            "PLANNED",
        ),
        (
            "T-B4-L1",
            "B4",
            "",
            "",
            "tester",
            "—",
            "[T-B0-TRACE-VALIDATE]",
            "L1 structural validation",
            "uv run python -m scripts.governance.igx_gov ticket",
            "L1 log .tmp/session/",
            "green L1",
            "LOW",
            "yes",
            "PLANNED",
        ),
        (
            "T-B4-L2",
            "B4",
            "FRZ-FRZ-03",
            "",
            "tester",
            "tests/qualification/",
            "[T-B2-DEBT,T-B3-DOC,T-B1-STRICT-QUALIFY]",
            "Risk-based cross-domain pytest",
            "uv run pytest (scoped paths)",
            "pytest log",
            "no regressions",
            "MED",
            "partial",
            "PLANNED",
        ),
        (
            "T-B4-L3-FULL",
            "B4",
            "FRZ-FRZ-03",
            "",
            "tester",
            "scripts/enterprise_audit/",
            "[T-B4-L2]",
            "Certifying FULL audit",
            "uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FULL --output-dir .tmp/proof/arch-freeze-full/ --evidence-bundle",
            "enterprise_audit_report.json",
            "85/85 PASS",
            "HIGH",
            "no",
            "PLANNED",
        ),
        (
            "T-B5-EVOL-B6",
            "B5",
            "",
            "FIND-FORMAL-B6",
            "independent_auditor",
            "evol_x/",
            "[T-B4-L3-FULL]",
            "Protocol 03 EVOL B6",
            "EVOL TES replay",
            "B6_ACCEPTANCE_PUBLICATION_SHA",
            "signed B6",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B5-EVOL-FRZ",
            "B5",
            evol,
            "",
            "operator+auditor",
            "checklist",
            "[T-B5-EVOL-B6]",
            "Promote FRZ-EVOL after EVOL B6",
            "EVOL qual @ TES",
            "checklist rows",
            "10/10 PASS",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B5-EVOL-FORMAL-CLOSURE",
            "B5",
            "",
            "FIND-FORMAL-B6",
            "operator+auditor",
            "evol_x/; roadmap §5",
            "[T-B5-EVOL-B6,T-B5-EVOL-FRZ]",
            "EVOL-X formal CLOSED: W7-06 publication; reconciliation package; roadmap §5 EVOL row",
            "L3-EVOL-REPLAY evidence reuse",
            "FORMAL_CLOSURE_PUBLICATION_SHA",
            "EVOL CLOSED — INDEPENDENTLY ACCEPTED",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B6-OD-ENT-00",
            "B6",
            "",
            "",
            "operator",
            "ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md",
            "[T-B5-EVOL-FORMAL-CLOSURE]",
            "OD-ENT-00 EVOL dependency acceptance",
            "—",
            "OD record",
            "OD-ENT-00 signed",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B6-ENT-B6",
            "B6",
            "",
            "FIND-FORMAL-B6",
            "independent_auditor",
            "ent_audit_x/",
            "[T-B6-OD-ENT-00]",
            "Protocol 03 ENT B6",
            "ENT TES replay",
            "B6 package",
            "signed ENT B6",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B6-OD-ENT-01",
            "B6",
            "",
            "FIND-ACCEPTED-BASELINE",
            "operator",
            "ENT_AUDIT_X_SHA_LADDER.md",
            "[T-B6-ENT-B6]",
            "Promote ACCEPTED_BASELINE_SHA",
            "promote_accepted_baseline CLI",
            "ladder field",
            "baseline promoted",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B6-OD-ENT-02",
            "B6",
            "",
            "",
            "github_admin",
            ".github/workflows/",
            "[T-B6-ENT-B6]",
            "Branch protection + required checks",
            "enterprise-audit.yml",
            "OD_ENT_02 packet",
            "protection live",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B6-FRZ-AUD",
            "B6",
            aud,
            "",
            "operator+auditor",
            "checklist",
            "[T-B6-ENT-B6]",
            "Promote FRZ-AUD after ENT B6",
            "ENT runner evidence",
            "checklist rows",
            "10/10 PASS",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B6-OD-ENT-08",
            "B6",
            "",
            "",
            "operator",
            "roadmap §5",
            "[T-B6-FRZ-AUD,T-B6-OD-ENT-01,T-B6-OD-ENT-02]",
            "Formal ENT stage closure OD-ENT-08",
            "—",
            "FORMAL_CLOSURE_PUBLICATION_SHA",
            "ENT CLOSED",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B7-LOCK-FREEZE-SHA",
            "B7",
            "FRZ-FRZ-05,FRZ-FRZ-06,FRZ-FRZ-07,FRZ-FRZ-08,FRZ-FRZ-09",
            "",
            "implementer+operator",
            "repo-wide freeze manifest",
            "[T-B7-PFP-ACTIVATE]",
            "Single commit FREEZE_CERTIFICATION_SHA (= IMPLEMENTATION_SHA); manifest freeze; no further product edits before L3 FREEZE (post-freeze policy active first)",
            "git rev-parse; freeze manifest",
            "FREEZE_CERTIFICATION_SHA",
            "SHA locked @ admission",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B7-L3-FREEZE",
            "B7",
            "FRZ-FRZ-03,FRZ-FRZ-12",
            "",
            "tester+auditor",
            "enterprise_audit",
            "[T-B7-LOCK-FREEZE-SHA]",
            "FREEZE mode @ FREEZE_CERTIFICATION_SHA",
            "run_enterprise_audit.py --mode FREEZE",
            "FREEZE report",
            "FREEZE PASS",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B7-VBN-HEAD-DOSSIER",
            "B7",
            "FRZ-RPL-01,FRZ-RPL-02,FRZ-RPL-03,FRZ-RPL-04",
            "FIND-VBN-PARTIAL",
            "implementer",
            "ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md; tests/qualification/ebh_5/",
            "[T-B6-OD-ENT-08]",
            "VBN §2.1–2.7 SSOT rollup; may prep @ closure baseline HEAD before lock; certifying acceptance remains post-lock",
            "pytest tests/qualification/ebh_5/ -q",
            "dossier §2.1–2.7",
            "prohibited_unresolved=0 frozen scope",
            "HIGH",
            "no",
            "PLANNED",
        ),
        (
            "T-B7-VBN-INDEPENDENT-ACCEPT",
            "B7",
            "FRZ-RPL-01,FRZ-RPL-02,FRZ-RPL-03,FRZ-RPL-04",
            "FIND-VBN-PARTIAL",
            "independent_auditor",
            "ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md",
            "[T-B7-VBN-HEAD-DOSSIER,T-B7-LOCK-FREEZE-SHA]",
            "Independent replay ENT VBN 12/12 + dossier §2.1–2.7 @ exact FREEZE_SHA; FIND-VBN-PARTIAL disposition",
            "pytest tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_*.py -q; ENT_AUDIT_X_SEMANTIC=1",
            "signed dossier §2.7",
            "independent sign @ FREEZE_SHA",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B7-ECAP-2C",
            "B7",
            "FRZ-FRZ-12",
            "",
            "independent_auditor",
            "ECAP",
            "[T-B7-L3-FREEZE]",
            "ECAP §2.C @ freeze SHA",
            "certifying replay",
            "§2.C dossier",
            "independent sign-off",
            "CRIT",
            "no",
            "BLOCKED_EXTERNAL",
        ),
        (
            "T-B7-PFP-ACTIVATE",
            "B7",
            "FRZ-FRZ-10",
            "",
            "operator",
            "ECAP §16; CHANGED-SINCE policy",
            "[T-B6-OD-ENT-08]",
            "Pre-lock: implement and activate post-freeze change policy before FREEZE_SHA lock",
            "run_enterprise_audit.py --mode CHANGED-SINCE (dry-run); hooks doc",
            "policy activation record",
            "post-freeze policy active",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B7-PFP-VERIFY",
            "B7",
            "FRZ-FRZ-10",
            "",
            "operator+auditor",
            "ECAP §16",
            "[T-B7-PFP-ACTIVATE,T-B7-L3-FREEZE]",
            "Pre-closure: verify FRZ-FRZ-10 enforcement @ FREEZE_SHA before T-B7-FRZ-FRZ",
            "certifying CHANGED-SINCE smoke @ freeze SHA",
            "FRZ-FRZ-10 evidence dossier",
            "policy verified active",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B7-TEN-REVAL",
            "B7",
            ten,
            "",
            "independent_auditor",
            "TENANT_X qual",
            "[T-B7-L3-FREEZE]",
            "Revalidate FRZ-TEN @ freeze SHA",
            "ENT tenant domain checks",
            "TENANT_X replay",
            "12/12 confirmed",
            "HIGH",
            "no",
            "PLANNED",
        ),
        (
            "T-PATHB-EXCL",
            "B7",
            "FRZ-PRD-01,FRZ-PRD-02",
            "FIND-PATHB-LIVE",
            "operator",
            "checklist Path B",
            "[T-B7-L3-FREEZE]",
            "Document Path B permitted OPEN @ freeze (no implicit global PASS)",
            "—",
            "Path B table",
            "exclusions recorded",
            "LOW",
            "yes",
            "PLANNED",
        ),
        (
            "T-B7-CMP-REVAL",
            "B7",
            "FRZ-CMP-01,FRZ-CMP-02,FRZ-CMP-03,FRZ-CMP-04,FRZ-CMP-05,FRZ-CMP-06,FRZ-CMP-07,FRZ-CMP-08",
            "FIND-CW-GATE-FAIL",
            "independent_auditor",
            "COMPAT-X",
            "[T-B1-STRICT-QUALIFY,T-B7-L3-FREEZE]",
            "Revalidate FRZ-CMP @ FREEZE SHA if B1 digest differs from historical PASS",
            "uv run pytest tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py",
            "CMP reval dossier",
            "CMP evidence @ FREEZE_SHA",
            "MED",
            "no",
            "PLANNED",
        ),
        (
            "T-B7-PROMOTE-RPL",
            "B7",
            rpl,
            "",
            "operator+auditor",
            "checklist",
            "[T-B7-L3-FREEZE,T-B7-VBN-INDEPENDENT-ACCEPT]",
            "Promote FRZ-RPL after VBN independent acceptance",
            "ENT VBN checks",
            "checklist promotion",
            "4 criteria",
            "MED",
            "no",
            "PLANNED",
        ),
        (
            "T-B7-PROMOTE-DEBT",
            "B7",
            debt,
            "",
            "operator+auditor",
            "checklist",
            "[T-B7-L3-FREEZE,T-B2-DEBT]",
            "Promote FRZ-DEBT after debt zero",
            "ENT debt checks",
            "checklist promotion",
            "6 criteria",
            "MED",
            "no",
            "PLANNED",
        ),
        (
            "T-B7-PROMOTE-DOC",
            "B7",
            doc,
            "",
            "operator+auditor",
            "checklist",
            "[T-B7-L3-FREEZE,T-B3-DOC]",
            "Promote FRZ-DOC after doc alignment",
            "ENT-A-DOC checks",
            "checklist promotion",
            "6 criteria",
            "MED",
            "yes",
            "PLANNED",
        ),
        (
            "T-B7-COORD-ARCH-FRZ",
            "B7",
            "FRZ-FRZ-02",
            "",
            "operator+auditor",
            "checklist",
            _coord_arch_frz_deps(),
            "Architecture-scoped FRZ PASS/N/A coordination (excl. Path B permitted OPEN)",
            "ENT FREEZE + promotion ledger",
            "FRZ-FRZ-02 satisfied",
            "arch-scoped OPEN=0 or N/A w/ evidence",
            "HIGH",
            "partial",
            "PLANNED",
        ),
        (
            "T-B7-FRZ-FRZ",
            "B7",
            frz,
            "",
            "operator+auditor",
            "freeze dossier",
            "[T-B7-ECAP-2C,T-B7-COORD-ARCH-FRZ,T-B7-PFP-VERIFY]",
            "ECAP §15 admission + promote FRZ-FRZ suite (authorized only)",
            "FRZ-FRZ suite",
            "admission dossier",
            "13/13 PASS/N/A",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B8-ROADMAP",
            "B8",
            "FRZ-FRZ-04",
            "",
            "operator",
            "PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md §5",
            "[T-B7-FRZ-FRZ]",
            "Publish ARCH-FREEZE CLOSED record",
            "—",
            "FORMAL_CLOSURE_PUBLICATION_SHA",
            "roadmap §5 updated",
            "CRIT",
            "no",
            "PLANNED",
        ),
        (
            "T-B8-ECAP-16",
            "B8",
            "",
            "",
            "operator",
            "ECAP §16",
            "[T-B8-ROADMAP]",
            "Post-closure: ECAP §16 monitoring and operator handoff (policy already active)",
            "ENT CHANGED-SINCE ongoing runs",
            "§16 handoff record",
            "monitoring live",
            "MED",
            "yes",
            "PLANNED",
        ),
    ]
    for fam in ("BND", "CTR", "OBS", "OWN", "PLG", "PRD", "REG", "REL", "SEC", "TYP"):
        ids = list(by_fam[fam])
        if fam == "PRD":
            ids = [i for i in ids if i not in PATH_B_FULL]
        rows.append(
            (
                f"T-B7-PROMOTE-{fam}",
                "B7",
                _frz_csv(ids),
                "",
                "operator+auditor",
                "checklist",
                "[T-B7-L3-FREEZE]",
                (
                    f"Promote architecture-scoped OPEN {fam} FRZ rows after FREEZE audit"
                    if fam == "PRD"
                    else f"Promote OPEN {fam} FRZ rows after FREEZE audit"
                ),
                f"ENT-A-{fam[:3]} checks",
                "checklist promotion",
                (f"{len(ids)} criteria; Path B split" if fam == "PRD" else f"{len(by_fam[fam])} criteria"),
                "MED",
                "partial",
                "PLANNED",
            )
        )
    if len(rows) != EXPECTED_WBS_TASKS:
        raise ValueError(f"WBS row count {len(rows)} != {EXPECTED_WBS_TASKS}")
    hdr = "| TASK_ID | BATCH | REQUIREMENT_IDS | FINDING_IDS | OWNER | FILES | DEPENDENCIES | IMPLEMENTATION_ACTION | TESTS | EVIDENCE | ACCEPTANCE | RISK | PARALLELISM | STATUS |"
    lines = [
        "# ARCH-FREEZE WBS and dependency DAG",
        "",
        f"**AUDIT_BASELINE_SHA:** `{AUDIT_BASELINE_SHA}`",
        f"**Task count:** {EXPECTED_WBS_TASKS} (unique TASK_ID rows below)",
        "",
        "**Wave 1 (T-IGX-010-B):** **B0–B4** = **INDEPENDENTLY ACCEPTED (SCOPED)** @ G6 AUDIT `3d96691f1b02b393a4de189139a6e1347db99382` / closure doc `950097ea4bcfd5bba828f663bcd993b8bd1e1d63`; B1 `6bdf78c626bfcb347ad3ad9e3e18b65222f1b169`; B2–B4 `f7ce9daebad35537da974de0dda3962126c6030e`; L3 FULL 85/85; B1 digest `0ffa2676…`. **R0** = **COMPLETE** @ `950097ea4bcfd5bba828f663bcd993b8bd1e1d63` (see the R0 task rows); **R1–R6** = **PLANNED** (`T-IGX-010-C`). Handoff: [`ARCH_FREEZE_EXECUTION_WAVE1_INDEPENDENT_AUDIT_HANDOFF.md`](./ARCH_FREEZE_EXECUTION_WAVE1_INDEPENDENT_AUDIT_HANDOFF.md).",
        "",
        hdr,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(*r))
    lines += [
        "",
        "## Critical path",
        "T-R0-WAVE1-SSOT-RECONCILE → T-R0-BASELINE-CLOSURE-HEAD → T-B0-TRACE-VALIDATE → T-B1-INV → T-B2-* → T-B4-L3-FULL → T-B5-EVOL-B6 → T-B5-EVOL-FRZ → T-B5-EVOL-FORMAL-CLOSURE → T-B6-OD-ENT-00 → T-B6-ENT-B6 → T-B6-OD-ENT-01/02 → T-B6-FRZ-AUD → T-B6-OD-ENT-08 → T-B7-PFP-ACTIVATE → T-B7-LOCK-FREEZE-SHA → T-B7-L3-FREEZE → T-B7-VBN-HEAD-DOSSIER → T-B7-VBN-INDEPENDENT-ACCEPT → (promotions + T-PATHB-EXCL) → T-B7-COORD-ARCH-FRZ → T-B7-ECAP-2C → T-B7-PFP-VERIFY → T-B7-FRZ-FRZ → T-B8-ROADMAP → T-B8-ECAP-16",
    ]
    _WBS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_matrix_open_tasks() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in _MATRIX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| FRZ-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 12:
            continue
        fid, st, task = parts[1], parts[2], parts[9]
        if st == "OPEN":
            mapping[fid] = task
    return mapping


def _parse_wbs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in _WBS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| T-"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 14:
            continue
        rows.append(
            {
                "TASK_ID": parts[0],
                "REQUIREMENT_IDS": parts[2],
                "FINDING_IDS": parts[3],
                "DEPENDENCIES": parts[6],
            }
        )
    return rows


def _parse_findings() -> set[str]:
    found: set[str] = set()
    for line in _GAPS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (FIND-[A-Z0-9-]+) \|", line)
        if m:
            found.add(m.group(1))
    return found


def _parse_gap_correction_tasks() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for line in _GAPS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (FIND-[A-Z0-9-]+) \|", line)
        if not m:
            continue
        fid = m.group(1)
        for task in re.findall(r"T-[A-Z0-9-]+", line):
            mapping[fid].add(task)
    return mapping


def _extract_frz_ids(raw: str) -> list[str]:
    return re.findall(r"FRZ-[A-Z]+-\d+", raw)


def _extract_task_deps(raw: str) -> list[str]:
    return re.findall(r"T-[A-Z0-9-]+", raw)


def _extract_findings(raw: str) -> list[str]:
    return re.findall(r"FIND-[A-Z0-9-]+", raw)


def validate(*, strict_counts: bool = True) -> int:
    errors: list[str] = []
    statuses = _parse_checklist()
    canonical = set(statuses)
    if len(canonical) != 225:
        errors.append(f"FRZ count expected 225 got {len(canonical)}")
    families = {fid.split("-")[1] for fid in canonical}
    if len(families) != 26:
        errors.append(f"FRZ families expected 26 got {len(families)}")
    pass_n = sum(1 for s in statuses.values() if s == "PASS")
    open_n = sum(1 for s in statuses.values() if s == "OPEN")
    if strict_counts:
        if pass_n != 108:
            errors.append(f"PASS expected 108 got {pass_n}")
        if open_n != 109:
            errors.append(f"OPEN expected 109 got {open_n}")

    wbs_rows = _parse_wbs()
    task_ids = {r["TASK_ID"] for r in wbs_rows}
    if len(task_ids) != len(wbs_rows):
        errors.append("duplicate WBS TASK_ID")
    if len(task_ids) != EXPECTED_WBS_TASKS:
        errors.append(f"WBS task count expected {EXPECTED_WBS_TASKS} got {len(task_ids)}")

    findings = _parse_findings()
    gap_tasks = _parse_gap_correction_tasks()
    finding_wbs: dict[str, set[str]] = defaultdict(set)
    for row in wbs_rows:
        for finding in _extract_findings(row["FINDING_IDS"]):
            finding_wbs[finding].add(row["TASK_ID"])
    for fid, tasks in gap_tasks.items():
        finding_wbs[fid].update(tasks)

    open_tasks = _parse_matrix_open_tasks()
    dangling = [f"{fid}->{task}" for fid, task in open_tasks.items() if task not in task_ids and task != "—"]
    if dangling:
        errors.append(f"dangling OPEN FRZ remediation tasks: {len(dangling)} (e.g. {dangling[0]})")

    for row in wbs_rows:
        tid = row["TASK_ID"]
        req = row["REQUIREMENT_IDS"]
        if "…" in req or "OPEN rows" in req or "*" in req:
            errors.append(f"{tid} placeholder in REQUIREMENT_IDS")
        for frz_id in _extract_frz_ids(req):
            if frz_id not in canonical:
                errors.append(f"{tid} unknown FRZ {frz_id}")
        for finding in _extract_findings(row["FINDING_IDS"]):
            if finding not in findings:
                errors.append(f"{tid} unknown finding {finding}")

    for finding in findings:
        if finding not in finding_wbs:
            errors.append(f"finding {finding} not referenced in WBS FINDING_IDS or gap CORRECTION")

    for bf in BLOCKER_FINDINGS:
        if bf not in findings:
            errors.append(f"missing BLOCKER finding {bf}")
        elif not finding_wbs.get(bf):
            errors.append(f"BLOCKER {bf} has no mapped WBS/gap task")

    graph: dict[str, list[str]] = {}
    for row in wbs_rows:
        graph[row["TASK_ID"]] = _extract_task_deps(row["DEPENDENCIES"])
        for dep in graph[row["TASK_ID"]]:
            if dep not in task_ids:
                errors.append(f"{row['TASK_ID']} missing dependency {dep}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> None:
        if node in visiting:
            errors.append(f"dependency cycle involving {node}")
            return
        if node in visited:
            return
        visiting.add(node)
        for nxt in graph.get(node, []):
            dfs(nxt)
        visiting.remove(node)
        visited.add(node)

    for tid in task_ids:
        dfs(tid)

    _validate_semantic_dag(graph, task_ids, errors)

    for row in wbs_rows:
        if row["TASK_ID"] == "T-B7-PROMOTE-PRD":
            prd_ids = set(_extract_frz_ids(row["REQUIREMENT_IDS"]))
            overlap = prd_ids & PATH_B_FULL
            if overlap:
                errors.append(f"T-B7-PROMOTE-PRD must not include Path B FRZ {sorted(overlap)} (use T-PATHB-EXCL)")
        if row["TASK_ID"] == "T-PATHB-EXCL":
            prd_ids = set(_extract_frz_ids(row["REQUIREMENT_IDS"]))
            if not PATH_B_FULL.issubset(prd_ids):
                errors.append("T-PATHB-EXCL must document FRZ-PRD-01 and FRZ-PRD-02")

    qual_text = _QUAL_PLAN.read_text(encoding="utf-8")
    if not _IIAC_PROTOCOL.is_file():
        errors.append("missing IIAC protocol (ARCH-FREEZE predecessor readiness G7)")
    if _IIAC_ACTIVATION.is_file():
        import json

        act = json.loads(_IIAC_ACTIVATION.read_text(encoding="utf-8"))
        if str(act.get("status") or "").upper() == "IN_FORCE":
            if not act.get("effective_from_sha"):
                errors.append("IIAC IN_FORCE without effective_from_sha in activation record")
    pre_admission = _QUAL / "ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md"
    if pre_admission.is_file():
        pa = pre_admission.read_text(encoding="utf-8")
        if "IIAC" not in pa and "INTERNAL AGENT CERTIFICATION" not in pa:
            errors.append("pre-admission report must cite IIAC predecessor fields (G7)")
    run_ids = re.findall(r"^\| (L[123]-[A-Z0-9-]+) \|", qual_text, re.M)
    if len(run_ids) != len(set(run_ids)):
        errors.append("duplicate qualification RUN_ID")
    if len(run_ids) != EXPECTED_QUAL_RUNS:
        errors.append(f"qual RUN_ID count expected {EXPECTED_QUAL_RUNS} got {len(run_ids)}")
    if re.search(r"^L[123]-", qual_text, re.M):
        errors.append("qual plan contains non-table RUN rows")

    for rel in (
        "scripts/governance/verify_arch_freeze_plan_traceability.py",
        "scripts/governance/verify_iiac_policy_activation.py",
        "docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md",
        "scripts/enterprise_audit/run_enterprise_audit.py",
        "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py",
        "tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py",
        "tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py",
        "tests/unit/governance/test_arch_freeze_b1_closed_world_inventory.py",
        "tests/qualification/ent_audit_x/acceptance",
    ):
        if not (_REPO / rel).exists():
            errors.append(f"missing command path {rel}")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        print(f"[FAIL] {len(errors)} traceability violation(s)", file=sys.stderr)
        return 1
    print("[OK] verify_arch_freeze_plan_traceability: all checks passed")
    print(
        f"  FRZ=225 families=26 PASS={pass_n} OPEN={open_n} "
        f"WBS={len(task_ids)} findings={len(findings)} dangling_open_frz=0 semantic_dag=OK"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-matrix", action="store_true")
    parser.add_argument("--sync-wbs", action="store_true")
    args = parser.parse_args()
    if args.sync_matrix:
        sync_matrix()
    if args.sync_wbs:
        sync_wbs()
    return validate()


if __name__ == "__main__":
    raise SystemExit(main())
