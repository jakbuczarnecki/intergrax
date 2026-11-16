#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501
"""Mechanical traceability validator for ARCH-FREEZE planning dossier artifacts.

Amended by T-AF2-VALIDATOR-AMEND (ticket T-IGX-050-B, MEP-AF2 v2.1 section C items 1-4 and C.4):

* every hard DAG rule is kept unchanged; two additive rules and a DAG self-test were added;
* counters are computed from the files instead of being hard-coded;
* ``--sync-wbs`` / ``--sync-matrix`` are forbidden (they rewrote rows from hard-coded data).
"""

from __future__ import annotations

import argparse
import json
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
# FIND-PROD-REACH-DEFERRED is historical (superseded by the B1 chain) and is severity INFO, not a blocker.
BLOCKER_FINDINGS = frozenset(
    {
        "FIND-PROD-REACH-CLASSIFICATION",
        "FIND-VBN-PARTIAL",
        "FIND-FORMAL-B6",
        "FIND-ACCEPTED-BASELINE",
    }
)
INFO_FINDINGS = frozenset({"FIND-PROD-REACH-DEFERRED"})
# Historical WBS tasks (the original 54 rows). The WBS total is this number plus the T-AF2-* / PT-* rows.
BASE_WBS_TASKS = 54
EXTENSION_TASK_PREFIXES = ("T-AF2-", "PT-")

# Word boundary: "PT-Q11" is one ID and is never read as "T-Q11"; "ENT-B6" never yields "T-B6".
_TASK_ID_RE = re.compile(r"\b(?:T|PT)-[A-Z0-9-]+")
_MATRIX_CROSSCHECK_FLAG_RE = re.compile(r"^\*\*MATRIX_CHECKLIST_CROSSCHECK:\*\*\s*ENABLED\b", re.M)
_MATRIX_HEADER_COUNTS_RE = re.compile(r"\*\*Rows:\*\*\s*(\d+)\s*\|\s*\*\*Families:\*\*\s*(\d+)")
_AUTHORITY_REF_RE = re.compile(r"authority_ref:\s*(.+)$")
SUPERSEDED_STATE = "SUPERSEDED_BY_CURRENT_AUTHORITY"

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
# Pre-lock validator ancestors: they stay before the lock and never follow ECAP-2C [PD2-01].
PRE_LOCK_ANCESTOR_TASKS = ("T-B5-EVOL-FRZ", "T-B6-FRZ-AUD")
POST_LOCK_TASKS = ("T-B7-ECAP-2C", "T-B7-LOCK-FREEZE-SHA")


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------


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


def _status_class(status: str) -> str:
    """Normalise a checklist/matrix status: OPEN, PASS (any PASS variant) or the raw text."""
    s = status.strip()
    if s == "OPEN":
        return "OPEN"
    if s.startswith("PASS"):
        return "PASS"
    return s


def _extract_task_ids(raw: str) -> list[str]:
    return [m.rstrip("-") for m in _TASK_ID_RE.findall(raw)]


def _is_extension_task(task_id: str) -> bool:
    return task_id.startswith(EXTENSION_TASK_PREFIXES)


def _parse_matrix_rows() -> dict[str, tuple[str, str]]:
    """FRZ_ID -> (CURRENT_STATUS, REMEDIATION_TASK) for every matrix row."""
    rows: dict[str, tuple[str, str]] = {}
    for line in _MATRIX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| FRZ-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 12:
            continue
        rows[parts[1]] = (parts[2], parts[9])
    return rows


def _parse_matrix_open_tasks() -> dict[str, str]:
    return {fid: task for fid, (st, task) in _parse_matrix_rows().items() if st == "OPEN"}


def _parse_wbs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in _WBS.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("| T-", "| PT-")):
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
                "STATUS": parts[13],
                "V21_QUALIFICATION": parts[14] if len(parts) > 14 else "",
            }
        )
    return rows


def _parse_findings() -> dict[str, str]:
    """FIND ID -> severity column of the gap register."""
    found: dict[str, str] = {}
    for line in _GAPS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (FIND-[A-Z0-9-]+) \|", line)
        if m:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            found[m.group(1)] = parts[1] if len(parts) > 1 else ""
    return found


def _parse_gap_correction_tasks() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for line in _GAPS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (FIND-[A-Z0-9-]+) \|", line)
        if not m:
            continue
        mapping[m.group(1)].update(_extract_task_ids(line))
    return mapping


def _extract_frz_ids(raw: str) -> list[str]:
    return re.findall(r"FRZ-[A-Z]+-\d+", raw)


def _extract_task_deps(raw: str) -> list[str]:
    return _extract_task_ids(raw)


def _extract_findings(raw: str) -> list[str]:
    return re.findall(r"FIND-[A-Z0-9-]+", raw)


def _qualification_state(cell: str) -> tuple[str, str]:
    """Split a V21_QUALIFICATION cell into (state, authority_ref)."""
    text = cell.strip()
    m = re.match(r"[A-Z][A-Z_]+", text)
    state = m.group(0) if m else ""
    ref = _AUTHORITY_REF_RE.search(text)
    return state, (ref.group(1).strip() if ref else "")


def _check_qualification_states(wbs_rows: list[dict[str, str]], errors: list[str]) -> set[str]:
    """SUPERSEDED_BY_CURRENT_AUTHORITY is a satisfied dependency, but only with a non-empty authority_ref.

    Returns the set of TASK_IDs that are satisfied by current authority. Such tasks stay DAG nodes:
    every edge through them still counts for reachability (no pruning), so hard rules hold unchanged.
    """
    satisfied: set[str] = set()
    for row in wbs_rows:
        state, ref = _qualification_state(row.get("V21_QUALIFICATION", ""))
        if state != SUPERSEDED_STATE:
            continue
        if not ref or ref in {"—", "-", "TBD", "TODO", "N/A"}:
            errors.append(f"{row['TASK_ID']}: {SUPERSEDED_STATE} requires a non-empty authority_ref")
            continue
        satisfied.add(row["TASK_ID"])
    return satisfied


# --------------------------------------------------------------------------------------
# DAG rules (hard rules unchanged; see _validate_semantic_dag)
# --------------------------------------------------------------------------------------


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


def _find_cycles(graph: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> None:
        if node in visiting:
            msg = f"dependency cycle involving {node}"
            if msg not in errors:
                errors.append(msg)
            return
        if node in visited:
            return
        visiting.add(node)
        for nxt in graph.get(node, []):
            dfs(nxt)
        visiting.remove(node)
        visited.add(node)

    for tid in list(graph):
        dfs(tid)
    return errors


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

    # Additive rule [PD2-01]: the pre-lock ancestors are never placed after ECAP-2C / LOCK.
    # (ECAP-2C is an ancestor of COORD, and the EVOL chain is an ancestor of LOCK, so such an edge is also a cycle.)
    for pre in PRE_LOCK_ANCESTOR_TASKS:
        for post in POST_LOCK_TASKS:
            if post in _ancestors(graph, pre):
                errors.append(f"PD2-01: {pre} is a pre-lock ancestor and must not follow {post}")


# --------------------------------------------------------------------------------------
# DAG self-test: a reference graph, one mutation per rule, all must be rejected
# --------------------------------------------------------------------------------------


def _reference_graph() -> dict[str, list[str]]:
    """Minimal DAG that satisfies every hard rule (the shape the real WBS must keep)."""
    g: dict[str, list[str]] = {
        "T-B4-L3-FULL": [],
        "T-B5-EVOL-B6": ["T-B4-L3-FULL"],
        "T-B5-EVOL-FRZ": ["T-B5-EVOL-B6"],
        "T-B5-EVOL-FORMAL-CLOSURE": ["T-B5-EVOL-B6", "T-B5-EVOL-FRZ"],
        "T-B6-OD-ENT-00": ["T-B5-EVOL-FORMAL-CLOSURE"],
        "T-B6-ENT-B6": ["T-B6-OD-ENT-00"],
        "T-B6-OD-ENT-02": ["T-B6-ENT-B6"],
        "T-B6-FRZ-AUD": ["T-B6-ENT-B6"],
        "T-B6-OD-ENT-08": ["T-B6-OD-ENT-02", "T-B6-FRZ-AUD"],
        "T-B7-PFP-ACTIVATE": ["T-B6-OD-ENT-08"],
        "T-B7-LOCK-FREEZE-SHA": ["T-B7-PFP-ACTIVATE"],
        "T-B7-L3-FREEZE": ["T-B7-LOCK-FREEZE-SHA"],
        "T-B7-VBN-HEAD-DOSSIER": ["T-B4-L3-FULL"],
        "T-B7-VBN-INDEPENDENT-ACCEPT": ["T-B7-LOCK-FREEZE-SHA", "T-B7-L3-FREEZE", "T-B7-VBN-HEAD-DOSSIER"],
        "T-B7-ECAP-2C": ["T-B7-L3-FREEZE"],
        "T-B7-PFP-VERIFY": ["T-B7-PFP-ACTIVATE", "T-B7-L3-FREEZE"],
        "T-B2-DEBT": ["T-B4-L3-FULL"],
        "T-B3-DOC": ["T-B4-L3-FULL"],
        "T-B7-TEN-REVAL": ["T-B7-L3-FREEZE"],
        "T-B7-CMP-REVAL": ["T-B7-L3-FREEZE"],
        "T-PATHB-EXCL": ["T-B4-L3-FULL"],
        "T-B7-PROMOTE-RPL": ["T-B7-VBN-INDEPENDENT-ACCEPT"],
        "T-B7-PROMOTE-DEBT": ["T-B7-ECAP-2C"],
        "T-B7-PROMOTE-DOC": ["T-B7-ECAP-2C"],
    }
    for task in PROMOTE_TASKS:
        g[task] = ["T-B7-ECAP-2C"]
    g["T-B7-COORD-ARCH-FRZ"] = list(FRZ_FRZ02_PREREQ_TASKS)
    g["T-B7-FRZ-FRZ"] = ["T-B7-COORD-ARCH-FRZ", "T-B7-ECAP-2C", "T-B7-PFP-VERIFY"]
    g["T-B8-ROADMAP"] = ["T-B7-FRZ-FRZ", "T-B7-PFP-VERIFY", "T-B7-COORD-ARCH-FRZ"]
    g["T-B8-ECAP-16"] = ["T-B8-ROADMAP"]
    return g


def _dag_errors(graph: dict[str, list[str]]) -> list[str]:
    errors = _find_cycles(graph)
    _validate_semantic_dag(graph, set(graph), errors)
    return errors


def _copy_graph(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    return {k: list(v) for k, v in graph.items()}


def _without_edge(graph: dict[str, list[str]], node: str, dep: str) -> dict[str, list[str]]:
    g = _copy_graph(graph)
    g[node] = [d for d in g[node] if d != dep]
    return g


def _isolated(graph: dict[str, list[str]], node: str) -> dict[str, list[str]]:
    """Cut every edge that points at ``node`` so that it is no longer an ancestor of anything."""
    return {k: [d for d in v if d != node] for k, v in graph.items()}


def _with_edge(graph: dict[str, list[str]], node: str, dep: str) -> dict[str, list[str]]:
    g = _copy_graph(graph)
    g[node].append(dep)
    return g


def _self_test_cases() -> list[tuple[str, dict[str, list[str]], str]]:
    """(label, mutated graph, substring that must appear in the errors)."""
    ref = _reference_graph()
    cases: list[tuple[str, dict[str, list[str]], str]] = []
    # The 20 COORD prerequisites: a prerequisite that no longer reaches COORD (by any path) must be rejected.
    for prereq in FRZ_FRZ02_PREREQ_TASKS:
        cases.append(
            (
                f"COORD without {prereq}",
                _isolated(ref, prereq),
                f"T-B7-COORD-ARCH-FRZ must follow {prereq}",
            )
        )
    hard = (
        (
            "OD-ENT-08 without OD-ENT-02",
            _without_edge(ref, "T-B6-OD-ENT-08", "T-B6-OD-ENT-02"),
            "OD-ENT-08 must follow T-B6-OD-ENT-02",
        ),
        (
            "PFP-ACTIVATE without OD-ENT-08",
            _without_edge(ref, "T-B7-PFP-ACTIVATE", "T-B6-OD-ENT-08"),
            "T-B7-PFP-ACTIVATE must follow T-B6-OD-ENT-08",
        ),
        (
            "LOCK without PFP-ACTIVATE",
            _without_edge(ref, "T-B7-LOCK-FREEZE-SHA", "T-B7-PFP-ACTIVATE"),
            "T-B7-LOCK-FREEZE-SHA must follow T-B7-PFP-ACTIVATE",
        ),
        (
            "VBN-ACCEPT without LOCK",
            _without_edge(
                _without_edge(ref, "T-B7-VBN-INDEPENDENT-ACCEPT", "T-B7-LOCK-FREEZE-SHA"),
                "T-B7-VBN-INDEPENDENT-ACCEPT",
                "T-B7-L3-FREEZE",
            ),
            "T-B7-VBN-INDEPENDENT-ACCEPT must follow T-B7-LOCK-FREEZE-SHA",
        ),
        (
            "VBN-HEAD depends on LOCK",
            _with_edge(ref, "T-B7-VBN-HEAD-DOSSIER", "T-B7-LOCK-FREEZE-SHA"),
            "T-B7-VBN-HEAD-DOSSIER must not require T-B7-LOCK-FREEZE-SHA",
        ),
        (
            "PROMOTE-RPL without VBN-ACCEPT",
            _without_edge(ref, "T-B7-PROMOTE-RPL", "T-B7-VBN-INDEPENDENT-ACCEPT"),
            "T-B7-PROMOTE-RPL must follow T-B7-VBN-INDEPENDENT-ACCEPT",
        ),
        (
            "EVOL chain broken (FORMAL-CLOSURE without EVOL-FRZ)",
            _without_edge(ref, "T-B5-EVOL-FORMAL-CLOSURE", "T-B5-EVOL-FRZ"),
            "T-B5-EVOL-FORMAL-CLOSURE must follow T-B5-EVOL-FRZ",
        ),
        (
            "EVOL chain broken (OD-ENT-00 without FORMAL-CLOSURE)",
            _without_edge(ref, "T-B6-OD-ENT-00", "T-B5-EVOL-FORMAL-CLOSURE"),
            "OD-ENT-00 must follow T-B5-EVOL-FORMAL-CLOSURE",
        ),
        (
            "PFP-ACTIVATE after L3-FREEZE",
            _with_edge(ref, "T-B7-PFP-ACTIVATE", "T-B7-L3-FREEZE"),
            "T-B7-PFP-ACTIVATE must precede T-B7-L3-FREEZE",
        ),
        (
            "L3-FREEZE without LOCK",
            _without_edge(ref, "T-B7-L3-FREEZE", "T-B7-LOCK-FREEZE-SHA"),
            "T-B7-L3-FREEZE must follow T-B7-LOCK-FREEZE-SHA",
        ),
        (
            "PFP-VERIFY without L3-FREEZE",
            _without_edge(ref, "T-B7-PFP-VERIFY", "T-B7-L3-FREEZE"),
            "T-B7-PFP-VERIFY must follow T-B7-L3-FREEZE",
        ),
        ("FRZ-FRZ without ECAP-2C", _isolated(ref, "T-B7-ECAP-2C"), "T-B7-FRZ-FRZ must follow T-B7-ECAP-2C"),
        (
            "ROADMAP without FRZ-FRZ",
            _without_edge(ref, "T-B8-ROADMAP", "T-B7-FRZ-FRZ"),
            "T-B8-ROADMAP must follow T-B7-FRZ-FRZ",
        ),
        (
            "ECAP-16 without ROADMAP",
            _without_edge(ref, "T-B8-ECAP-16", "T-B8-ROADMAP"),
            "T-B8-ECAP-16 must follow T-B8-ROADMAP",
        ),
        (
            "Path B: PATHB-EXCL after FRZ-FRZ",
            _with_edge(ref, "T-PATHB-EXCL", "T-B7-FRZ-FRZ"),
            "T-PATHB-EXCL must not depend on T-B7-FRZ-FRZ",
        ),
        ("cycle", _with_edge(ref, "T-B4-L3-FULL", "T-B8-ECAP-16"), "dependency cycle involving"),
        # PD2-01 fixture: the edge T-B5-EVOL-FRZ <- T-B7-ECAP-2C (EVOL-FRZ placed after ECAP-2C) must be rejected.
        (
            "PD2-01: EVOL-FRZ <- ECAP-2C",
            _with_edge(ref, "T-B5-EVOL-FRZ", "T-B7-ECAP-2C"),
            "PD2-01: T-B5-EVOL-FRZ is a pre-lock ancestor and must not follow T-B7-ECAP-2C",
        ),
        (
            "PD2-01: FRZ-AUD <- ECAP-2C",
            _with_edge(ref, "T-B6-FRZ-AUD", "T-B7-ECAP-2C"),
            "PD2-01: T-B6-FRZ-AUD is a pre-lock ancestor and must not follow T-B7-ECAP-2C",
        ),
    )
    cases.extend(hard)
    return cases


def dag_self_test() -> list[str]:
    """Prove the DAG rules themselves work. Returns a list of self-test failures (empty = pass)."""
    failures: list[str] = []
    ref = _reference_graph()
    if len(FRZ_FRZ02_PREREQ_TASKS) != 20 or len(set(FRZ_FRZ02_PREREQ_TASKS)) != 20:
        failures.append(f"self-test: COORD must have 20 distinct prerequisites, got {len(FRZ_FRZ02_PREREQ_TASKS)}")
    if _find_cycles(ref):
        failures.append("self-test: reference graph has a cycle")
    base_errors = _dag_errors(ref)
    if base_errors:
        failures.append(f"self-test: reference graph violates rules: {base_errors[:3]}")
    for label, graph, expected in _self_test_cases():
        errs = _dag_errors(graph)
        if not any(expected in e for e in errs):
            failures.append(f"self-test: mutation '{label}' was not rejected (expected '{expected}')")
    # SUPERSEDED_BY_CURRENT_AUTHORITY: satisfied only with an authority_ref.
    ok_rows = [{"TASK_ID": "T-X", "V21_QUALIFICATION": f"{SUPERSEDED_STATE} - authority_ref: abc123"}]
    bad_rows = [{"TASK_ID": "T-X", "V21_QUALIFICATION": SUPERSEDED_STATE}]
    ok_errs: list[str] = []
    bad_errs: list[str] = []
    if _check_qualification_states(ok_rows, ok_errs) != {"T-X"} or ok_errs:
        failures.append("self-test: SUPERSEDED row with authority_ref was not accepted")
    if _check_qualification_states(bad_rows, bad_errs) or not bad_errs:
        failures.append("self-test: SUPERSEDED row without authority_ref was not rejected")
    return failures


# --------------------------------------------------------------------------------------
# Matrix vs checklist cross-check (opt-in via the matrix header flag from T-AF2-SSOT-RECONCILE-B)
# --------------------------------------------------------------------------------------


def _matrix_crosscheck_enabled() -> bool:
    return bool(_MATRIX_CROSSCHECK_FLAG_RE.search(_MATRIX.read_text(encoding="utf-8")))


def _matrix_crosscheck(statuses: dict[str, str], errors: list[str]) -> None:
    matrix = _parse_matrix_rows()
    for fid in sorted(set(statuses) - set(matrix)):
        errors.append(f"matrix cross-check: {fid} is in the checklist but not in the matrix")
    for fid in sorted(set(matrix) - set(statuses)):
        errors.append(f"matrix cross-check: {fid} is in the matrix but not in the checklist")
    for fid in sorted(set(statuses) & set(matrix)):
        if _status_class(matrix[fid][0]) != _status_class(statuses[fid]):
            errors.append(
                f"matrix cross-check: {fid} status differs (matrix '{matrix[fid][0]}' vs checklist '{statuses[fid]}')"
            )
    header = _MATRIX_HEADER_COUNTS_RE.search(_MATRIX.read_text(encoding="utf-8"))
    families = {fid.split("-")[1] for fid in statuses}
    if not header:
        errors.append("matrix cross-check: header '**Rows:** N | **Families:** M' not found")
    elif (int(header.group(1)), int(header.group(2))) != (len(statuses), len(families)):
        errors.append(
            f"matrix cross-check: header declares {header.group(1)} rows / {header.group(2)} families, "
            f"checklist has {len(statuses)} / {len(families)}"
        )


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def collect(errors: list[str] | None = None) -> tuple[list[str], dict[str, object]]:
    """Run every check. Returns (errors, computed counters)."""
    errors = [] if errors is None else errors
    stats: dict[str, object] = {}

    statuses = _parse_checklist()
    canonical = set(statuses)
    families = {fid.split("-")[1] for fid in canonical}
    if not canonical:
        errors.append("checklist has no FRZ rows")
    classes = defaultdict(int)
    for st in statuses.values():
        classes[_status_class(st)] += 1
    pass_n = classes.get("PASS", 0)
    classes_plain = sum(1 for st in statuses.values() if st == "PASS")
    open_n = classes.get("OPEN", 0)
    for cls in sorted(set(classes) - {"PASS", "OPEN"}):
        errors.append(f"checklist: unknown FRZ status '{cls}' ({classes[cls]} rows); expected OPEN or PASS*")
    stats.update(frz=len(canonical), families=len(families), passed=pass_n, passed_plain=classes_plain, open=open_n)

    crosscheck = _matrix_crosscheck_enabled()
    stats["matrix_crosscheck"] = "ENABLED" if crosscheck else "OFF"
    if crosscheck:
        _matrix_crosscheck(statuses, errors)

    wbs_rows = _parse_wbs()
    task_ids = {r["TASK_ID"] for r in wbs_rows}
    if len(task_ids) != len(wbs_rows):
        errors.append("duplicate WBS TASK_ID")
    extension_n = sum(1 for t in task_ids if _is_extension_task(t))
    base_n = len(task_ids) - extension_n
    if base_n != BASE_WBS_TASKS:
        errors.append(f"WBS historical task count expected {BASE_WBS_TASKS} got {base_n}")
    stats.update(wbs=len(task_ids), wbs_base=base_n, wbs_extension=extension_n)

    findings = _parse_findings()
    gap_tasks = _parse_gap_correction_tasks()
    finding_wbs: dict[str, set[str]] = defaultdict(set)
    for row in wbs_rows:
        for finding in _extract_findings(row["FINDING_IDS"]):
            finding_wbs[finding].add(row["TASK_ID"])
    for fid, tasks in gap_tasks.items():
        finding_wbs[fid].update(tasks)

    open_tasks = _parse_matrix_open_tasks()
    dangling: list[str] = []
    for fid, task in open_tasks.items():
        ids = _extract_task_ids(task)
        if ids:
            dangling.extend(f"{fid}->{t}" for t in ids if t not in task_ids)
        elif task != "—":
            dangling.append(f"{fid}->{task}")
    if dangling:
        errors.append(f"dangling OPEN FRZ remediation tasks: {len(dangling)} (e.g. {dangling[0]})")
    stats["dangling_open_frz"] = len(dangling)

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
    for inf in INFO_FINDINGS:
        if inf not in findings:
            errors.append(f"missing INFO finding {inf}")
        elif findings[inf].upper() != "INFO":
            errors.append(f"{inf} must have severity INFO in the gap register, got '{findings[inf]}'")

    satisfied = _check_qualification_states(wbs_rows, errors)
    stats["superseded_by_authority"] = len(satisfied)

    graph: dict[str, list[str]] = {}
    for row in wbs_rows:
        graph[row["TASK_ID"]] = _extract_task_deps(row["DEPENDENCIES"])
        for dep in graph[row["TASK_ID"]]:
            if dep not in task_ids:
                errors.append(f"{row['TASK_ID']} missing dependency {dep}")

    errors.extend(_find_cycles(graph))
    _validate_semantic_dag(graph, task_ids, errors)
    errors.extend(dag_self_test())

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
    if not run_ids:
        errors.append("qualification plan has no RUN_ID rows")
    stats["qual_runs"] = len(set(run_ids))
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

    stats["findings"] = len(findings)
    stats["info_findings"] = sorted(INFO_FINDINGS & set(findings))
    return errors, stats


def validate() -> int:
    errors, stats = collect()
    if errors:
        for err in errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        print(f"[FAIL] {len(errors)} traceability violation(s)", file=sys.stderr)
        return 1
    print("[OK] verify_arch_freeze_plan_traceability: all checks passed")
    print(
        f"  FRZ={stats['frz']} families={stats['families']} PASS={stats['passed']} (plain {stats['passed_plain']}) OPEN={stats['open']} "
        f"WBS={stats['wbs']} (historical {stats['wbs_base']} + extension {stats['wbs_extension']}) "
        f"findings={stats['findings']} qual_runs={stats['qual_runs']} "
        f"dangling_open_frz={stats['dangling_open_frz']} matrix_crosscheck={stats['matrix_crosscheck']} "
        f"superseded_by_authority={stats['superseded_by_authority']} semantic_dag=OK dag_self_test=OK"
    )
    for inf in stats["info_findings"]:  # type: ignore[attr-defined]
        print(f"  [INFO] {inf}: historical, superseded by the B1 chain (not a blocker)")
    return 0


FORBIDDEN_SYNC_MESSAGE = (
    "{flag} is forbidden: it rewrote WBS/matrix rows from hard-coded data. "
    "Edit the WBS and the matrix by hand under a ticket; see MEP-AF2 v2.1 section C "
    "(docs/project/maintainers/qualification/arch_freeze_program/MEP-AF2_v2.1.md, "
    "'C. 54-task WBS reconciliation')."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--sync-matrix", action="store_true", help="forbidden; exits non-zero")
    parser.add_argument("--sync-wbs", action="store_true", help="forbidden; exits non-zero")
    parser.add_argument("--self-test", action="store_true", help="run only the DAG self-test")
    args = parser.parse_args(argv)
    for flag, used in (("--sync-wbs", args.sync_wbs), ("--sync-matrix", args.sync_matrix)):
        if used:
            print(f"[FORBIDDEN] {FORBIDDEN_SYNC_MESSAGE.format(flag=flag)}", file=sys.stderr)
            return 2
    if args.self_test:
        failures = dag_self_test()
        for failure in failures:
            print(f"[ERROR] {failure}", file=sys.stderr)
        if failures:
            print(f"[FAIL] {len(failures)} DAG self-test failure(s)", file=sys.stderr)
            return 1
        print("[OK] DAG self-test passed")
        return 0
    return validate()


if __name__ == "__main__":
    raise SystemExit(main())
