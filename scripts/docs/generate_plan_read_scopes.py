#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Insert Cursor read-scope blocks into plan domain hubs (G1-E2)."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_DOCS_SCRIPTS = _Path(__file__).resolve().parents[1] / "docs"
if str(_DOCS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_DOCS_SCRIPTS))

import re
from pathlib import Path

from plan_hub_lib import (
    PLAN_DIR,
    SAT_DIR,
    SKIP_PLAN_HUBS,
    normalize_plan_satellite_budget_line,
    satellite_links,
    upsert_plan_read_scope,
)

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCOPE = (
    "Open `## 6` / `### 6.1*` maintenance queues — **P0/P1** rows with Status ≠ Done only; "
    "skip closed/complete registers unless re-validating a cited gap"
)

EXTRA_SCOPES: dict[str, str] = {
    "PLATFORM_FOUNDATION": (
        "§6.1 gate maintenance (default) · §6.3 deferred product only · §4.0a scope split. "
        "**On demand:** [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md) · "
        "[`plan/satellites/PLATFORM_FOUNDATION_06_closed_queues.md`](plan/satellites/PLATFORM_FOUNDATION_06_closed_queues.md) (re-validate closed only)"
    ),
    "ORCHESTRATION": (
        "Active `### 6.1*` queues with open P0/P1 · Phase AUDIT-IDEAL **Planned** rows. "
        "Closed ORCH-* registers — satellite only when re-validating"
    ),
    "NEXUS_EXECUTION_FLOW": (
        "§6.1 FLOW maintenance · open P0/P1 rows · Phase AUDIT-IDEAL gap table. "
        "Historical flow registers — [`plan/satellites/`](plan/satellites/) satellite on demand"
    ),
    "UNIFIED_EXECUTION_RUNTIME": (
        "§6.1 UAEP maintenance · R-Policy / SEC / COST open rows · phase satellites on demand"
    ),
    "AGENT_CONTRACTS_AND_ASSEMBLY": (
        "§6.1bc ACP-FINISH status · AUDIT-IDEAL §12–§20 table (**Done** skip unless cited) · "
        "[`plan/satellites/AGENT_CONTRACTS_AND_ASSEMBLY_audit_history.md`](plan/satellites/AGENT_CONTRACTS_AND_ASSEMBLY_audit_history.md) on demand"
    ),
    "INTEGRATIONS": (
        "Phase INT / H-INT hub queues · §6.1 open P0/P1 · M.6 expansion registers — satellite on demand"
    ),
    "MODALITY": (
        "§6.1 MOD maintenance · open modality integration rows · skip closed MOD-LC narrative"
    ),
    "RELIABILITY_FAILURE_AND_HITL": (
        "§6.1 REL / HITL maintenance · open retry/HITL rows · closed reliability LC — satellite only"
    ),
    "ELASTIC_CAPACITY_AND_SCALING": (
        "ECP phase registers · open P0/P1 capacity rows · skip closed scaling history unless cited"
    ),
}


def _audit_slice_plan_hints() -> dict[str, str]:
    return {}


def _domain_plan_read_scope() -> dict[str, str]:
    return {}


def _pf_satellites_from_disk() -> list[str]:
    return sorted(p.name for p in SAT_DIR.glob("PLATFORM_FOUNDATION_*.md"))


def build_scope(domain: str, hub_text: str, hints: dict[str, str], explicit: dict[str, str]) -> str:
    if domain in EXTRA_SCOPES:
        return EXTRA_SCOPES[domain]
    if domain in explicit:
        return explicit[domain]
    parts: list[str] = [hints.get(domain, DEFAULT_SCOPE)]
    sats = satellite_links(hub_text)
    if domain == "PLATFORM_FOUNDATION" and not sats:
        sats = _pf_satellites_from_disk()[:2]
    if sats:
        links = " · ".join(f"[`plan/satellites/{name}`](plan/satellites/{name})" for name in sats[:2])
        parts.append(f"**On demand (one max):** {links}")
    if "## Phase AUDIT-IDEAL" in hub_text:
        parts.append("Phase AUDIT-IDEAL — **Planned** / open rows only")
    if re.search(r"### 6\.1", hub_text):
        parts.append("§6.1 maintenance queues — open P0/P1 only")
    return ". ".join(parts)


def main() -> None:
    hints = _audit_slice_plan_hints()
    explicit = _domain_plan_read_scope()

    for path in sorted(PLAN_DIR.glob("*.md")):
        if path.name in SKIP_PLAN_HUBS:
            continue
        domain = path.stem
        text = path.read_text(encoding="utf-8")
        scope = build_scope(domain, text, hints, explicit)
        updated = upsert_plan_read_scope(text, domain, scope)
        updated = normalize_plan_satellite_budget_line(updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
