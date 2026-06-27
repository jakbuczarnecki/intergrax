#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Generate docs/guides/audit_slices/<DOMAIN>.md — compact audit context + CODE_ENTRY (F5-B)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "guides" / "audit_slices"

DOMAIN_CODE_ENTRIES: dict[str, tuple[str, ...]] = {
    "PLATFORM_FOUNDATION": (
        "`intergrax/scaffold/` — scaffolding CLI",
        "`scripts/maintenance/check_plan_hub_size.py` — plan hub gate",
    ),
    "UNIFIED_EXECUTION_RUNTIME": (
        "`intergrax/runtime/nexus/policy/` — PolicyEngine",
        "`intergrax/runtime/nexus/execution/` — UAEP / HarnessKernel",
        "`intergrax/runtime/nexus/events/` — RuntimeEvent spine",
    ),
    "ORCHESTRATION": (
        "`intergrax/runtime/nexus/orchestration/` — intake, NexusLoop",
        "`intergrax/runtime/nexus/orchestration/graph/` — execution graph",
    ),
    "NEXUS_EXECUTION_FLOW": (
        "`intergrax/runtime/nexus/orchestration/nexus_loop.py` — NexusLoop",
        "`intergrax/runtime/nexus/orchestration/intake/` — task intake",
    ),
    "AGENT_CONTRACTS_AND_ASSEMBLY": (
        "`intergrax/runtime/nexus/agent/` — agent contracts, registry",
        "`agents/` — Tier-2 agent implementations",
    ),
    "INTEGRATIONS": (
        "`intergrax/integrations/` — integration catalog",
        "`intergrax/integrations/registry.py` — slug registration",
    ),
    "RAG": (
        "`intergrax/rag/` — retrieval engine",
        "`intergrax/rag/engine.py` — RAG pipeline entry",
    ),
    "TOOLS": (
        "`intergrax/tools/` — ToolRuntime",
        "`intergrax/tools/runtime.py` — tool invoke path",
    ),
    "MEMORY": (
        "`intergrax/runtime/nexus/memory/` — memory stores",
        "`intergrax/memory/` — LTM facades",
    ),
    "TIER3_APPLICATION_ENVIRONMENT": (
        "`applications/` — Tier-3 hosts",
        "`intergrax/runtime/nexus/application/` — HarnessApplication",
    ),
    "OBSERVABILITY": (
        "`intergrax/runtime/nexus/observability/` — trace spine",
        "`intergrax/runtime/nexus/events/` — RuntimeEvent",
    ),
    "CRITIC_VERIFICATION": (
        "`intergrax/runtime/nexus/critic/` — CVL orchestrator",
    ),
}

DOMAIN_SLICES: dict[str, dict[str, str]] = {
    "PLATFORM_FOUNDATION": {
        "layers": "1–2, 32",
        "ideal": "§1 Strategic frame · §2 Tier model · §32 Documentation governance",
        "audit_map": "§1–§2 · §32 · maturity §5",
        "invariants": "SYS-INV-TIER-* · SYS-INV-DOC-* · P2-ARCH-01",
        "plan_hub": "§6.1 maintenance · §6.3 deferred · [`plan/satellites/`](../plan/satellites/) on demand",
        "architecture": "§1–§6 hub · [`satellites/`](../architecture/satellites/) on demand",
    },
    "UNIFIED_EXECUTION_RUNTIME": {
        "layers": "4–5, 8, 23–24",
        "ideal": "§4 Identity · §5 Policy · §8 Execution runtime · §23 Security · §24 Cost",
        "audit_map": "§4–§5 · §8 · §23–§24",
        "invariants": "SYS-INV-POL-* · SYS-INV-UAEP-*",
        "plan_hub": "§6.1av hub · phase satellites on demand",
        "architecture": "§42.1–§42.15 hub · [`satellites/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md`](../architecture/satellites/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md) on demand",
    },
    "ORCHESTRATION": {
        "layers": "3, 9",
        "ideal": "§3 Intake · §9 Orchestration / graph",
        "audit_map": "§3 · §9",
        "invariants": "SYS-INV-ORCH-*",
        "plan_hub": "Phase ORCH-* hub · satellites on demand",
        "architecture": "§10–§26 hub · [`satellites/ORCHESTRATION_extended_depth.md`](../architecture/satellites/ORCHESTRATION_extended_depth.md) on demand",
    },
    "NEXUS_EXECUTION_FLOW": {
        "layers": "8–10",
        "ideal": "§8 Runtime · §9 Graph · §10 Subagents",
        "audit_map": "§8–§10",
        "invariants": "SYS-INV-FLOW-* · SYS-INV-DELEG-*",
        "plan_hub": "Phase FLOW hub · satellites on demand",
        "architecture": "§1–§8 hub · [`satellites/NEXUS_EXECUTION_FLOW_extended_depth.md`](../architecture/satellites/NEXUS_EXECUTION_FLOW_extended_depth.md) on demand",
    },
    "CRITIC_VERIFICATION": {
        "layers": "25–27, 30",
        "ideal": "§18 Critic · §25 Evaluation",
        "audit_map": "§25–§27 · §30",
        "invariants": "SYS-INV-EVAL-* · SYS-INV-CRIT-*",
        "plan_hub": "AUDIT-IDEAL · §CVL-4 backlog · audit_history satellite",
        "architecture": "CVL contracts · PEV · evaluator loop",
    },
    "AGENT_CONTRACTS_AND_ASSEMBLY": {
        "layers": "17–20, 31 · ACP §21",
        "ideal": "§12–§21 Agent / registry / ACP",
        "audit_map": "§17–§20 · §31",
        "invariants": "SYS-INV-ACP-* · SYS-INV-AGENT-*",
        "plan_hub": "§6 open · ACP closeout registers on demand",
        "architecture": "§12–§21 hub · [`satellites/`](../architecture/satellites/) on demand",
    },
    "INTEGRATIONS": {
        "layers": "11–12",
        "ideal": "§11 Integration library · §12 Provider model",
        "audit_map": "§11–§12",
        "invariants": "SYS-INV-INT-*",
        "plan_hub": "Phase INT / H-INT hub · satellites on demand",
        "architecture": "wiring + design principles hub · [`satellites/INTEGRATIONS_provider_catalog.md`](../architecture/satellites/INTEGRATIONS_provider_catalog.md) on demand",
    },
}


def code_entry_block(domain: str) -> str:
    entries = DOMAIN_CODE_ENTRIES.get(domain)
    if not entries:
        entries = (
            f"`docs/architecture/{domain}.md` — read-scope block only",
            f"`docs/plan/{domain}.md` — read-scope block only",
            "`docs/guides/SYMBOL_INDEX.md` — symbol grep map",
        )
    lines = ["## Code entry (grep first — F5-B)", ""]
    for e in entries:
        lines.append(f"- {e}")
    lines.append("")
    return "\n".join(lines)


def render(domain: str, spec: dict[str, str]) -> str:
    return f"""# Audit read slice — `{domain}`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/{domain}.md`](../../audit/{domain}.md) §0

---

## Audit-map layers

{spec["layers"]}

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | {spec["ideal"]} |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | {spec["audit_map"]} |
| `docs/guides/SYSTEM_INVARIANTS.md` | {spec["invariants"]} (grep IDs — do not read full file) |
| `docs/plan/{domain}.md` | **Read-scope:** {spec["plan_hub"]} |
| `docs/architecture/{domain}.md` | {spec["architecture"]} |

{code_entry_block(domain)}
## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
"""


def main() -> None:
    import runpy

    gen = runpy.run_path(str(ROOT / "scripts" / "generate_domain_audit_prompts.py"))
    for entry in gen["DOMAINS"]:
        did = entry["id"]
        layers = entry["layers"]
        if did in DOMAIN_SLICES:
            continue
        DOMAIN_SLICES[did] = {
            "layers": layers,
            "ideal": f"Sections matching audit-map layers {layers}",
            "audit_map": f"Layers {layers} · maturity §5",
            "invariants": "Grep SYS-INV-* IDs from audit dimensions only",
            "plan_hub": f"Hub §6 · [`plan/satellites/`](../plan/satellites/) satellites on demand",
            "architecture": f"Read-scope block + TOC sections for layers {layers}",
        }

    OUT.mkdir(parents=True, exist_ok=True)
    for domain, spec in DOMAIN_SLICES.items():
        path = OUT / f"{domain}.md"
        path.write_text(render(domain, spec), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    readme = OUT / "README.md"
    readme.write_text(
        "# Audit read slices\n\n"
        "Compact per-domain audit context + CODE_ENTRY paths. "
        "Use **instead of** loading full IDEAL + AUDIT_MAP + plan/arch.\n\n"
        "Regenerate: `uv run python scripts/audit/generate_audit_read_slices.py`\n",
        encoding="utf-8",
    )
    print(f"wrote {readme.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
