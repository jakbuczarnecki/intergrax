#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Generate docs/guides/audit_slices/<DOMAIN>.md — compact audit context per domain."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guides" / "audit_slices"

# Layer → IDEAL doc sections (approx) + INVARIANTS ids to skim
DOMAIN_SLICES: dict[str, dict[str, str]] = {
    "PLATFORM_FOUNDATION": {
        "layers": "1–2, 32",
        "ideal": "§1 Strategic frame · §2 Tier model · §32 Documentation governance",
        "audit_map": "§1–§2 · §32 · maturity §5",
        "invariants": "SYS-INV-TIER-* · SYS-INV-DOC-* · P2-ARCH-01",
        "plan_hub": "§4 ladder · §6.1 maintenance · §6.3 deferred · satellite index",
        "architecture": "§1–§5 · §5.2 reuse · §5.3 terminology",
    },
    "UNIFIED_EXECUTION_RUNTIME": {
        "layers": "4–5, 8, 23–24",
        "ideal": "§4 Identity · §5 Policy · §8 Execution runtime · §23 Security · §24 Cost",
        "audit_map": "§4–§5 · §8 · §23–§24",
        "invariants": "SYS-INV-POL-* · SYS-INV-UAEP-*",
        "plan_hub": "§6 open queue · phase registers on demand",
        "architecture": "UAEP · PolicyEngine · ToolRuntime · RuntimeEvent sections",
    },
    "ORCHESTRATION": {
        "layers": "3, 9",
        "ideal": "§3 Intake · §9 Orchestration / graph",
        "audit_map": "§3 · §9",
        "invariants": "SYS-INV-ORCH-*",
        "plan_hub": "Phase ORCH-* · §6.1aw maintenance",
        "architecture": "Intake · scheduler · graph · NexusPlan sections",
    },
    "NEXUS_EXECUTION_FLOW": {
        "layers": "8–10",
        "ideal": "§8 Runtime · §9 Graph · §10 Subagents",
        "audit_map": "§8–§10",
        "invariants": "SYS-INV-FLOW-* · SYS-INV-DELEG-*",
        "plan_hub": "Phase FLOW · FLOW-CTL · §6.1aw",
        "architecture": "Flow reference §1–§27 · gap register",
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
        "architecture": "§12–§25 · §37 capability routing · ACP §21",
    },
}


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
| `docs/plan/{domain}.md` | **Hub:** {spec["plan_hub"]} |
| `docs/architecture/{domain}.md` | {spec["architecture"]} |

## Do not load unless cited

- Full multi-thousand-line plan files
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- `docs/guides/audit_slices/` for other domains

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
"""


def main() -> None:
    # Fill generic slices for all audit domains
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
            "plan_hub": f"Hub §6 open rows · [`plan/plan/`](../plan/plan/) satellites on demand",
            "architecture": f"TOC sections for layers {layers} · see Cursor read scope block",
        }

    OUT.mkdir(parents=True, exist_ok=True)
    for domain, spec in DOMAIN_SLICES.items():
        path = OUT / f"{domain}.md"
        path.write_text(render(domain, spec), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    # README
    readme = OUT / "README.md"
    readme.write_text(
        "# Audit read slices\n\n"
        "Compact per-domain audit context. Use **instead of** loading full IDEAL + AUDIT_MAP + plan.\n\n"
        "Regenerate: `uv run python scripts/generate_audit_read_slices.py`\n",
        encoding="utf-8",
    )
    print(f"wrote {readme.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
