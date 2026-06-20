#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Insert Cursor read-scope blocks into architecture domain docs (E2)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "docs" / "architecture"
MARKER = "## Cursor read scope (token budget)"
MIN_LINES = 500

SCOPES: dict[str, str] = {
    "PLATFORM_FOUNDATION": (
        "§1–§5 + §5.2 reuse + §5.3 terminology. "
        "Extended §7–§8: [`arch/PLATFORM_FOUNDATION_extended_depth.md`](arch/PLATFORM_FOUNDATION_extended_depth.md). "
        "§43+: [`arch/PLATFORM_FOUNDATION_production_gates.md`](arch/PLATFORM_FOUNDATION_production_gates.md)."
    ),
    "AGENT_CONTRACTS_AND_ASSEMBLY": (
        "§12–§21 (contract, registry, capability, ACP). "
        "Extended §22–§39: [`arch/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md`](arch/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md). "
        "§40+: [`arch/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md`](arch/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md)."
    ),
    "TIER3_APPLICATION_ENVIRONMENT": (
        "§20–§25 host profile + manifest wiring. "
        "Extended §26–§39: [`arch/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md`](arch/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md). "
        "§40+: [`arch/TIER3_APPLICATION_ENVIRONMENT_production_gates.md`](arch/TIER3_APPLICATION_ENVIRONMENT_production_gates.md)."
    ),
    "TOOLS": (
        "ToolRuntime path + plugin model + policy invoke (hub § through production posture). "
        "Selection / invocation patterns: [`arch/TOOLS_selection_and_plugins.md`](arch/TOOLS_selection_and_plugins.md). "
        "RuntimeConfig fields: [`arch/TOOLS_runtime_config_reference.md`](arch/TOOLS_runtime_config_reference.md)."
    ),
    "UNIFIED_EXECUTION_RUNTIME": (
        "UAEP + PolicyEngine + RuntimeEvent spine (§42.1–§42.15). "
        "Extended: [`arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md`](arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md)."
    ),
    "ORCHESTRATION": (
        "intake + NexusLoop + graph executor (§10–§26). "
        "Extended: [`arch/ORCHESTRATION_extended_depth.md`](arch/ORCHESTRATION_extended_depth.md)."
    ),
    "NEXUS_EXECUTION_FLOW": (
        "§1–§20 flow narrative. "
        "Reference §21+: [`arch/NEXUS_EXECUTION_FLOW_scenario_catalog.md`](arch/NEXUS_EXECUTION_FLOW_scenario_catalog.md)."
    ),
    "INTEGRATIONS": (
        "manifest registration + IntegrationProfile + wiring. "
        "Catalog: [`arch/INTEGRATIONS_provider_catalog.md`](arch/INTEGRATIONS_provider_catalog.md)."
    ),
    "CRITIC_VERIFICATION": (
        "CVL contracts + orchestrator + wiring. Skip historical LC narrative unless cited."
    ),
    "REASONING_AND_COGNITION": (
        "DecisionRecord + planner/classifier spine. Skip historical sprint logs unless cited."
    ),
    "ADAPTIVE_HARNESS_INTELLIGENCE": (
        "L4 adaptive loop contracts. Skip maturity history unless AHI task."
    ),
    "LLM_ADAPTERS": (
        "adapter envelope + provider routing. Skip legacy migration tables unless cited."
    ),
    "MEMORY": (
        "LTM store contracts + scope model. Skip store inventory tables — use code grep."
    ),
    "RAG": (
        "retrieval pipeline + index lifecycle. Skip full corpus tables unless RAG task."
    ),
    "SKILLS": (
        "skill selection hook + registry. Skip LC narrative unless SK task."
    ),
    "CONTEXT_ENGINEERING": (
        "context assembly engine + scoring. Skip historical gap logs unless cited."
    ),
    "CODE_CRAFT": (
        "ephemeral codegen loop contracts. Skip LC closeout unless ECC task."
    ),
    "OBSERVABILITY": (
        "trace spine + RuntimeEvent. Skip OBS-LC history unless cited."
    ),
    "RELIABILITY_FAILURE_AND_HITL": (
        "retry/HITL contracts. Skip failure taxonomy appendices unless cited."
    ),
    "MODALITY": (
        "vision/audio modality adapters. Skip modality inventory unless MOD task."
    ),
    "ELASTIC_CAPACITY_AND_SCALING": (
        "capacity adapter contracts. Skip scaling history unless ECP task."
    ),
    "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE": (
        "DX gates + eval harness. Skip W-OPS history unless DX task."
    ),
}

SCOPE_BLOCK_RE = re.compile(
    rf"{re.escape(MARKER)}.*?\n---\n",
    re.DOTALL,
)


def block(domain: str, scope: str) -> str:
    return (
        f"{MARKER}\n\n"
        f"**Do not read this entire file in one session** ({domain} canon).\n\n"
        f"- **Implement / audit default:** {scope}\n"
        f"- **Use** table of contents below — `Read` with offset/limit per §.\n"
        f"- **Plan hub:** [`plan/{domain}.md`](../plan/{domain}.md) (scoped §6 only).\n"
        f"- **Audit slice:** [`guides/audit_slices/{domain}.md`](../guides/audit_slices/{domain}.md).\n"
        f"- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.\n\n"
        f"---\n"
    )


def upsert_scope(text: str, domain: str, scope: str) -> str:
    new_block = block(domain, scope)
    if MARKER in text:
        return SCOPE_BLOCK_RE.sub(new_block, text, count=1)
    insert_at = text.find("\n---\n")
    if insert_at == -1:
        return text
    insert_at += len("\n---\n")
    return text[:insert_at] + "\n" + new_block + "\n" + text[insert_at:].lstrip("\n")


def main() -> None:
    for path in sorted(ARCH.glob("*.md")):
        domain = path.stem
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < MIN_LINES and domain not in SCOPES:
            continue
        scope = SCOPES.get(
            domain,
            "Read TOC sections matching current task only; skip appendices and paydown logs.",
        )
        text = path.read_text(encoding="utf-8")
        updated = upsert_scope(text, domain, scope)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
