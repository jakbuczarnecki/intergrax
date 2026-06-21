#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Insert Cursor read-scope blocks into architecture domain docs (E2)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "docs" / "architecture"
MARKER = "## Cursor read scope (token budget)"
MIN_LINES = 200

SCOPES: dict[str, str] = {
    "PLATFORM_FOUNDATION": (
        "§1–§6 platform spine. "
        "Extended §7+: [`arch/PLATFORM_FOUNDATION_extended_depth.md`](arch/PLATFORM_FOUNDATION_extended_depth.md). "
        "§43+: [`arch/PLATFORM_FOUNDATION_production_gates.md`](arch/PLATFORM_FOUNDATION_production_gates.md)."
    ),
    "AGENT_CONTRACTS_AND_ASSEMBLY": (
        "§12–§21 (contract, registry, capability, ACP). "
        "Extended §22–§39 + checklist §45: [`arch/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md`](arch/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md). "
        "§40+: [`arch/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md`](arch/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md)."
    ),
    "TIER3_APPLICATION_ENVIRONMENT": (
        "§20–§25 host profile + manifest wiring. "
        "Extended §26–§39: [`arch/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md`](arch/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md). "
        "§40+: [`arch/TIER3_APPLICATION_ENVIRONMENT_production_gates.md`](arch/TIER3_APPLICATION_ENVIRONMENT_production_gates.md)."
    ),
    "TOOLS": (
        "ToolRuntime path + plugin model + policy invoke (hub through production posture). "
        "Selection / invocation: [`arch/TOOLS_selection_and_plugins.md`](arch/TOOLS_selection_and_plugins.md). "
        "RuntimeConfig fields: [`arch/TOOLS_runtime_config_reference.md`](arch/TOOLS_runtime_config_reference.md)."
    ),
    "UNIFIED_EXECUTION_RUNTIME": (
        "UAEP + RuntimeEvent spine (§42.1–§42.7). "
        "Extended §42.8+: [`arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md`](arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md)."
    ),
    "ORCHESTRATION": (
        "intake + NexusLoop + graph executor (§10–§26). "
        "§27+: [`arch/ORCHESTRATION_production_gates.md`](arch/ORCHESTRATION_production_gates.md)."
    ),
    "NEXUS_EXECUTION_FLOW": (
        "§1–§8 flow spine (purpose → classification). "
        "Extended §9+: [`arch/NEXUS_EXECUTION_FLOW_extended_depth.md`](arch/NEXUS_EXECUTION_FLOW_extended_depth.md)."
    ),
    "INTEGRATIONS": (
        "IntegrationLayer contract + wiring + checklists (hub). "
        "Provider catalog: [`arch/INTEGRATIONS_provider_catalog.md`](arch/INTEGRATIONS_provider_catalog.md)."
    ),
    "CRITIC_VERIFICATION": (
        "CVL contracts + orchestrator + wiring (§1–§6). "
        "Extended §7+: [`arch/CRITIC_VERIFICATION_extended_depth.md`](arch/CRITIC_VERIFICATION_extended_depth.md)."
    ),
    "REASONING_AND_COGNITION": (
        "DecisionRecord + planner/classifier spine (§1–§7). "
        "Extended §8+: [`arch/REASONING_AND_COGNITION_extended_depth.md`](arch/REASONING_AND_COGNITION_extended_depth.md)."
    ),
    "ADAPTIVE_HARNESS_INTELLIGENCE": (
        "L4 adaptive loop contracts (§1–§7). "
        "Extended §8+: [`arch/ADAPTIVE_HARNESS_INTELLIGENCE_extended_depth.md`](arch/ADAPTIVE_HARNESS_INTELLIGENCE_extended_depth.md)."
    ),
    "LLM_ADAPTERS": (
        "adapter envelope + routing hub. "
        "Failover: [`arch/LLM_ADAPTERS_routing_failover.md`](arch/LLM_ADAPTERS_routing_failover.md). "
        "Providers: [`arch/LLM_ADAPTERS_providers_catalog.md`](arch/LLM_ADAPTERS_providers_catalog.md). "
        "Audit register: [`arch/LLM_ADAPTERS_audit_register.md`](arch/LLM_ADAPTERS_audit_register.md)."
    ),
    "MEMORY": (
        "LTM store contracts + scope model (§1–§7). "
        "Extended §8+: [`arch/MEMORY_extended_depth.md`](arch/MEMORY_extended_depth.md)."
    ),
    "RAG": (
        "retrieval pipeline + index lifecycle (hub). "
        "Pipelines detail: [`arch/RAG_pipelines_detail.md`](arch/RAG_pipelines_detail.md)."
    ),
    "SKILLS": (
        "skill selection hook + registry (hub). "
        "Catalog: [`arch/SKILLS_skill_catalog.md`](arch/SKILLS_skill_catalog.md)."
    ),
    "CONTEXT_ENGINEERING": (
        "context assembly engine + scoring (§1–§7). "
        "Extended §8+: [`arch/CONTEXT_ENGINEERING_extended_depth.md`](arch/CONTEXT_ENGINEERING_extended_depth.md)."
    ),
    "CODE_CRAFT": (
        "ephemeral codegen loop contracts (§1–§6). "
        "Extended §7+: [`arch/CODE_CRAFT_extended_depth.md`](arch/CODE_CRAFT_extended_depth.md)."
    ),
    "OBSERVABILITY": (
        "trace spine + HOS + signal planes (§1–§4). "
        "Extended §5+: [`arch/OBSERVABILITY_extended_depth.md`](arch/OBSERVABILITY_extended_depth.md)."
    ),
    "RELIABILITY_FAILURE_AND_HITL": (
        "§30–§32 failure + retry + HITL core. "
        "Extended §33+: [`arch/RELIABILITY_FAILURE_AND_HITL_extended_depth.md`](arch/RELIABILITY_FAILURE_AND_HITL_extended_depth.md). "
        "§35+: [`arch/RELIABILITY_FAILURE_AND_HITL_production_gates.md`](arch/RELIABILITY_FAILURE_AND_HITL_production_gates.md)."
    ),
    "MODALITY": (
        "modality adapters hub. "
        "Tool surface: [`arch/MODALITY_tool_surface_detail.md`](arch/MODALITY_tool_surface_detail.md)."
    ),
    "ELASTIC_CAPACITY_AND_SCALING": (
        "capacity adapter contracts (§1–§7). "
        "Extended §8+: [`arch/ELASTIC_CAPACITY_AND_SCALING_extended_depth.md`](arch/ELASTIC_CAPACITY_AND_SCALING_extended_depth.md)."
    ),
    "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE": (
        "§39–§41 DX + minimal runtime flow. "
        "Extended §42+: [`arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_extended_depth.md`](arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_extended_depth.md). "
        "§43+: [`arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md`](arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md)."
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
            "Read TOC sections matching current task only; load matching `arch/<DOMAIN>_*.md` satellite on demand.",
        )
        text = path.read_text(encoding="utf-8")
        updated = upsert_scope(text, domain, scope)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
