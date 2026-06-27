#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Insert Cursor read-scope blocks into architecture domain docs (E2)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "docs" / "architecture"
MARKER = "## Cursor read scope (token budget)"
MIN_LINES = 200

SCOPES: dict[str, str] = {
    "PLATFORM_FOUNDATION": (
        "§1–§6 platform spine. "
        "Extended §7+: [`satellites/PLATFORM_FOUNDATION_extended_depth.md`](satellites/PLATFORM_FOUNDATION_extended_depth.md). "
        "§43+: [`satellites/PLATFORM_FOUNDATION_production_gates.md`](satellites/PLATFORM_FOUNDATION_production_gates.md)."
    ),
    "AGENT_CONTRACTS_AND_ASSEMBLY": (
        "§12–§21 (contract, registry, capability, ACP). "
        "Extended §22–§39 + checklist §45: [`satellites/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md`](satellites/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md). "
        "§40+: [`satellites/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md`](satellites/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md)."
    ),
    "TIER3_APPLICATION_ENVIRONMENT": (
        "§20–§25 host profile + manifest wiring. "
        "Extended §26–§39: [`satellites/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md`](satellites/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md). "
        "§40+: [`satellites/TIER3_APPLICATION_ENVIRONMENT_production_gates.md`](satellites/TIER3_APPLICATION_ENVIRONMENT_production_gates.md)."
    ),
    "TOOLS": (
        "ToolRuntime path + plugin model + policy invoke (hub through production posture). "
        "Selection / invocation: [`satellites/TOOLS_selection_and_plugins.md`](satellites/TOOLS_selection_and_plugins.md). "
        "RuntimeConfig fields: [`satellites/TOOLS_runtime_config_reference.md`](satellites/TOOLS_runtime_config_reference.md)."
    ),
    "UNIFIED_EXECUTION_RUNTIME": (
        "UAEP + RuntimeEvent spine (§42.1–§42.7). "
        "Extended §42.8+: [`satellites/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md`](satellites/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md)."
    ),
    "ORCHESTRATION": (
        "intake + NexusLoop + graph executor (§10–§26). "
        "§27+: [`satellites/ORCHESTRATION_production_gates.md`](satellites/ORCHESTRATION_production_gates.md)."
    ),
    "NEXUS_EXECUTION_FLOW": (
        "§1–§8 flow spine (purpose → classification). "
        "Extended §9+: [`satellites/NEXUS_EXECUTION_FLOW_extended_depth.md`](satellites/NEXUS_EXECUTION_FLOW_extended_depth.md)."
    ),
    "INTEGRATIONS": (
        "IntegrationLayer contract + wiring + checklists (hub). "
        "Provider catalog: [`satellites/INTEGRATIONS_provider_catalog.md`](satellites/INTEGRATIONS_provider_catalog.md)."
    ),
    "CRITIC_VERIFICATION": (
        "CVL contracts + orchestrator + wiring (§1–§6). "
        "Extended §7+: [`satellites/CRITIC_VERIFICATION_extended_depth.md`](satellites/CRITIC_VERIFICATION_extended_depth.md)."
    ),
    "REASONING_AND_COGNITION": (
        "DecisionRecord + planner/classifier spine (§1–§7). "
        "Extended §8+: [`satellites/REASONING_AND_COGNITION_extended_depth.md`](satellites/REASONING_AND_COGNITION_extended_depth.md)."
    ),
    "ADAPTIVE_HARNESS_INTELLIGENCE": (
        "L4 adaptive loop contracts (§1–§7). "
        "Extended §8+: [`satellites/ADAPTIVE_HARNESS_INTELLIGENCE_extended_depth.md`](satellites/ADAPTIVE_HARNESS_INTELLIGENCE_extended_depth.md)."
    ),
    "LLM_ADAPTERS": (
        "adapter envelope + routing hub. "
        "Failover: [`satellites/LLM_ADAPTERS_routing_failover.md`](satellites/LLM_ADAPTERS_routing_failover.md). "
        "Providers: [`satellites/LLM_ADAPTERS_providers_catalog.md`](satellites/LLM_ADAPTERS_providers_catalog.md). "
        "Audit register: [`satellites/LLM_ADAPTERS_audit_register.md`](satellites/LLM_ADAPTERS_audit_register.md)."
    ),
    "MEMORY": (
        "LTM store contracts + scope model (§1–§7). "
        "Extended §8+: [`satellites/MEMORY_extended_depth.md`](satellites/MEMORY_extended_depth.md)."
    ),
    "RAG": (
        "retrieval pipeline + index lifecycle (hub). "
        "Pipelines detail: [`satellites/RAG_pipelines_detail.md`](satellites/RAG_pipelines_detail.md)."
    ),
    "SKILLS": (
        "skill selection hook + registry (hub). "
        "Catalog: [`satellites/SKILLS_skill_catalog.md`](satellites/SKILLS_skill_catalog.md)."
    ),
    "CONTEXT_ENGINEERING": (
        "context assembly engine + scoring (§1–§7). "
        "Extended §8+: [`satellites/CONTEXT_ENGINEERING_extended_depth.md`](satellites/CONTEXT_ENGINEERING_extended_depth.md)."
    ),
    "CODE_CRAFT": (
        "ephemeral codegen loop contracts (§1–§6). "
        "Extended §7+: [`satellites/CODE_CRAFT_extended_depth.md`](satellites/CODE_CRAFT_extended_depth.md)."
    ),
    "OBSERVABILITY": (
        "trace spine + HOS + signal planes (§1–§4). "
        "Extended §5+: [`satellites/OBSERVABILITY_extended_depth.md`](satellites/OBSERVABILITY_extended_depth.md)."
    ),
    "RELIABILITY_FAILURE_AND_HITL": (
        "§30–§32 failure + retry + HITL core. "
        "Extended §33+: [`satellites/RELIABILITY_FAILURE_AND_HITL_extended_depth.md`](satellites/RELIABILITY_FAILURE_AND_HITL_extended_depth.md). "
        "§35+: [`satellites/RELIABILITY_FAILURE_AND_HITL_production_gates.md`](satellites/RELIABILITY_FAILURE_AND_HITL_production_gates.md)."
    ),
    "MODALITY": (
        "modality adapters hub. "
        "Tool surface: [`satellites/MODALITY_tool_surface_detail.md`](satellites/MODALITY_tool_surface_detail.md)."
    ),
    "ELASTIC_CAPACITY_AND_SCALING": (
        "capacity adapter contracts (§1–§7). "
        "Extended §8+: [`satellites/ELASTIC_CAPACITY_AND_SCALING_extended_depth.md`](satellites/ELASTIC_CAPACITY_AND_SCALING_extended_depth.md)."
    ),
    "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE": (
        "§39–§41 DX + minimal runtime flow. "
        "Extended §42+: [`satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_extended_depth.md`](satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_extended_depth.md). "
        "§43+: [`satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md`](satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md)."
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
            "Read TOC sections matching current task only; load matching `satellites/<DOMAIN>_*.md` satellite on demand.",
        )
        text = path.read_text(encoding="utf-8")
        updated = upsert_scope(text, domain, scope)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
