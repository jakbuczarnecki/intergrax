# © Artur Czarnecki. All rights reserved.
"""One-shot helper: split intergrax_runtime_architecture monolith into domain files."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "intergrax_runtime_architecture.md"
ARCH = REPO / "docs" / "architecture"
BACKUP = REPO / "docs" / "_archive_intergrax_runtime_architecture_monolith.md"

# Top-level section number -> target file (without .md)
SECTION_MAP: dict[int, str] = {
    1: "PLATFORM_FOUNDATION",
    2: "PLATFORM_FOUNDATION",
    3: "PLATFORM_FOUNDATION",
    4: "PLATFORM_FOUNDATION",
    5: "PLATFORM_FOUNDATION",
    6: "PLATFORM_FOUNDATION",
    7: "PLATFORM_FOUNDATION",
    8: "PLATFORM_FOUNDATION",
    9: "ORCHESTRATION",
    10: "ORCHESTRATION",
    11: "AGENT_CONTRACTS_AND_ASSEMBLY",
    12: "AGENT_CONTRACTS_AND_ASSEMBLY",
    13: "AGENT_CONTRACTS_AND_ASSEMBLY",
    14: "AGENT_CONTRACTS_AND_ASSEMBLY",
    15: "AGENT_CONTRACTS_AND_ASSEMBLY",
    16: "AGENT_CONTRACTS_AND_ASSEMBLY",
    17: "INTEGRATIONS_ARCHITECTURE",
    18: "INTEGRATIONS_ARCHITECTURE",
    19: "TIER3_APPLICATION_ENVIRONMENT",
    20: "TIER3_APPLICATION_ENVIRONMENT",
    21: "TIER3_APPLICATION_ENVIRONMENT",
    22: "TOOLS_RUNTIME",
    23: "ORCHESTRATION",
    24: "ORCHESTRATION",
    25: "ORCHESTRATION",
    26: "ORCHESTRATION",
    27: "MEMORY_CANON_POINTER",  # pointer file -> MEMORY_ARCHITECTURE
    28: "CONTEXT_ENGINEERING",
    29: "RELIABILITY_FAILURE_AND_HITL",
    30: "RELIABILITY_FAILURE_AND_HITL",
    31: "RELIABILITY_FAILURE_AND_HITL",
    32: "RELIABILITY_FAILURE_AND_HITL",
    33: "OBSERVABILITY_CANON_POINTER",
    34: "EVALUATION_AND_BENCHMARKING",
    35: "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    36: "BUSINESS_EXAMPLES_POINTER",
    37: "BUSINESS_EXAMPLES_POINTER",
    38: "BUSINESS_EXAMPLES_POINTER",
    39: "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    40: "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    41: "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    42: "UNIFIED_EXECUTION_RUNTIME",
    43: "PLATFORM_FOUNDATION",
    44: "PLATFORM_FOUNDATION",
    45: "AGENT_CONTRACTS_AND_ASSEMBLY",
    46: "INTEGRATIONS_ARCHITECTURE",
    47: "ORCHESTRATION",
    48: "PLATFORM_FOUNDATION",
    49: "PLATFORM_FOUNDATION",
    50: "PLATFORM_FOUNDATION",
    51: "PLATFORM_FOUNDATION",
    52: "IMPLEMENTATION_POINTER",
    53: "PLATFORM_FOUNDATION",
    54: "ADAPTIVE_HARNESS_POINTER",
    55: "CRITIC_VERIFICATION_POINTER",
}

FILE_TITLES: dict[str, str] = {
    "PLATFORM_FOUNDATION": "Platform Foundation — Tiers, Principles, and Boundaries",
    "UNIFIED_EXECUTION_RUNTIME": "Unified Execution Runtime Specification (UAEP)",
    "ORCHESTRATION": "Orchestration, Nexus, and Execution Graph",
    "AGENT_CONTRACTS_AND_ASSEMBLY": "Agent Contracts, Registry, and Capability Model",
    "INTEGRATIONS_ARCHITECTURE": "Integration and Adapter Architecture",
    "TOOLS_RUNTIME": "Tool Runtime and Unified Tool Model",
    "CONTEXT_ENGINEERING": "Context Engineering",
    "RELIABILITY_FAILURE_AND_HITL": "Reliability, Failure Model, and Human-in-the-Loop",
    "EVALUATION_AND_BENCHMARKING": "Evaluation and Benchmarking",
    "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE": "Experimentation Workflow and Developer Experience",
    "TIER3_APPLICATION_ENVIRONMENT": "Tier-3 Application Environment, Sandbox, and Shadow Workspace",
}

POINTER_FILES: dict[str, str] = {
    "MEMORY_CANON_POINTER": """# Memory Model (canonical pointer)

**Canonical document:** [`MEMORY_ARCHITECTURE.md`](../MEMORY_ARCHITECTURE.md)

Platform memory (STM, LTM, org/task scopes, hooks, persistence, context compiler depth) is specified in the dedicated memory architecture document. The implementation plan tracks completion under Phase MEM and MEM-DEPTH.

**Related:** [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`RAG_AND_RETRIEVAL.md`](RAG_AND_RETRIEVAL.md) · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.35
""",
    "OBSERVABILITY_CANON_POINTER": """# Observability and Tracing (canonical pointer)

**Canonical document:** [`OBSERVABILITY_ARCHITECTURE.md`](../OBSERVABILITY_ARCHITECTURE.md)

Unified observability spine, event bus, trace bridge, and telemetry contracts are specified in the dedicated observability architecture document. ADR: [`ADR-OBS-001.md`](../adr/ADR-OBS-001.md).

**Related:** [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.24 · [`POLICY_AND_GOVERNANCE.md`](POLICY_AND_GOVERNANCE.md)
""",
    "ADAPTIVE_HARNESS_POINTER": """# Adaptive Harness Intelligence (canonical pointer)

**Canonical document:** [`ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md`](../ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md)

L4 adaptive runtime (feedback loops, bounded self-tuning) is specified in the dedicated AHI architecture document. Implementation: Phase W-ADAPT in [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md).
""",
    "CRITIC_VERIFICATION_POINTER": """# Critic and Verification Layer (canonical pointer)

**Canonical document:** [`CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md`](../CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md)

PEV verify depth and critic/verification contracts are specified in the dedicated CVL architecture document. ADR: [`ADR-CRITIC-001.md`](../adr/ADR-CRITIC-001.md).
""",
    "BUSINESS_EXAMPLES_POINTER": """# Business Agent Examples (out of platform canon)

Problem Radar, Vendor Discovery, and Organization Worker examples are **reference business agents**, not Harness platform architecture.

Each business agent documents its own design under `agents/<name>/ARCHITECTURE.md`. Platform composition patterns: [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) · [`AGENT_CREATION_GUIDE.md`](../AGENT_CREATION_GUIDE.md).
""",
    "IMPLEMENTATION_POINTER": """# Phase L — Agent OS Readiness (implementation pointer)

Agent OS readiness directives and phase-L checklists live in the implementation plan, not in architecture canon.

**See:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md) · [`plan/phases/platform-quality.md`](../plan/phases/platform-quality.md)
""",
}

SECTION_HEADER = re.compile(r"^# (\d+)\. (.+)$", re.MULTILINE)


def parse_sections(text: str) -> list[tuple[int, str, str]]:
    matches = list(SECTION_HEADER.finditer(text))
    sections: list[tuple[int, str, str]] = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((num, title, text[start:end].rstrip() + "\n"))
    return sections


def relink_section_refs(body: str, section_to_file: dict[int, str]) -> str:
    """Rewrite single §N references to linked doc paths where possible."""

    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        target = section_to_file.get(n)
        if target and target not in POINTER_FILES:
            return f"[§{n}]({target}.md)"
        return m.group(0)

    return re.sub(r"§(\d+)", repl, body)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")

    # Skip file title block before first numbered section
    first_section = SECTION_HEADER.search(text)
    if not first_section:
        raise SystemExit("No sections found")
    preamble = text[: first_section.start()].strip()

    sections = parse_sections(text)
    section_to_file = {n: SECTION_MAP.get(n, "PLATFORM_FOUNDATION") for n, _, _ in sections}

    buckets: dict[str, list[str]] = {}
    for num, _title, body in sections:
        target = SECTION_MAP.get(num, "PLATFORM_FOUNDATION")
        if target in POINTER_FILES:
            continue
        buckets.setdefault(target, []).append(body)

    ARCH.mkdir(parents=True, exist_ok=True)

    for name, bodies in buckets.items():
        title = FILE_TITLES.get(name, name.replace("_", " ").title())
        header = (
            f"# {title}\n\n"
            f"**Status:** Canonical architecture (decomposed from platform canon)  \n"
            f"**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  \n"
            f"**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../IDEAL_HARNESS_AI_ARCHITECTURE.md)\n\n"
            f"---\n\n"
        )
        content = header + "\n\n---\n\n".join(
            relink_section_refs(b, section_to_file) for b in bodies
        )
        (ARCH / f"{name}.md").write_text(content + "\n", encoding="utf-8")

    for name, content in POINTER_FILES.items():
        (ARCH / f"{name.replace('_POINTER', '')}.md").write_text(
            content.strip() + "\n", encoding="utf-8"
        )

    # Extract RAG-related chunks from PLATFORM_FOUNDATION §7.1.2 into RAG file if not exists
    rag_path = ARCH / "RAG_AND_RETRIEVAL.md"
    if not rag_path.exists():
        rag_path.write_text(
            """# RAG and Retrieval Architecture

**Status:** Canonical architecture  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Catalog:** Integration category `rag` in [`INTEGRATIONS.md`](../INTEGRATIONS.md)  
**Implementation plan:** [`plan/phases/rag-context-memory.md`](../plan/phases/rag-context-memory.md)

---

RAG and retrieval are Tier-0 platform capabilities consumed through Nexus policy and `ToolRuntime`. Agents MUST NOT embed vendor-specific vector stores directly.

## Design principles

- Retrieval is a **tool-backed** capability (`rag.*` tools), not ad-hoc agent HTTP calls.
- Indexing, chunking, and embedding profiles are integration contracts.
- Context assembly consumes retrieval results through the context engineering layer ([`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md)).
- Memory scopes ([`MEMORY_ARCHITECTURE.md`](../MEMORY_ARCHITECTURE.md)) and RAG indices are distinct namespaces.

## Related canon

- Platform foundation §7.1.2 (integration categories) — [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md)
- Tool runtime — [`TOOLS_RUNTIME.md`](TOOLS_RUNTIME.md)
- Unified execution §42.12 ToolRuntime — [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md)

See [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md) layer 14 (RAG and Retrieval).
""",
            encoding="utf-8",
        )

    # Policy, identity, registry stubs from §42 cross-refs
    for stub_name, stub_title, stub_body in [
        (
            "POLICY_AND_GOVERNANCE",
            "Policy and Governance",
            "Policy-first execution: `PolicyEngine`, `RuntimePolicyBundle`, pre/post hooks. "
            "Authoring: [`AGENT_CREATION_GUIDE.md`](../AGENT_CREATION_GUIDE.md) Appendix H. "
            "Runtime contracts: [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.11, §42.37.",
        ),
        (
            "IDENTITY_TRUST_AND_TENANCY",
            "Identity, Trust, and Tenancy",
            "User/service/agent identity, tenant context, delegation audit, secrets layer. "
            "See audit map layer 4 and security closeout in the implementation plan.",
        ),
        (
            "REGISTRY_ARCHITECTURE",
            "Registry Architecture",
            "Agent, tool, skill, and integration registries; snapshots and conformance CI. "
            "Implementation: Phase REG in [`plan/phases/registry-capability.md`](../plan/phases/registry-capability.md).",
        ),
        (
            "CAPABILITY_GRAPH",
            "Capability Graph Architecture",
            "Environment capability slices, blast-radius wiring, graph-based policy. "
            "Implementation: Phase CG in [`plan/phases/registry-capability.md`](../plan/phases/registry-capability.md).",
        ),
        (
            "PROMPT_REGISTRY",
            "Prompt Engineering and Prompt Registry",
            "Versioned prompts, compilation layers (system, task, policy, context, memory). "
            "Implementation: Phase PE in [`plan/phases/registry-capability.md`](../plan/phases/registry-capability.md).",
        ),
        (
            "SECURITY_AND_DATA_GOVERNANCE",
            "Security and Data Governance",
            "V-SEC bridge, data classification, middleware assembly. "
            "Implementation: Phase SEC in [`plan/phases/governance-security.md`](../plan/phases/governance-security.md).",
        ),
        (
            "COST_AND_RESOURCE_GOVERNANCE",
            "Cost and Resource Governance",
            "Token/cost metering, budgets, policy bundles. "
            "Implementation: Phase COST in [`plan/phases/governance-security.md`](../plan/phases/governance-security.md).",
        ),
        (
            "REASONING_AND_PLANNING",
            "Reasoning, Planning, and Cognition",
            "Structured planning contracts, `DecisionRecord`, planner strategies. "
            "Authoring: [`AGENT_CREATION_GUIDE.md`](../AGENT_CREATION_GUIDE.md) Appendix I. "
            "Flow narrative: [`NEXUS_EXECUTION_FLOW_REFERENCE.md`](../NEXUS_EXECUTION_FLOW_REFERENCE.md).",
        ),
        (
            "SUBAGENTS_AND_COORDINATION",
            "Subagents and Multi-Agent Coordination",
            "Delegation, `SubtaskContract`, memory namespaces, merge policies. "
            "Canon: [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.14. "
            "Authoring: Appendix I §I.6 in agent creation guide.",
        ),
        (
            "INTERFACE_AND_INTAKE",
            "Interface and Task Intake",
            "Normalized `TaskEnvelope`, API/CLI/worker convergence, tenant metadata. "
            "See audit map layer 3.",
        ),
        (
            "TESTING_CI_AND_ARCHITECTURE_GATES",
            "Testing, CI, and Architecture Gates",
            "Regression gate (`pytest -m gate`), harness boundary scripts, observability gates. "
            "Verification commands: [`AGENTS.md`](../../AGENTS.md).",
        ),
    ]:
        path = ARCH / f"{stub_name}.md"
        if not path.exists():
            path.write_text(
                f"# {stub_title}\n\n"
                f"**Status:** Canonical architecture  \n"
                f"**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)\n\n"
                f"---\n\n{stub_body}\n",
                encoding="utf-8",
            )

    print(f"Wrote {len(list(ARCH.glob('*.md')))} files under {ARCH}")


if __name__ == "__main__":
    main()
