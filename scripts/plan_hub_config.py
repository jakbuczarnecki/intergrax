# © Artur Czarnecki. All rights reserved.
"""Per-domain plan hub split configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlanSplitConfig:
    domain: str
    hub_h3_prefixes: tuple[str, ...] = ()
    keep_h2_prefixes: tuple[str, ...] = ()
    move_h2_prefixes: tuple[str, ...] = ()
    move_h2_appendix: bool = True
    move_h2_phase_closeout: bool = False
    move_h2_detail_prefixes: tuple[str, ...] = ()
    split_h3_in_h2_prefixes: tuple[str, ...] = ()
    foreign_block_start: str | None = None
    foreign_block_end: str | None = None
    foreign_stub: str | None = None
    dedupe_sync_footer: bool = False


PLATFORM = PlanSplitConfig(
    domain="PLATFORM_FOUNDATION",
    hub_h3_prefixes=(
        "6.1 Harness platform maintenance",
        "6.1av ",
        "6.1p ",
        "6.2af ",
        "6.3 ",
        "6.3a ",
    ),
    move_h2_phase_closeout=True,
    move_h2_detail_prefixes=(
        "0. Architecture at a glance",
        "2. Map:",
        "3. Implementation Phases",
        "4. Priority Order",
        "Phase AUDIT-IDEAL",
        "Documentation model",
        "5. Definition of Done",
        "1. Plan Objective",
    ),
    dedupe_sync_footer=True,
)

CRITIC = PlanSplitConfig(
    domain="CRITIC_VERIFICATION",
    hub_h3_prefixes=(
        "6.1ak ",
        "6.1av ",
        "6.1aw ",
    ),
    keep_h2_prefixes=(
        "Phase AUDIT-IDEAL",
        "Audit §CVL-4",
        "Audit –CVL-4",
    ),
    move_h2_prefixes=(
        "Audit §CVL-1",
        "Audit §CVL-2",
        "Audit §CVL-3",
        "Audit –CVL-1",
        "Audit –CVL-2",
        "Audit –CVL-3",
        "Sprint CVL-LC-",
        "Phase CRITIC_VERIFICATION-LC",
    ),
    foreign_block_start="(Global)",
    foreign_block_end="# Audit Result: Critic",
    foreign_stub="""---

## Cross-domain phase registers (canonical elsewhere)

Foreign **Platform / ORCH / FLOW / FAUDIT** registers were removed from this hub.

| Need | Canonical source |
|------|------------------|
| Platform gate maintenance | [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) §6.1 |
| ORCH closeout | [`ORCHESTRATION.md`](ORCHESTRATION.md) |
| FAUDIT-32 | [`plan/satellites/PLATFORM_FOUNDATION_phase_closeout.md`](plan/satellites/PLATFORM_FOUNDATION_phase_closeout.md) |
| FLOW depth | [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) |

**Audit history (CVL-1…3, LC closeout):** [`plan/satellites/CRITIC_VERIFICATION_audit_history.md`](plan/satellites/CRITIC_VERIFICATION_audit_history.md)

---
""",
    move_h2_appendix=True,
)

NEXUS_FLOW = PlanSplitConfig(
    domain="NEXUS_EXECUTION_FLOW",
    hub_h3_prefixes=(
        "6.1aw ",
        "6.1av ",
    ),
    keep_h2_prefixes=(
        "Phase AUDIT-IDEAL",
        "Phase FLOW",
        "Phase FLOW-CTL",
    ),
    foreign_block_start="### 6.2ak Phase CRIT-V execution order",
    foreign_block_end="## Phase FLOW — Nexus execution depth",
    foreign_stub="""---

## Cross-domain registers (canonical elsewhere)

| Need | Source |
|------|--------|
| CRIT-V | [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |
| ORCH | [`ORCHESTRATION.md`](ORCHESTRATION.md) |
| Platform §6 | [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) |

---
""",
    move_h2_appendix=True,
)

ORCHESTRATION_CFG = PlanSplitConfig(
    domain="ORCHESTRATION",
    hub_h3_prefixes=(
        "6.1aw ",
        "6.1av ",
    ),
    keep_h2_prefixes=(
        "Phase AUDIT-IDEAL",
        "Phase ORCH",
        "Phase ORCH-STRAT",
        "Phase ORCH-CONFIG",
        "Phase ORCH-5",
        "Phase ORCH-6",
    ),
    foreign_block_start="### 6.1b Phase N (complete)",
    foreign_block_end="## Phase ORCH — Orchestration control plane closeout",
    foreign_stub="""---

## Cross-domain pasted content removed

| Need | Source |
|------|--------|
| Platform appendices | [`plan/satellites/PLATFORM_FOUNDATION_appendices.md`](plan/satellites/PLATFORM_FOUNDATION_appendices.md) |
| Master registers | [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md) |

---
""",
    move_h2_appendix=False,
)

UAEP = PlanSplitConfig(
    domain="UNIFIED_EXECUTION_RUNTIME",
    hub_h3_prefixes=("6.1av ",),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_prefixes=(
        "Phase SEC-PLANES",
        "Phase SEC-ENT",
        "Phase GOV-AUDIT",
        "Phase SEC —",
        "Phase COST —",
        "Phase CLEAN",
        "Phase GR-DOC",
    ),
    foreign_block_start="### ORCH — Master register",
    foreign_block_end="## Phase SEC-PLANES",
    foreign_stub="""---

## Cross-domain ORCH/flow registers removed

See [`ORCHESTRATION.md`](ORCHESTRATION.md) · [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md).

---
""",
    move_h2_appendix=True,
)

INTEGRATIONS_CFG = PlanSplitConfig(
    domain="INTEGRATIONS",
    keep_h2_prefixes=(
        "Phase AUDIT-IDEAL",
        "Phase H-INT-GRAPH",
        "Phase INTEGRATIONS-LC",
    ),
    move_h2_prefixes=("Phase INT —",),
    move_h2_appendix=True,
)

EXP_DX = PlanSplitConfig(
    domain="EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    keep_h2_prefixes=(
        "Phase DX-IDEA",
        "Phase MVP-EVOL",
        "4. Priority",
        "5. Definition",
    ),
    move_h2_prefixes=(
        "Phase DX-LC",
        "Phase DX —",
        "Phase AA —",
        "Phase W-OPS",
        "Phase EVAL —",
    ),
    move_h2_detail_prefixes=(
        "Phase AUDIT-IDEAL",
        "4. Priority Order",
    ),
    move_h2_appendix=True,
)

AGENT_CONTRACTS = PlanSplitConfig(
    domain="AGENT_CONTRACTS_AND_ASSEMBLY",
    hub_h3_prefixes=("6.1",),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    keep_h2_prefixes=(
        "Agent architecture completion — executive",
        "Phase AUDIT-IDEAL",
    ),
    move_h2_prefixes=(
        "Phase ACP —",
        "Phase ACP-CLOSE",
        "Phase ACP-FINISH",
        "Phase ACP-DEPTH",
        "Phase AS —",
        "Phase PE —",
        "Phase REG —",
        "Phase CG —",
        "Phase ACP-LC",
    ),
    move_h2_appendix=True,
)

SKILLS_CFG = PlanSplitConfig(
    domain="SKILLS",
    hub_h3_prefixes=("6.1av ", "6.1aw "),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_prefixes=("Phase TS —", "Phase SKILLS-LC"),
    move_h2_appendix=True,
)

CONTEXT_CFG = PlanSplitConfig(
    domain="CONTEXT_ENGINEERING",
    hub_h3_prefixes=("6.1",),
    keep_h2_prefixes=("Status summary",),
    move_h2_prefixes=(
        "Layer audit register",
        "Gap traceability",
        "Phase CE-",
        "Master deliverables",
        "Inherited closeout",
        "Sprints",
        "Phase CONTEXT",
    ),
    move_h2_detail_prefixes=(
        "Verification commands",
        "Explicitly out of scope",
        "Suggested PR order",
    ),
    move_h2_appendix=True,
)

CODE_CRAFT_CFG = PlanSplitConfig(
    domain="CODE_CRAFT",
    hub_h3_prefixes=("6.1",),
    keep_h2_prefixes=("Delivery rules",),
    move_h2_prefixes=(
        "Phase ECC-",
        "Audit §",
        "Sprint S",
        "Phase CODE_CRAFT-LC",
    ),
    move_h2_appendix=True,
)

REASONING_CFG = PlanSplitConfig(
    domain="REASONING_AND_COGNITION",
    hub_h3_prefixes=("6.1",),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    keep_h2_prefixes=("Phase COG-PROD —",),
    move_h2_prefixes=(
        "Phase COG-DOC",
        "Phase COG-DEPTH",
        "COG-DEPTH —",
        "Phase COG-LC",
    ),
    move_h2_appendix=True,
)

OBSERVABILITY_CFG = PlanSplitConfig(
    domain="OBSERVABILITY",
    hub_h3_prefixes=("6.1",),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_prefixes=(
        "Phase IDEAL-L3",
        "Phase OBS —",
        "Phase OBS-BUS",
        "Phase EBE —",
        "Phase OBS-EVOL",
        "Phase OBSERVABILITY-LC",
    ),
    move_h2_appendix=True,
)

ADAPTIVE_HARNESS_INTELLIGENCE_CFG = PlanSplitConfig(
    domain="ADAPTIVE_HARNESS_INTELLIGENCE",
    hub_h3_prefixes=("6.1",),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_prefixes=("Phase W-ADAPT —", "Phase AHI-LC"),
    move_h2_appendix=True,
)

ELASTIC_CAPACITY_CFG = PlanSplitConfig(
    domain="ELASTIC_CAPACITY_AND_SCALING",
    hub_h3_prefixes=("6.1",),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_prefixes=(
        "Phase ECP-",
        "Phase ECP —",
        "Phase ELASTIC",
    ),
    move_h2_appendix=True,
)

LLM_ADAPTERS_CFG = PlanSplitConfig(
    domain="LLM_ADAPTERS",
    keep_h2_prefixes=(
        "Phase AUDIT-IDEAL",
        "Phase M-LLM-X",
    ),
    move_h2_prefixes=(
        "Layer Completion Mode",
        "Phase LLM-LC",
    ),
    move_h2_appendix=True,
)

TOOLS_CFG = PlanSplitConfig(
    domain="TOOLS",
    hub_h3_prefixes=("6.1av ", "6.1aw "),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_prefixes=(
        "Phase LEG —",
        "Phase TS —",
        "Phase TOOL-ENG-DOC —",
        "Layer completion audit",
        "Layer completion final",
        "Layer completion sprints",
        "Phase TOOL-ENG —",
        "Phase TOOLS-LC",
    ),
    move_h2_appendix=True,
)

TIER3_PLAN = PlanSplitConfig(
    domain="TIER3_APPLICATION_ENVIRONMENT",
    hub_h3_prefixes=("6.2y ", "6.1"),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_detail_prefixes=(
        "Architecture fidelity matrix",
        "Master implementation backlog",
        "Cross-plan — §43",
        "Fidelity verification gates",
    ),
    move_h2_prefixes=(
        "Phase H-APP",
        "Phase H-APP-DOC",
        "Phase H-APP-WIRING",
        "Phase H-APP-CON",
        "Phase H-APP-EVOL",
        "Phase H-APP-OPS",
        "Phase H-APP-FREEZE",
        "Tier-3 Layer Completion",
        "Phase TIER3-LC",
    ),
    move_h2_appendix=True,
)

MEMORY_CFG = PlanSplitConfig(
    domain="MEMORY",
    hub_h3_prefixes=("6.1",),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_prefixes=(
        "Phase MEM-VEC",
        "Phase MEM —",
        "Phase MEM-DEPTH",
        "Phase CTX —",
        "Phase MEMORY-LC",
    ),
    move_h2_appendix=True,
)

RAG_CFG = PlanSplitConfig(
    domain="RAG",
    hub_h3_prefixes=("6.1",),
    split_h3_in_h2_prefixes=("Phase AUDIT-IDEAL",),
    move_h2_detail_prefixes=(
        "Audit traceability matrix",
        "Full implementation task register",
        "Step-by-step rollout",
    ),
    move_h2_prefixes=(
        "Phase RAG —",
        "Phase M-RAG",
        "Phase M-RAG-DEPTH",
        "Phase M-RAG-GRAPH",
        "Phase M-RAG-ITERATION",
        "Phase RAG-LC",
        "Layer Completion",
    ),
    move_h2_appendix=True,
)

CONFIGS: dict[str, PlanSplitConfig] = {
    c.domain: c
    for c in (
        PLATFORM,
        CRITIC,
        NEXUS_FLOW,
        ORCHESTRATION_CFG,
        UAEP,
        INTEGRATIONS_CFG,
        EXP_DX,
        AGENT_CONTRACTS,
        LLM_ADAPTERS_CFG,
        TOOLS_CFG,
        TIER3_PLAN,
        MEMORY_CFG,
        RAG_CFG,
        SKILLS_CFG,
        CONTEXT_CFG,
        CODE_CRAFT_CFG,
        REASONING_CFG,
        OBSERVABILITY_CFG,
        ADAPTIVE_HARNESS_INTELLIGENCE_CFG,
        ELASTIC_CAPACITY_CFG,
    )
}
