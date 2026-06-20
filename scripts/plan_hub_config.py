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
| FAUDIT-32 | [`plan/plan/PLATFORM_FOUNDATION_phase_closeout.md`](plan/plan/PLATFORM_FOUNDATION_phase_closeout.md) |
| FLOW depth | [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) |

**Audit history (CVL-1…3, LC closeout):** [`plan/plan/CRITIC_VERIFICATION_audit_history.md`](plan/plan/CRITIC_VERIFICATION_audit_history.md)

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
| Platform appendices | [`plan/plan/PLATFORM_FOUNDATION_appendices.md`](plan/plan/PLATFORM_FOUNDATION_appendices.md) |
| Master registers | [`plan/plan/PLATFORM_FOUNDATION_master_registers.md`](plan/plan/PLATFORM_FOUNDATION_master_registers.md) |

---
""",
    move_h2_appendix=False,
)

UAEP = PlanSplitConfig(
    domain="UNIFIED_EXECUTION_RUNTIME",
    keep_h2_prefixes=(
        "Phase AUDIT-IDEAL",
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

See [`ORCHESTRATION.md`](ORCHESTRATION.md) · [`plan/plan/PLATFORM_FOUNDATION_master_registers.md`](plan/plan/PLATFORM_FOUNDATION_master_registers.md).

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
        "Phase AUDIT-IDEAL",
        "Phase DX-IDEA",
        "4. Priority",
        "5. Definition",
        "Phase MVP-EVOL",
    ),
    move_h2_prefixes=(
        "Phase DX-LC",
        "Phase DX —",
        "Phase AA —",
        "Phase W-OPS",
        "Phase EVAL —",
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
    )
}
