# SCENARIO-FACTORY-X — R2 Amendment Dossier

**Status:** `DOCUMENTATION_PREPARED` · **NOT** `CANONICAL_R2_APPROVED` · **NOT** `IMPLEMENTATION_AUTHORIZED`  
**Campaign ticket:** [`T-IGX-070-A`](../../../../../governance/tickets/T-IGX-070-A.md)  
**Canonical binding authority:** R1 [`SFX-MASTER-PLAN.md`](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) — **unchanged**  
**Baseline HEAD (audit):** `b4146cbfab9858c058fc44df2ad43d8670d4925d` (`development`)  
**External input (non-canonical):** `INTEGRAX_SFX_R2_STRATEGIC_BLUEPRINT_2026-11-23.md`

## Campaign verdict

| State | Result |
| --- | --- |
| `DOCUMENTATION_PREPARED` | **YES** — package under this tree |
| `INDEPENDENT_REVIEW_PASS` | **PASS_WITH_FINDINGS** — [`r2/SFX_R2_INDEPENDENT_AUDIT_MATRIX.md`](./r2/SFX_R2_INDEPENDENT_AUDIT_MATRIX.md) |
| `OPERATOR_APPROVAL_PENDING` | **YES** — OD backlog §OD |
| `CANONICAL_R2_APPROVED` | **NO** |
| `IMPLEMENTATION_AUTHORIZED` | **NO** — ARCH-FREEZE, GATE-SFX-ENTRY |

## Package map

| # | Deliverable | Artifact |
| --- | --- | --- |
| 1 | Architecture & target product (R2) | [`r2/SFX_R2_TARGET_PRODUCT_ARCHITECTURE.md`](./r2/SFX_R2_TARGET_PRODUCT_ARCHITECTURE.md) |
| 2 | R1→R2 delta & reuse matrix | [`r2/SFX_R2_DELTA_REUSE_MATRIX.md`](./r2/SFX_R2_DELTA_REUSE_MATRIX.md) |
| 3 | Autonomous mission specification | [`r2/SFX_R2_AUTONOMOUS_MISSION_SPEC.md`](./r2/SFX_R2_AUTONOMOUS_MISSION_SPEC.md) |
| 4 | Proposed contracts & schemas | [`design/schemas/proposed/`](./design/schemas/proposed/) · [`r2/SFX_R2_CONTRACT_CATALOG.md`](./r2/SFX_R2_CONTRACT_CATALOG.md) |
| 5 | SFX↔SEF cross-program amendment | [`r2/SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md`](./r2/SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md) |
| 6 | Proposed ADR changes | [`proposals/SFX_R2_ADR_AMENDMENTS_PROPOSED.md`](./proposals/SFX_R2_ADR_AMENDMENTS_PROPOSED.md) |
| 7 | Requirements traceability (R2 rows) | [`proposals/SFX_REQUIREMENTS_TRACEABILITY_R2_PROPOSAL.json`](./proposals/SFX_REQUIREMENTS_TRACEABILITY_R2_PROPOSAL.json) |
| 8 | WBS & dependency DAG overlay | [`r2/SFX_R2_WBS_DAG_AMENDMENT.md`](./r2/SFX_R2_WBS_DAG_AMENDMENT.md) |
| 9 | UX / Studio mission cockpit | [`r2/SFX_R2_UX_MISSION_EXTENSION.md`](./r2/SFX_R2_UX_MISSION_EXTENSION.md) |
| 10 | Security & adversarial qualification | [`r2/SFX_R2_SECURITY_ADVERSARIAL_QUALIFICATION.md`](./r2/SFX_R2_SECURITY_ADVERSARIAL_QUALIFICATION.md) |
| 11 | REF/GEN evaluation plan | [`r2/SFX_R2_REF_GEN_EVALUATION_PLAN.md`](./r2/SFX_R2_REF_GEN_EVALUATION_PLAN.md) |
| 12 | Commercialization roadmap | [`r2/SFX_R2_COMMERCIALIZATION_ROADMAP.md`](./r2/SFX_R2_COMMERCIALIZATION_ROADMAP.md) |
| 13 | Operator decision register | [`r2/SFX_R2_OPERATOR_DECISION_REGISTER.md`](./r2/SFX_R2_OPERATOR_DECISION_REGISTER.md) |
| 14 | Full lifecycle stage I/O | [`r2/SFX_R2_LIFECYCLE_STAGE_SPEC.md`](./r2/SFX_R2_LIFECYCLE_STAGE_SPEC.md) |
| 15 | Master plan amendment proposal | [`proposals/SFX-MASTER-PLAN-R2-AMENDMENT-PROPOSAL.md`](./proposals/SFX-MASTER-PLAN-R2-AMENDMENT-PROPOSAL.md) |

## Preserved architecture decisions (R2 non-negotiable)

1. **Option B:** Tier-2 SFX product semantics, Tier-3 Studio/API/MCP/CLI, existing Intergrax Core/runtime.
2. **SFX owns** business discovery, goals, candidate selection, business-outcome evaluation views.
3. **SEF-X owns** autonomous software engineering, coding backends, engineering jobs, bounded repair, workspaces, engineering evidence.
4. **Intergrax Runtime** remains the only scenario-execution authority (`execute_scenario_task` / NexusLoop).
5. **Reuse** governance, persistence, tenant, approval, evidence, checkpoints — no second engine/FSM/approval/registry.
6. **SFX never** grants production authority or independently certifies generated scenarios.
7. **Gates:** ARCH-FREEZE, GATE-SFX-ENTRY, GATE-SEF-CONSUME, SCENARIO-GATE, PROD-Q.

## Code-surface verification @ `b4146cbf`

| Check | Finding |
| --- | --- |
| `agents/scenario_factory/` | **Absent** — R1 W3+ not started (expected pre-GATE-SFX-ENTRY) |
| SFX JSON schemas (17) | **Present** under `design/schemas/` (PROPOSED design drafts) |
| `EngineeringHandoffPort` in product code | **Not implemented** — specified in R1 plan only |
| SEF `CON-N-01` human submit | **Binding** in [`ASF-SEF-MASTER-PLAN.md`](../../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md) |
| Platform execution | **Reuse** HostTaskExecutionPort / PolicyEngine / CW repositories (per R1 inventory) |

## Critical path (earliest authorized Invoice→POC slice)

```text
ARCH-FREEZE CLOSED → GATE-SFX-ENTRY → W3 (R2-W3-01..05 / SFX-P5*) → W4 (discovery+handoff; HO-2 human until OD-SEF/R2 grant) →
W5 (evaluator + mission driver PROPOSED) → REF-1 autonomous loop (R2-W7-01) → internal sandbox demo gate (NOT production)
```

HO-2 blocks **autonomous multi-submit** until [`SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md`](./r2/SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md) is operator-approved on the SEF track.

## Publication status

| Item | Status |
| --- | --- |
| R2 documentation commit (070-A) | `10cf95df89e806191d5ff801917ece91ff144bad` on `origin/development` |
| Environment-only commit (out of SFX scope) | `d53be07acac08c20d8f4ac8b00746c27d709d071` |
| R2 corrective + finalization batch | **PR #756** branch `integ/sfx-r2-corr-001` — **not merged** |
| `REMOTE_PUBLICATION_SHA` | **NONE** until PR merge |
| `CANONICAL_R2_APPROVED` | **NO** |
| Canonical master plan merge | **Blocked** — requires operator R2 adoption |
| `SFX_PRE_IMPLEMENTATION_READY` | **Not granted** |

Handoff SSOT: [SFX_R2_PUBLICATION_HANDOFF.md](./SFX_R2_PUBLICATION_HANDOFF.md).


## R2 readiness for operator approval

**Ready for operator review:** design package is internally consistent, traceable, and implementable **after** standard gates. **Not ready** for implementation: HO-2/ARCH-FREEZE/ADR acceptance blockers remain.

**Next authorized step after ARCH-FREEZE:** reconcile R2 overlay into wave planning ticket, operator `Approved:` on T-IGX-070-A and cross-track OD for delegation grant; then GATE-SFX-ENTRY checklist unchanged from R1.