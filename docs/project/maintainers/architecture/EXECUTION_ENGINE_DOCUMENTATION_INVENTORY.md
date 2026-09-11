# Execution Engine — Documentation Inventory (P0)

**Status:** `INVENTORY` (classification only — no mass rewrites)

**Ownership reference (unchanged):**

```text
Decision = WHAT · AD = WHO · Governance = WHETHER
ExecutionRuntime = lifecycle/how · Nexus = HOW/WHEN topology
HITL = human authority · 5E = recovery · 5F = evidence (owned elsewhere)
```

---

## Maintainer entry point (CANONICAL)

Single maintainer hub — navigation and status only (`MAINTAINER_HUB`):

```text
docs/project/maintainers/architecture/EXECUTION_ENGINE.md
├─ Execution Runtime      → UNIFIED_EXECUTION_RUNTIME.md
├─ Governance & Authority → GOVERNED_EXECUTION.md + DECISION_APPROVAL_GOVERNANCE.md + NPSC-5D docs
├─ Nexus                  → NEXUS_EXECUTION_FLOW.md
├─ Child Execution        → UEA + NPSC-5B maintainers docs
├─ Multi-Agent            → NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md
├─ Recovery               → NPSC_5E_RECOVERY_CHECKPOINT_RETRY_ARCHITECTURE.md
├─ Evidence               → NPSC_5F_* + OBSERVABILITY.md
└─ Scale / Resilience     → ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md
```

Hub links to canonical owners only; it does not own normative semantics (see hub §Purpose).

---

## Domain inventory

### Execution Runtime

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md` | **CANONICAL** | UEA — conflict winner per enterprise verification |
| `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` | **CANONICAL** | UER lifecycle owner narrative |
| `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` | **CANONICAL** (plan pair) | Maintainer plan ↔ arch pair |
| `docs/project/architecture/UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md` | **CANONICAL** (supporting) | Implementation map |
| `docs/project/architecture/UNIFIED_EXECUTION_IMPLEMENTATION_READINESS.md` | **FUTURE / PLANNED** | Readiness tracking |
| `docs/project/maintainers/qualification/EXECUTION_ENGINE_ENTERPRISE_VERIFICATION.md` | **CANONICAL** (qualification record) | Post-NPSC-3G certification snapshot |
| `docs/audit_results/2026-08-18/EXECUTION_RUNTIME.md` | **HISTORICAL** | Point-in-time audit |

### Nexus

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | **CANONICAL** | Topology / flow owner |
| `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` | **CANONICAL** (plan pair) | Plan counterpart |
| `docs/project/maintainers/architecture/NPSC_5B_R3_NEXUS_FANOUT_CONTRACT_REQUIREMENT.md` | **CANONICAL** (constraint) | Fan-out contract |

### Governance & Authority

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/architecture/DECISION_APPROVAL_GOVERNANCE.md` | **CANONICAL** | Governance plane |
| `docs/project/maintainers/architecture/NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md` | **CANONICAL** | NPSC-5D plane |
| `docs/project/maintainers/architecture/NPSC_5C_DECISION_INTEGRATION_REQUIREMENT.md` | **CANONICAL** (constraint) | Decision integration |

### Lineage

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/maintainers/qualification/NPSC_5E_P0A_EXECUTION_LINEAGE_BASELINE_QUALIFICATION.md` | **CANONICAL** (qual) | P0A baseline |
| `docs/project/maintainers/architecture/DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_ARCHITECTURE_R1.md` | **CANONICAL** | DG-001 architecture |
| `docs/project/maintainers/qualification/DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_READ_INTEGRATION_R1.md` | **CANONICAL** (qual) | R1 final qual record |

### Retry / Checkpoint / Recovery

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/maintainers/architecture/NPSC_5E_RECOVERY_CHECKPOINT_RETRY_ARCHITECTURE.md` | **CANONICAL** | 5E architecture hub |
| `docs/project/maintainers/qualification/NPSC_5E_FINAL_RECOVERY_PLANE_QUALIFICATION_AND_FREEZE.md` | **CANONICAL** (freeze) | Latest composite freeze |
| Nested `NPSC_5E_R*` qualification docs | **CANONICAL** (historical chain) | Predecessor freeze records |

### Agent distribution / fan-out

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/maintainers/architecture/NPSC_5B_CROSS_SYSTEM_FANOUT_OWNERSHIP_RECONCILIATION.md` | **CANONICAL** | Ownership reconciliation |

### HITL

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` | **CANONICAL** | HITL + reliability |
| `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md` | **CANONICAL** (plan pair) | Plan counterpart |

### Tools

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/architecture/TOOLS.md` | **CANONICAL** | Tool plane |

### Observability / Evidence

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/architecture/OBSERVABILITY.md` | **CANONICAL** | Observability spine |
| `docs/project/architecture/assets/fullsize/observability-evidence-spine.md` | **CANONICAL** (diagram asset) | Evidence spine diagram |
| `docs/project/maintainers/architecture/NPSC_5F_EXECUTION_EVIDENCE_REPLAY_OBSERVABILITY_ARCHITECTURE.md` | **CANONICAL** | 5F evidence plane; R1 **FROZEN / PASS**; R2 active |
| `docs/project/maintainers/qualification/NPSC_5F_R1_FINAL_DURABLE_EVIDENCE_COMMIT_TENANT_INTEGRITY_QUALIFICATION_AND_FREEZE.md` | **CANONICAL** (freeze) | 5F/R1 Final |
| `docs/project/maintainers/qualification/NPSC_5F_R2_JOURNAL_COMPLETENESS_ORDERING.md` | **ACTIVE** | R2 implemented; await R2 Final freeze |
| `docs/project/technical/platform/execution_evidence_*.md` | **SUPPORTING** / **PLANNED** | Implementation plans where present |

### Scale / resilience

| Document | Classification | Rationale |
| --- | --- | --- |
| `docs/project/maintainers/architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md` | **CANONICAL** | Maintainer scale/resilience architecture |
| `docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md` | **CANONICAL** (qual) | P0 inventory |
| `docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md` | **CANONICAL** (qual) | W0 guardrails |
| `docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W1_ADMISSION_DEADLINE.md` | **CANONICAL** (qual) | W1 admission / deadline |

### Qualification acceleration (this initiative)

| Document | Classification |
| --- | --- |
| `docs/project/maintainers/architecture/EXECUTION_ENGINE.md` | **CANONICAL** (`MAINTAINER_HUB`) |
| `EXECUTION_CERTIFICATION_ACCELERATION_P0.md` | **CANONICAL** (P0 inventory) |
| `EXECUTION_CERTIFICATION_ACCELERATION_R1.md` | **CANONICAL** (qual) — R1 **QUALIFIED** |
| `EXECUTION_CERTIFICATION_ACCELERATION_R2.md` | **CANONICAL** (qual) — R2 **QUALIFIED** |
| `EXECUTION_CERTIFICATION_ACCELERATION_R3.md` | **CANONICAL** (qual) — R3 **QUALIFIED** |
| `EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md` | **CANONICAL** (supporting architecture) — **R3 QUALIFIED**; R3A ENV contract |

---

## Duplicate / stale / contradictory notes

| Issue | Classification | Detail |
| --- | --- | --- |
| Multiple `UNIFIED_EXECUTION_RUNTIME.md` under `docs/audit_results/legacy/**` | **DUPLICATE** / **HISTORICAL** | Superseded by `docs/project/architecture/` |
| UEA vs older audit narratives | **STALE** in legacy audits | UEA wins (documented in enterprise verification) |
| Plan vs arch pairs | **Not contradictory** | Intentional dual topology |

No **CONTRADICTORY** pair identified among current `docs/project/architecture` canonical Execution docs in P0 read scope.

---

## Diagram inventory

| Asset / doc | Topic | Classification |
| --- | --- | --- |
| `UNIFIED_EXECUTION_ARCHITECTURE_DIAGRAMS.md` | Execution flow | **CORRECT** (canonical diagram pack) |
| `EXECUTION_ENGINE_ENTERPRISE_VERIFICATION.md` §2 mermaid | Lifecycle entry | **CORRECT** (matches UEA/UER) |
| `docs/project/architecture/assets/fullsize/orchestration-control-plane.md` | Nexus / control | **CORRECT** (supporting) |
| `docs/project/architecture/assets/fullsize/observability-evidence-spine.md` | Evidence | **CORRECT** |
| Legacy audit diagram copies | Various | **STALE** |
| End-to-end **qualification parallelization** diagram | Runner topology | **MISSING** (R1 doc above is text-only — acceptable until R1) |

**DIAGRAMS TO REPLACE:** none mandatory in P0; add R1 coordinator diagram when runner lands.

---

## Cross-links

- Maintainer hub: [`EXECUTION_ENGINE.md`](EXECUTION_ENGINE.md)
- Acceleration P0: [`EXECUTION_CERTIFICATION_ACCELERATION_P0.md`](../qualification/EXECUTION_CERTIFICATION_ACCELERATION_P0.md)
- Qualification acceleration architecture: [`EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md`](EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md)
