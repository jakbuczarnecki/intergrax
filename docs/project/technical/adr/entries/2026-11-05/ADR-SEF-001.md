# ADR-SEF-001: Engineering Execution Authority vs Scenario Runtime Plane

| Field | Value |
|-------|-------|
| **Status** | **Proposed** (OD-SEF-03 — preparation approved; **not Accepted**) |
| **Date** | 2026-11-05 |
| **Deciders** | Operator (OD-SEF-01/03); implementation gated on ARCH-FREEZE + contracts |
| **Related** | [`ASF_SEF_MASTER_ARCHITECTURE_AND_IMPLEMENTATION_PLAN.md`](../../../maintainers/plans/satellites/ASF_SEF_MASTER_ARCHITECTURE_AND_IMPLEMENTATION_PLAN.md) · [`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md`](../../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md) · [`ADR-CODECRAFT-001`](../2026-06-10/ADR-CODECRAFT-001.md) · [`ADR-HARNESS-001`](../2026-09-20/ADR-HARNESS-001.md) · [`T-IGX-SEF-A`](../../../../../governance/tickets/T-IGX-SEF-A.md) |

## Context

Intergrax needs a **bounded engineering factory** (intent → spec → discovery → build plan → execution → evidence → PR) that is **semantically separate** from **scenario production runtime** (admitted scenarios via Nexus). Prior corrective findings (GAP-CRIT-05) conflated DevOS/IDE workflows with Nexus execution authority.

Alternatives considered:

| Variant | Summary | Rejection rationale |
|---------|---------|---------------------|
| **A — Embedded factory core** | Engineering loop inside Nexus as peer to scenario agents | Second execution SSOT; violates SYS-INV-10 / single Nexus encapsulation |
| **B — External factory SSOT** | Separate governance/policy engine for engineering | Duplicate PolicyEngine / FRZ authority |
| **C + D-lite (selected)** | DevOS + engineering orchestrator + `CodingExecutionBackend`; scenario plane unchanged | OD-SEF-01 APPROVED |
| **Option 4 pilot** | Proof-only factory under `platform_proofs/` | **NOT AUTHORIZED** (OD-SEF-02) |

## Decision

Adopt **Variant C + D-lite**:

1. **Engineering Execution Authority (EEA)** — DevOS ticket/plan gates + PROPOSED engineering orchestrator (`agents/sef_orchestration/`, Class A/B) orchestrates **engineering jobs** only: workspace lease, spec/build plan, discovery, edit/test cycles, evidence assembly, PR preparation.
2. **Scenario Runtime Plane** — unchanged canonical chain: **`HostTaskExecutionPort` → `NexusLoop` → `PolicyEngine` → `RuntimeToolInvoker` / ToolRuntime** for admitted scenario execution.
3. **`CodingExecutionBackend`** — **provider-neutral** port for repo edits and local verification; adapters (e.g. IDE-hosted, CI, future vendors) are replaceable; **no** hard-coded production vendor selection in Tier-0/1.
4. **No second SSOT** for Nexus, ToolRuntime, PolicyEngine, FRZ/ECAP, or Cursor-as-runtime-authority.

### Platform boundary map

| Domain | Role in SEF | Engineering plane | Scenario plane |
|--------|-------------|-------------------|------------------|
| **DevOS / Governance tickets** | Admission before engineering jobs | **Requires** `plan_approved` + `verify_plan_approval.py`; `igx_gov ticket` at quality_gate | N/A for factory orchestration |
| **Nexus / UAEP** | Scenario execution root | REF demos via proof host only | Primary runtime |
| **Governance / PolicyEngine** | Authorization & side effects | Read-only allowlists for eng sandbox | Full path for admitted scenarios |
| **ToolRuntime** | Tool invocation & side effects | Sandbox-limited profiles | Production profiles + PROD-Q |
| **CodeCraft** | Ephemeral dev patterns (ADR-CODECRAFT-001) | Supporting eng sandbox | Not production execution owner |
| **Agent Distribution** | Package/install agents | Produce/version agent artifacts | Activate via distribution on hosts |
| **ENT-AUDIT-X / ECAP / FRZ** | Evidence & qualification | Candidate bundles; **no implementer B6** | Scoped qual records; production admission separate |

### Engineering job prerequisites

- Ticket lifecycle **`plan_approved`** with operator **`Approved:`** line and matching **plan digest**.
- Mechanical **`verify_plan_approval.py`** before **`in_progress`** engineering work.
- **Workspace / tenant:** engineering lease scoped to ticket paths; tenant context must not widen to cross-tenant state (align TENANT-X contracts at implementation).
- **Budgets:** job budget, bounded **repair** attempts (`RepairAttemptRecord`), cancel tokens, checkpoint paths for L2 resume (Master Plan Deliverable 6–7).
- **Evidence:** EngineeringEvidenceBundle + SHA ladder inputs; qualification and **scoped independent B6** only after ARCH-FREEZE chain (REQ-ASF-037); **production admission** remains PROD-Q / SCENARIO-GATE — factory evidence does not substitute.

## Consequences

### Positive

- Clear EEA vs scenario runtime split; reuse existing Nexus/Governance/ToolRuntime investments.
- Provider-neutral coding backends support replaceability (EBH-5 lineage) without new platform cores.
- Ticket-first DevOS alignment reduces unsigned engineering drift.

### Negative

- New contracts (Deliverable 6 NEW/EXTEND rows) and orchestrator implementation remain **post–P1-B1** work.
- Parallel **SEF-X** and **SCENARIO-READINESS-X** require explicit roadmap coupling (Option 3) — schedule discipline on operators.

## Compliance

- Tier boundaries: `intergrax/` must not import `agents/`/`applications/`; engineering orchestrator lives Tier-2.
- Policy/security: meaningful external side effects remain PolicyEngine-governed on scenario path; engineering path DENY production tool profiles without PROD-Q.
- ADR status remains **`Proposed`** until separate acceptance review (OD-SEF-03); **do not** promote to **`Accepted`** in P1-B1.

## Implementation notes

- **Not in scope @ Proposed:** CAP-049..052 product code, REF-A/B/C execution, GATE-SEF-FULL runs.
- **Revision triggers:** new execution authority; Class C frozen touch; provider lock-in; failed substitution tests.
- **Verification (when implemented):** contract tests per CON-* matrix; `check_harness_adr.py` after index update.
