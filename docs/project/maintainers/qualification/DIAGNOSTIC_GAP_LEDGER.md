# Diagnostic Gap Ledger

**Scope:** platform-wide Diagnostic Engine qualification backlog  
**Owner:** Observability / DIAG maintainers  
**Architecture:** [`docs/project/architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md)  
**Plan:** [`docs/project/maintainers/plans/OBSERVABILITY.md`](../plans/OBSERVABILITY.md)

This ledger records **proven** diagnostic gaps and **qualification candidates** discovered during real application and proof executions. It is evidence-driven backlog for the central Diagnostic Engine — not an application-specific wish list.

## Qualification taxonomy (documentation only)

| Level | Meaning |
|-------|---------|
| **DQ-0** | No canonical diagnostic visibility |
| **DQ-1** | Execution identity visible |
| **DQ-2** | Timeline reconstructable |
| **DQ-3** | Causal boundary localized |
| **DQ-4** | Primary failure proven |
| **DQ-5** | Full operator-ready diagnostic story |

## Entry contract

Each entry MUST distinguish:

- **PROVEN GAP** — confirmed by real execution evidence
- **QUALIFICATION CANDIDATE** — requires revalidation after a fix or fresh run
- **IDEA** — deferred design exploration only

Required fields per entry: ID, discovered by, failure scenario, terminal symptom, what engine could prove, last proven boundary, first failed/unknown boundary, root cause automatically proven (YES/NO), missing canonical evidence, missing diagnostic capability, manual work required, universal platform improvement, why not application-specific, priority (P0/P1/P2), status (OPEN / FIXED / REVALIDATED / DEFERRED / REQUIRES REVALIDATION / DESIGN REQUIRED), related implementation, qualification result after fix.

---

## DG-001 — PRE-EXECUTION WORKER STARTUP FAILURE VISIBILITY

| Field | Value |
|-------|-------|
| **ID** | DG-001 |
| **Discovered by** | LKW File Watcher proof / background worker qualification |
| **Failure scenario** | LKW background worker fails during composition before canonical Task/Run creation |
| **Observed terminal symptom** | Worker container exits; no TaskId/RunId materialized |
| **What Diagnostic Engine could prove** | Nothing — fresh execution scope unavailable |
| **Last proven canonical boundary** | N/A (pre-execution) |
| **First failed/unknown boundary** | Host/application worker composition root |
| **Root cause automatically proven** | NO |
| **Missing canonical evidence** | Startup failure signal on platform spine before Task/Run exists |
| **Missing diagnostic capability** | Pre-execution host/worker startup failure representation |
| **Manual work that was required** | Container startup log inspection |
| **Universal platform improvement** | Represent critical host/application/worker startup failure before Task/Run/Attempt exists |
| **Why this is not application-specific** | Any queue-enabled Tier-3 host can fail before execution identity is minted |
| **Priority** | P1 |
| **Status** | OPEN / DESIGN REQUIRED |
| **Related implementation** | — |
| **Qualification result after fix** | Pending |

---

## DG-002 — DIAGNOSTIC SCOPE DISCOVERY

| Field | Value |
|-------|-------|
| **ID** | DG-002 |
| **Discovered by** | LKW File Watcher proof / operator diagnostic qualification |
| **Failure scenario** | Operator has transport correlation or problem signal but no canonical TaskId/RunId |
| **Observed terminal symptom** | DIAG-7 request requires explicit TaskId + RunId |
| **What Diagnostic Engine could prove** | Partial when IDs supplied; none when only transport/correlation refs exist |
| **Last proven canonical boundary** | Transport or problem reference (when present) |
| **First failed/unknown boundary** | Scope discovery from correlation references |
| **Root cause automatically proven** | NO |
| **Missing canonical evidence** | Canonical mapping from transport/problem/execution/correlation refs to TaskId/RunId |
| **Missing diagnostic capability** | Scope discovery without explicit execution IDs |
| **Manual work that was required** | Manual ID lookup across logs/stores |
| **Universal platform improvement** | Canonical discovery from transport/problem/execution/correlation references |
| **Why this is not application-specific** | Any async or multi-host workload may surface correlation without explicit scope |
| **Priority** | P1 |
| **Status** | OPEN |
| **Related implementation** | — |
| **Qualification result after fix** | Pending |

---

## DG-003 — OPERATOR DIAGNOSTIC STORY PROJECTION

| Field | Value |
|-------|-------|
| **ID** | DG-003 |
| **Discovered by** | LKW diagnostic qualification protocol |
| **Failure scenario** | DiagnosticOrchestrator runs ExecutionReconstruction and LifecycleAnalysis internally but bounded public result lacks operator-friendly timeline |
| **Observed terminal symptom** | Operator must interpret internal reconstruction artifacts manually |
| **What Diagnostic Engine could prove** | Internal reconstruction when scope known |
| **Last proven canonical boundary** | Varies by case |
| **First failed/unknown boundary** | Safe public operator projection |
| **Root cause automatically proven** | NO |
| **Missing canonical evidence** | Typed safe operator projection (no raw sensitive payloads) |
| **Missing diagnostic capability** | Last-good / first-failed / evidence story in bounded public API |
| **Manual work that was required** | Manual timeline assembly from partial engine output |
| **Universal platform improvement** | Safe typed operator diagnostic projection on orchestrator result |
| **Why this is not application-specific** | All hosted applications need bounded operator-readable failure stories |
| **Priority** | P2 |
| **Status** | OPEN / architecture review |
| **Related implementation** | — |
| **Qualification result after fix** | Pending |

---

## DG-004 — ASYNC CAUSAL CONTINUITY QUALIFICATION

| Field | Value |
|-------|-------|
| **ID** | DG-004 |
| **Discovered by** | LKW File Watcher proof (pre DG-A fix) |
| **Failure scenario** | File Watcher → MessageBus → background execution relationship could not be proven while worker assembly was broken |
| **Observed terminal symptom** | No fresh background execution; transport causal chain incomplete |
| **What Diagnostic Engine could prove** | Not yet qualified — worker assembly blocked execution |
| **Last proven canonical boundary** | File Watcher enqueue (when observed) |
| **First failed/unknown boundary** | Background worker admission |
| **Root cause automatically proven** | NO |
| **Missing canonical evidence** | Fresh transport causal evidence across watcher → bus → worker |
| **Missing diagnostic capability** | Async continuity proof across host boundaries |
| **Manual work that was required** | Manual correlation across watcher logs and worker absence |
| **Universal platform improvement** | Prove async causal continuity across MessageBus and background execution |
| **Why this is not application-specific** | Any queue-backed async host path shares this continuity requirement |
| **Priority** | P1 |
| **Status** | REQUIRES REVALIDATION |
| **Related implementation** | `fix(lkw): use canonical queue execution dependencies` |
| **Qualification result after fix** | Pending post-fix File Watcher run |

---

## DG-005 — RUNTIME EVENT PERSISTENCE TOPOLOGY QUALIFICATION

| Field | Value |
|-------|-------|
| **ID** | DG-005 |
| **Discovered by** | LKW File Watcher proof observation |
| **Failure scenario** | HTTP host and worker appeared to use different runtime event physical topology |
| **Observed terminal symptom** | Diagnostic reconstruction may not see unified event history |
| **What Diagnostic Engine could prove** | Unknown until fresh execution proves impact |
| **Last proven canonical boundary** | Not proven as defect |
| **First failed/unknown boundary** | Cross-topology event visibility |
| **Root cause automatically proven** | NO |
| **Missing canonical evidence** | Proof that split topology prevents diagnosis |
| **Missing diagnostic capability** | Cross-topology runtime history completeness qualification |
| **Manual work that was required** | Manual store topology comparison |
| **Universal platform improvement** | Ensure diagnostic reconstruction sees complete runtime history |
| **Why this is not application-specific** | Split host/worker topologies are common across Tier-3 applications |
| **Priority** | P2 |
| **Status** | REQUIRES REVALIDATION |
| **Related implementation** | — |
| **Qualification result after fix** | Pending post-fix File Watcher run |

---

## DG-A — BACKGROUND KAFKA WORKER CAUSAL DEPENDENCY WIRING DEFECT

| Field | Value |
|-------|-------|
| **ID** | DG-A (implementation gap, not diagnostic engine gap) |
| **Discovered by** | LKW File Watcher proof |
| **Failure scenario** | `background_worker_factory.py` bypassed `resolve_host_queue_execution_dependencies` |
| **Observed terminal symptom** | `TypeError: create_kafka_worker() missing 1 required keyword-only argument: 'causal_evidence_persistence'` |
| **What Diagnostic Engine could prove** | N/A — pre-execution composition failure (see DG-001) |
| **Last proven canonical boundary** | Worker composition root |
| **First failed/unknown boundary** | `create_kafka_worker` assembly |
| **Root cause automatically proven** | YES (static/type surface) |
| **Missing canonical evidence** | N/A |
| **Missing diagnostic capability** | N/A |
| **Manual work that was required** | Container error log inspection |
| **Universal platform improvement** | All queue-enabled hosts must consume `HostQueueExecutionDependencies` |
| **Why this is not application-specific** | Canonical queue-worker contract applies to all Tier-3 queue hosts |
| **Priority** | P0 |
| **Status** | FIXED (pending revalidation) |
| **Related implementation** | `applications/local_workspace_application/host/background_worker_factory.py` |
| **Qualification result after fix** | Pending post-fix File Watcher run |
