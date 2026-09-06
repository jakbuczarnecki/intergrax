# Diagnostic Gap Ledger

> **HARDEN qualification status (2026-08-30):** Diagnostic hardening **closed**. Matrix M1–M24: **22 PROVEN**, **2 NOT_APPLICABLE** (M21, M22), **0** open P0/P1/P2 qualification gaps. See [`DIAGNOSTIC_HARDENING_CLOSEOUT.md`](DIAGNOSTIC_HARDENING_CLOSEOUT.md) and [`DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md`](DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md).
>
> **This ledger** remains the **operational** evidence-driven backlog (DG-xxx entries below) for future platform improvements - distinct from the closed HARDEN qualification program.

**Scope:** platform-wide Diagnostic Engine operational gap backlog
**Owner:** Observability / DIAG maintainers
**Architecture:** [`docs/project/architecture/DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md) (canonical) · [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md)
**Plan:** [`docs/project/maintainers/plans/OBSERVABILITY.md`](../plans/OBSERVABILITY.md)

This ledger records **proven** diagnostic gaps and **qualification candidates** discovered during real application and proof executions. It is evidence-driven backlog for the central Diagnostic Engine - not an application-specific wish list.

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

- **PROVEN GAP** - confirmed by real execution evidence
- **QUALIFICATION CANDIDATE** - requires revalidation after a fix or fresh run
- **IDEA** - deferred design exploration only

Required fields per entry: ID, discovered by, failure scenario, terminal symptom, what engine could prove, last proven boundary, first failed/unknown boundary, root cause automatically proven (YES/NO), missing canonical evidence, missing diagnostic capability, manual work required, universal platform improvement, why not application-specific, priority (P0/P1/P2), status (OPEN / FIXED / REVALIDATED / DEFERRED / REQUIRES REVALIDATION / DESIGN REQUIRED / PARTIALLY ADDRESSED / DISPROVEN AS BLOCKER / QUALIFICATION CANDIDATE), related implementation, qualification result after fix.

---

## DG-001 - PRE-EXECUTION / OPERATOR STARTUP FAILURE VISIBILITY

| Field | Value |
|-------|-------|
| **ID** | DG-001 |
| **Discovered by** | LKW File Watcher proof / background worker qualification; LKW public Windows proof bootstrap qualification |
| **Failure scenario** | Critical host/application/worker or public proof bootstrap fails before canonical Task/Run creation |
| **Observed terminal symptom** | Worker container exits before TaskId/RunId; **or** official public `.bat` fails with `ModuleNotFoundError: local_workspace_application` before workload starts |
| **What Diagnostic Engine could prove** | Nothing for execution scope - no TaskId/RunId. Non-execution subject path is structurally available but not emitted by these bootstrap surfaces |
| **Last proven canonical boundary** | N/A (pre-execution) |
| **First failed/unknown boundary** | Host/application worker composition root; public proof launcher Python bootstrap |
| **Root cause automatically proven** | NO |
| **Missing canonical evidence** | Startup/bootstrap failure signal on platform spine before Task/Run exists |
| **Missing diagnostic capability** | Pre-execution host/worker/bootstrap failure representation with operator discovery |
| **Manual work that was required** | Container startup log inspection; manual `PYTHONPATH` workaround before bootstrap fix |
| **Universal platform improvement** | Represent critical host/application/worker/public-proof bootstrap failure before Task/Run/Attempt exists |
| **Why this is not application-specific** | Any queue-enabled Tier-3 host or public proof launcher can fail before execution identity is minted |
| **Priority** | P1 |
| **Status** | PARTIALLY ADDRESSED / DESIGN REQUIRED |
| **Related implementation** | HOST-DIAG-2 typed `DiagnosticSubjectRef` + `signal_subjects` on `DiagnosticOrchestrationRequest`; HOST-DIAG-3 `HostedApplicationDiagnosticEventPublisher` for bounded `APPLICATION_FAILED` projection when product composition supplies tenant binding |
| **Qualification result after fix** | **Partial.** Subject model + orchestrator non-execution input exist. Missing: canonical producer at real worker/host bootstrap and at public proof launcher bootstrap; operator entrypoint does not discover bootstrap `ModuleNotFoundError`. Public Windows proof bootstrap failure is a concrete qualification fixture under this gap (not a separate DG). |

---

## DG-002 - DIAGNOSTIC SCOPE DISCOVERY

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
| **Status** | PARTIALLY ADDRESSED |
| **Related implementation** | `f2198be56` — ProblemId scope discovery (Slice 1); `causal_transport_scope` — transport reference scope discovery (Slice 2); EventId/correlation pending |
| **Qualification result after fix** | ProblemId and transport reference → execution scope proven; EventId/correlation paths remain; transport lookup bounded-scale qualification open |

---

## DG-003 - OPERATOR DIAGNOSTIC STORY PROJECTION

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
| **Related implementation** | - |
| **Qualification result after fix** | Pending |

---

## DG-004 - ASYNC TRANSPORT CAUSAL CONTINUITY IN EXECUTION DIAGNOSIS

| Field | Value |
|-------|-------|
| **ID** | DG-004 |
| **Discovered by** | LKW File Watcher proof (pre- and post DG-A fix) |
| **Failure scenario** | Queue-backed File Watcher → MessageBus → background execution path must be diagnosable as one async continuity story |
| **Observed terminal symptom (pre DG-A)** | No fresh background execution; worker assembly blocked qualification |
| **Observed terminal symptom (post DG-A)** | Fresh background execution exists; RuntimeEvents exist; `DiagnosticOrchestrator` reaches **DQ-2** but `has_transport_evidence=false` |
| **What Diagnostic Engine could prove (post-fix)** | Execution timeline reconstructable; transport causal continuity **absent** from reconstruction/result |
| **Last proven canonical boundary** | Background execution admission + runtime event history (post DG-A) |
| **First failed/unknown boundary** | Transport → execution causal continuity in diagnostic reconstruction |
| **Root cause automatically proven** | NO |
| **Missing canonical evidence** | Transport causal evidence visible to `ExecutionReconstructor` / orchestrator result for fresh watcher execution |
| **Missing diagnostic capability** | Async continuity proof across MessageBus and background execution in central diagnosis |
| **Manual work that was required** | Manual correlation across watcher logs, bus, and worker execution |
| **Universal platform improvement** | Prove async causal continuity across MessageBus and background execution |
| **Why this is not application-specific** | Any queue-backed async host path shares this continuity requirement |
| **Priority** | P1 |
| **Status** | OPEN / PROVEN GAP (partial qualification) |
| **Related implementation** | DG-A fix `24506c3c14e30984d78b7b22c5cd4c42e711d125`; APP-DIAG baseline program (see OBSERVABILITY plan § Phase APP-DIAG) |
| **Qualification result after fix** | Worker assembly defect cleared (DG-A **REVALIDATED**). Fresh run proves execution + runtime history but **not** transport causal continuity. Cause layer (producer vs persistence vs identity mapping vs lookup vs reconstructor consumption) **NOT YET PROVEN**. Current LKW state: **DQ-2**, not DQ-3. |

---

## DG-005 - RUNTIME EVENT PERSISTENCE TOPOLOGY QUALIFICATION

| Field | Value |
|-------|-------|
| **ID** | DG-005 |
| **Discovered by** | LKW File Watcher proof observation |
| **Failure scenario** | HTTP host and worker appeared to use different runtime event physical topology |
| **Observed terminal symptom** | Diagnostic reconstruction may not see unified event history |
| **What Diagnostic Engine could prove (post DG-A fresh worker run)** | `runtime_history_completeness=complete`, `has_runtime_events=true` for fresh background execution |
| **Last proven canonical boundary** | Fresh worker execution runtime history complete for diagnosed scope |
| **First failed/unknown boundary** | Cross-topology event visibility for split HTTP-host/worker diagnostics |
| **Root cause automatically proven** | NO |
| **Missing canonical evidence** | Proof that split topology prevents diagnosis across host boundaries |
| **Missing diagnostic capability** | Cross-topology runtime history completeness qualification |
| **Manual work that was required** | Manual store topology comparison |
| **Universal platform improvement** | Ensure diagnostic reconstruction sees complete runtime history across host/worker topologies |
| **Why this is not application-specific** | Split host/worker topologies are common across Tier-3 applications |
| **Priority** | P2 |
| **Status** | QUALIFICATION CANDIDATE |
| **Related implementation** | - |
| **Qualification result after fix** | Suspected topology mismatch **disproven as blocker** for fresh worker execution diagnosis. Remains a qualification candidate for HTTP-host/worker cross-process cases only. |

---

## DG-A - BACKGROUND KAFKA WORKER CAUSAL DEPENDENCY WIRING DEFECT

| Field | Value |
|-------|-------|
| **ID** | DG-A (application/platform composition defect, not Diagnostic Engine feature gap) |
| **Discovered by** | LKW File Watcher proof |
| **Failure scenario** | `background_worker_factory.py` bypassed `resolve_host_queue_execution_dependencies` |
| **Observed terminal symptom** | `TypeError: create_kafka_worker() missing 1 required keyword-only argument: 'causal_evidence_persistence'` |
| **What Diagnostic Engine could prove** | N/A - pre-execution composition failure (see DG-001) |
| **Last proven canonical boundary** | Worker composition root |
| **First failed/unknown boundary** | `create_kafka_worker` assembly |
| **Root cause automatically proven** | YES (static/type surface) |
| **Missing canonical evidence** | N/A |
| **Missing diagnostic capability** | N/A |
| **Manual work that was required** | Container error log inspection |
| **Universal platform improvement** | All queue-enabled hosts must consume `HostQueueExecutionDependencies` |
| **Why this is not application-specific** | Canonical queue-worker contract applies to all Tier-3 queue hosts |
| **Priority** | P0 |
| **Status** | REVALIDATED |
| **Related implementation** | `24506c3c14e30984d78b7b22c5cd4c42e711d125` - `fix(lkw): use canonical queue execution dependencies` |
| **Qualification result after fix** | Old TypeError gone. Worker starts and remains alive. Kafka task consumed. Fresh runtime Task/Run created. RuntimeEvents exist. DiagnosticOrchestrator reaches DQ-2 on fresh execution. |

---

## Platform program - APP-DIAG / SCAFFOLD-DIAG baseline (registered)

**Canonical roadmap:** [`docs/project/maintainers/plans/OBSERVABILITY.md`](../plans/OBSERVABILITY.md) § Phase APP-DIAG

**Proven architectural gap:** APPLICATION DIAGNOSTIC BASELINE NOT GUARANTEED - a Tier-3 application composition root (LKW worker factory before DG-A; public proof launcher before bootstrap fix) can bypass mandatory observability/diagnostics/queue spine wiring. Scaffold generates observability extension templates but does **not** enforce universal diagnostic baseline or conformance gate.

**Target invariant (documentation only):** no Intergrax application is production-valid unless executions are canonically diagnosable; every scaffold-generated application is diagnostics-ready before domain logic is added.

**Implementation:** not in scope of LKW bootstrap / ledger reconciliation task - roadmap slices only.

---

## DG-006

**NOT CREATED.** Transport causal evidence absence in execution diagnosis is the post-fix proven state of **DG-004** - one canonical entry per capability gap.
