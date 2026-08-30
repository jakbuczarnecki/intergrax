# Intergrax Central Diagnostics

**Intergrax Central Diagnostics** is the **one** canonical deterministic diagnostic engine for the platform. It interprets persisted platform facts — primarily `RuntimeEvent` execution evidence — into tenant-scoped `Problem` state, bounded operator read models, and optional investigation inputs. It does **not** mint execution identity, own observability export, or treat vendor telemetry or AI conclusions as truth.

**Persisted platform facts are truth. AI is not truth.**

| Canonical architecture | Maintainer plan | Qualification |
| ---------------------- | --------------- | ------------- |
| This document | [`docs/project/maintainers/plans/OBSERVABILITY.md`](../maintainers/plans/OBSERVABILITY.md) (DIAG slices) | [`DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md`](../maintainers/qualification/DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md) · [`DIAGNOSTIC_HARDENING_CLOSEOUT.md`](../maintainers/qualification/DIAGNOSTIC_HARDENING_CLOSEOUT.md) |

**Observability companion:** execution evidence recording, HOS, and export are documented in [`OBSERVABILITY.md`](OBSERVABILITY.md). Diagnostics **consumes** canonical evidence; observability **records** and **projects** it.

**Primary audience:** Principal / Staff engineers, harness integrators, and operators wiring diagnostic persistence, terminal triggers, or read APIs.

---

## What central diagnostics is

Central diagnostics answers:

> **What did the platform deterministically detect and persist as a recurring operational Problem?**

It is implemented under `intergrax/runtime/diagnostics/` as a **single spine**:

- `ExecutionReconstructor` — factual reconstruction from canonical evidence
- `LifecycleAnomalyAnalyzer` + `DiagnosticAssessmentBuilder` — deterministic assessment
- `ProblemGroupingEngine` — structural grouping hypotheses
- `ProblemLifecycleEngine` — stable `Problem` identity and lifecycle
- `DiagnosticOrchestrator` — canonical write/process entry point
- `DiagnosticReadService` — canonical read/reconstruction entry point

There is **no** scenario-local, proof-local, or AI-incident-specific canonical diagnostic authority.

---

## Authority model

```text
CANONICAL
  RuntimeEvent / persisted platform facts
  Problem durable state (derived, but platform-owned)

DERIVED
  ExecutionReconstruction / DiagnosticAssessment
  Observability export envelopes
  Fine-grained OTel spans (RAG / context only — see OBSERVABILITY)

NON-CANONICAL
  AI investigation conclusions
  Vendor dashboards / OTLP backends
  Operator interpretation without canonical backing
```

| Fact class | Authority | Notes |
| ---------- | --------- | ----- |
| `RuntimeEvent` | **Canonical execution evidence** | Persisted before derived observability/export where persist-first contract applies |
| `Problem` | **Derived / reconciled diagnostic state** | Durable operational pattern; rebuildable from evidence + grouping |
| Observability export | **Derived projection** | Export failure does not alter canonical truth |
| Vendor backend | **Not authority** | Missing telemetry ≠ missing platform truth |
| AI investigation output | **Non-canonical interpretation** | May consume bounded typed projections; cannot create or override canonical `Problem` truth |

**Frozen invariants:**

```text
RuntimeEvent = canonical execution evidence
Problem = derived/reconciled diagnostic state
Observability export = derived projection
Vendor backend = not authority
AI investigation output = non-canonical interpretation
```

---

## Diagnostics ≠ Observability

| System | Question |
| ------ | -------- |
| **Diagnostics** | What did the platform deterministically detect and maintain as a `Problem`? |
| **Observability** | How do we record, project, and export platform behavior for operators and vendors? |

They share infrastructure (HOS, `RuntimeEvent` persistence) but are **not** one system. Observability outage may cause **missing telemetry** but **cannot** alter platform truth or a correct business result. See [`OBSERVABILITY.md`](OBSERVABILITY.md) for export boundary, exporter health, and vendor neutrality.

---

## Business result vs diagnostic state

```text
business result != diagnostic state
```

```text
diagnostic failure must not destroy a correct business result
```

A successful governed execution remains successful even when Problem persistence fails, diagnostic post-processing throws, or observability export is degraded. Diagnostic subsystem failures may themselves be recorded as canonical runtime failure evidence — without recursive observability dependency loops.

---

## Canonical RuntimeEvent flow

Production order on the terminal execution path:

```text
runtime operation
  → RuntimeEventBus publish/record
  → RuntimeEventPersistence append          # canonical evidence first
  → terminal diagnostic trigger (when wired)
  → DiagnosticOrchestrator
  → ProblemLifecycleEngine.reconcile
  → ProblemPersistence
  → DiagnosticReadService (read path)
```

```mermaid
flowchart TB
    EX[Execution Runtime]
    BUS[RuntimeEventBus]
    REP[RuntimeEventPersistence]
    TRG[TerminalExecutionDiagnosticTrigger]
    ORC[DiagnosticOrchestrator]
    REC[ExecutionReconstructor / Analysis]
    PLC[ProblemLifecycleEngine]
    PP[ProblemPersistence]
    DS[DocumentStore abstraction]
    READ[DiagnosticReadService]

    EX --> BUS --> REP
    REP --> TRG --> ORC
    ORC --> REC --> PLC --> PP --> DS
    PP --> READ
    REP --> READ
```

**Persist-first:** canonical `RuntimeEvent` evidence is persisted before derived observability export on paths where that contract applies. Vendor export is **not** part of execution truth.

---

## How a Problem is created

1. Terminal execution (or explicit orchestration) supplies bounded scope: `tenant_id` + `TaskId` + `RunId` (execution subjects) and/or `application_id` + `instance_id` (application-instance subjects).
2. `ExecutionReconstructor` rebuilds factual execution evidence from `RuntimeEventPersistence` (+ optional causal evidence).
3. `LifecycleAnomalyAnalyzer` and `DiagnosticAssessmentBuilder` produce deterministic findings — or **no findings** on clean paths (M1).
4. `ProblemGroupingEngine` proposes grouping candidates (deterministic strategy in production).
5. `ProblemLifecycleEngine.reconcile` mints or updates a stable `ProblemId` from reconciliation key + subject refs.

`Problem` represents a **logical recurring issue**, not a single raw event.

---

## Problem identity

| Concept | Contract |
| ------- | -------- |
| **Deterministic signature** | `DeterministicProblemSignature` + strategy id/version |
| **Reconciliation key** | `tenant_id` + strategy metadata + signature — finds same logical Problem |
| **Stable `problem_id`** | Opaque minted id — same Problem across occurrences |
| **Occurrences** | Distinct accepted execution (or signal) attachments |
| **`first_seen_at` / `last_seen_at`** | Min/max of accepted occurrence `observed_at` |
| **Tenant scope** | Problem identity is **tenant-scoped** — same logical issue in tenant A and B are separate Problems |

Cross-tenant direct `problem_id` read **must not** reveal another tenant's Problem.

---

## Lifecycle

```text
OPEN
 │
 │ explicit resolve (ProblemLifecycleEngine.resolve)
 ▼
RESOLVED
 │
 │ recurrence (same reconciliation key / subject)
 ▼
OPEN (same problem_id)
```

```mermaid
stateDiagram-v2
    [*] --> OPEN: first occurrence
    OPEN --> RESOLVED: explicit resolve
    RESOLVED --> OPEN: recurrence
```

- Same `problem_id` on recurrence
- Occurrence history preserved; `first_seen_at` preserved
- New occurrence appended; `occurrence_count` increments
- **No** auto-resolve when a pattern is absent from a later batch

---

## Problem Store architecture

Mongo is a **provider**, not part of the central diagnostics contract:

```text
IntegrationProfile
  → DocumentStore abstraction
  → ConditionalDocumentStore capability
  → DocumentStoreProblemPersistence
  → ProblemLifecycleEngine
```

Do **not** document `Diagnostics → MongoDB` as production architecture. The engine is **vendor-neutral** — no Mongo-specific API, OTel API, or Datadog dependency in central semantics.

---

## Diagnostic read / reconstruction

`DiagnosticReadService` is **not** a separate source of truth. It composes reads over `ProblemPersistence`, `RuntimeEventPersistence`, and causal evidence where applicable.

```text
Problem
  → occurrence
  → SubjectRef
  → ExecutionReconstructor
  → RuntimeEvents
  → DiagnosticAssessment (derived at read time)
```

```mermaid
flowchart TB
    P[Problem record]
    O[ProblemOccurrence]
    S[SubjectRef]
    ER[ExecutionReconstructor]
    RE[RuntimeEvents]
    A[DiagnosticAssessment]

    P --> O --> S --> ER --> RE --> A
```

**Missing evidence branch:**

```text
Problem exists
+ RuntimeEvent evidence unavailable
→ Problem remains
→ occurrence reconstruction = UNAVAILABLE
→ reason = EXECUTION_EVIDENCE_UNAVAILABLE
→ assessment = absent

missing evidence ≠ fabricated diagnosis
```

---

## Application-instance subjects

`DiagnosticSubjectKind` supports:

| Kind | Scope | Reconstruction |
| ---- | ----- | -------------- |
| `EXECUTION` | `tenant_id` + `TaskId` + `RunId` | Full execution-style reconstruction when evidence exists |
| `APPLICATION_INSTANCE` | `tenant_id` + `application_id` + `instance_id` | Bounded signal assessment — **not** full execution-style reconstruction unless evidence contract provides it |

HOST-DIAG-3 may project hosting failures into central non-execution diagnostics when product composition supplies tenant binding. Hosting lifecycle does **not** synthesize `TaskId`/`RunId`.

---

## M22 — no typed unsupported scope outcome

Central diagnostics does **not** expose a typed valid-but-unsupported scope result.

| Input | Outcome |
| ----- | ------- |
| Supported scope, no violations | `has_findings=False` — clean semantics (M1) |
| Supported scope, violations | Findings → Problem reconciliation |
| Invalid orchestration input | Rejected before analysis (e.g. empty request) |

There is **no** separate `unsupported scope` typed outcome. Qualification status: **M22 = NOT_APPLICABLE**.

---

## M21 — AI not canonical authority

AI-assisted investigation (e.g. scenario conclusions, `InvestigationConclusion`) may interpret canonical facts but **cannot** create, override, or prove canonical `Problem` truth. Qualification status: **M21 = NOT_APPLICABLE**.

---

## Failure isolation

```text
Business result ─────────────→ SUCCESS

             │
             ▼
       diagnostics
             │
      Problem Store DOWN
             │
             ▼
 subsystem failure evidence

Business result unchanged
```

| Scenario | Behavior |
| -------- | -------- |
| Problem Store UP | Problem persists durably |
| Problem Store DOWN | Business survives; RuntimeEvents survive; Problem write fails visibly |
| Problem Store UP again (same process) | Subsequent Problem writes work |
| Outage occurrence | **Not** automatically replayed |

```text
Problem persistence failure does not create an implicit replay queue.
failed write = no automatic replay
```

Problem Store failure **cannot** change execution truth.

---

## Cross-process durability and OCC

**Qualified behavior (not unlimited deployment guarantees):**

- Durable Problem survives process restart; another process can read it (M9)
- Mongo-backed concurrent occurrence and lifecycle updates use optimistic concurrency / CAS
- Conflicts converge through bounded retry per implementation (M10/M11)

---

## Execution identity correlation

| Field | Role |
| ----- | ---- |
| `tenant_id` | Isolation boundary for Problems and reads |
| `task_id` (`TaskId`) | Work intent — authoritative from Execution Runtime |
| `run_id` (`RunId`) | One governed lifecycle — authoritative from Execution Runtime |
| `attempt_id` (`AttemptId`) | Global try of the run — on `RuntimeEvent`; do not conflate with `RunId` |
| `execution_id` (`ExecutionId`) | **TARGET** schedulable unit — not yet on all `RuntimeEvent` paths |

Do not mix `RunId`, `ExecutionId`, or `AttemptId` in diagnostic contracts.

---

## What central diagnostics does not do

- Mint `TaskId` / `RunId` / `AttemptId` / `ExecutionId`
- Use vendor telemetry as runtime truth
- Automatically replay failed Problem writes
- Fabricate diagnosis when evidence is missing
- Auto-resolve Problems when patterns disappear
- Provide a typed unsupported-scope outcome (M22 N/A)
- Treat AI conclusions as canonical Problems (M21 N/A)
- Depend on Mongo, OTel, or any specific vendor in its contract

---

## Host wiring (summary)

Product harness hosts wire through shared `document_store` capabilities:

- `wire_problem_persistence` → `DocumentStoreProblemPersistence`
- `RuntimeEventPersistence` on harness runtime
- `try_build_terminal_execution_diagnostic_trigger` after terminal `RuntimeEvent` persist
- `DiagnosticReadService` for operator/dashboard read path

Full hosting composition: [`APPLICATION_HOSTING.md`](APPLICATION_HOSTING.md).

---

## Qualification summary

Diagnostic hardening program **closed** (HARDEN-1 through HARDEN-5).

```text
M1–M24:  PROVEN=22  PARTIALLY_PROVEN=0  MISSING=0  NOT_APPLICABLE=2  DEFERRED=0
P0=P1=P2=0
```

N/A: **M21** (AI not authority), **M22** (no unsupported-scope contract).

Evidence index: [`DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md`](../maintainers/qualification/DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md).

---

## Related deep reference

Detailed DIAG-1..7 slice semantics, causal evidence, grouping, and orchestration contracts remain in [`OBSERVABILITY.md`](OBSERVABILITY.md) § DIAG subsystem. This document is the **primary entry point** for diagnostics architecture; OBSERVABILITY owns observability-specific export, HOS, and journal semantics.
