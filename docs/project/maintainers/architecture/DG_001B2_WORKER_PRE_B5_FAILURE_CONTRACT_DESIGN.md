# DG-001B2 — Worker pre-B5 failure contract design

| Field | Value |
|-------|-------|
| **Task** | `DG-001B2-WORKER-PRE-B5-FAILURE-CONTRACT-DESIGN` |
| **Mode** | architecture / contract design only — **no implementation** |
| **Branch** | `development` |
| **Start HEAD** | `aa3f509b5041b530fa48ec1d006179eb1049a458` |
| **Ancestor check** | `git merge-base --is-ancestor 028df0857e8fab2ff9c98612dba4755e8c2383ff HEAD` → exit 0 (DG-001B1 base) |
| **Design date** | 2026-09-08 |
| **Predecessors** | [`DG_001B0_BOOTSTRAP_FAILURE_SURFACE_MAPPING.md`](../qualification/DG_001B0_BOOTSTRAP_FAILURE_SURFACE_MAPPING.md) · [`DG_001B1_WORKER_PRE_B5_FAILURE_ARCHITECTURE_AUDIT_R1.md`](../qualification/DG_001B1_WORKER_PRE_B5_FAILURE_ARCHITECTURE_AUDIT_R1.md) |

---

## 1. Verdict

```text
DG-001B2 WORKER PRE-B5 FAILURE CONTRACT DESIGN = PASS (design frozen for DG-001B3)
```

**Purpose:** define an enterprise-grade, typed, modular contract for representing hosted worker bootstrap failures **before Central Diagnostics readiness (B5)** — without creating fake Problems, without bypassing `ProblemLifecycleEngine`, and without modifying Diagnostics core.

**Production changes:** none (this document only).

---

## 2. Problem statement

### 2.1 What we are solving

Hosted background workers (LKW reference path) can fail at stages **W0–W5** while Central Diagnostics prerequisites are incomplete. Today those failures surface only as stderr, structured logs, or process exit codes. Operators and platform tooling cannot correlate them with `list_problems`, tenant scope, or durable diagnostic identity.

DG-001B0 mapped these surfaces as **Category B**. DG-001B1 froze that W7–W8 (post-B5 guarded bootstrap) remain **qualified** and that W0–W5 remain the open blind spot. DG-001B2 answers:

> How should the platform represent a worker bootstrap failure when it does not yet have the full context required to create a normal `DiagnosticProblem`?

### 2.2 Constraints (non-negotiable)

| Constraint | Source |
|------------|--------|
| No fake `tenant_id`, `instance_id`, or orchestrator state | DG-001B1 §6, §8.1 |
| No bypass of `ProblemLifecycleEngine` or HOST-DIAG-3 ordering | DG-001B architecture §11 |
| No second diagnostics pipeline | DG-001B0 §6 Category C |
| No Diagnostics core changes in this slice | DG-001B1 §9.1 |
| No new `HostedApplicationEventType` values | DG-001B1 §9.1 |
| No `dict[str, Any]` / reflection / dynamic contracts | Task charter §5, §7 |
| Modular, reusable across bootstrap surfaces (worker, future hosted slices) | Task charter §5 |

### 2.3 Success criteria

A future implementation slice (DG-001B3 producer) can:

1. Emit a **typed, bounded failure fact** at any W-stage without lying about identity.
2. Route that fact through **pluggable reporters** (observability-only, structured log, deferred promotion).
3. **Promote** the same fact into canonical `APPLICATION_FAILED` → Problem projection **only when B5 is satisfied**.
4. Reuse existing `HostedProcessBootstrapPhase`, `HostedProcessBootstrapFailureFacts`, and HOST-DIAG-3 adapters at the promotion boundary.

---

## 3. Current state (post DG-001B1)

### 3.1 B0–B7 readiness ladder (LKW worker, code-evidenced)

```text
B0  logging.basicConfig                                    [W0]
B2  message-bus env gate                                   [W1]
B2  settings / environment profile                         [W2]
B2  activate_local_workspace_reference_production_authority [W3]
B2  resolve_lkw_runtime_document_store()                   [W4]
B3–B5  build_local_workspace_worker_bootstrap_diagnostics() [W5 success]
B1  HostedProcessBootstrapContext.create()                  [W6 — instance_id minted]
B6  run_guarded_hosted_process_bootstrap(WORKER_CONSTRUCTION) [W7 — qualified]
B7  run_guarded_hosted_process_bootstrap(STARTUP)          [W8 — qualified]
```

**Critical ordering:** `instance_id` mint (B1) and guarded bootstrap (B6/B7) occur **after** B3–B5 wiring. Failures at W1–W5 precede both B1 and a complete B5 stack in the current product entrypoint.

### 3.2 Existing contracts (reuse inventory)

| Contract | Location | Role today |
|----------|----------|------------|
| `HostedProcessBootstrapContext` | `intergrax/hosting/process_bootstrap.py` | B1 identity mint (`application_id`, `instance_id`, `process_role`) |
| `HostedProcessBootstrapPhase` | same | Bounded bootstrap phase taxonomy |
| `HostedProcessBootstrapFailureFacts` | same | Bounded payload facts for `APPLICATION_FAILED` |
| `run_guarded_hosted_process_bootstrap` | same | Sync guard; emits `APPLICATION_FAILED` on exception |
| `HostedApplicationEvent` / `APPLICATION_FAILED` | `intergrax/hosting/contracts/events.py` | Canonical hosting failure event (**requires** `instance_id`) |
| `HostedDiagnosticTenantBinding` | `intergrax/applications/_shared/hosted_application_diagnostic_wiring.py` | Product-owned diagnostic tenant scope |
| `HostedApplicationDiagnosticEventPublisher` | same | Observability-first; HOST-DIAG-3 Problem projection |
| `hosted_application_failure_to_problem_signal` | `hosted_application_failure_projection.py` | Deterministic event → signal adapter |

### 3.3 Gap

There is **no platform primitive** for pre-B5 failure that:

- accepts **partial identity** honestly,
- remains **typed and immutable**,
- supports **plugin reporters** without Diagnostics core coupling,
- defines an explicit **promotion** path to canonical events when B5 becomes available.

`run_guarded_hosted_process_bootstrap` is **conditional**: it requires a caller-supplied `HostedApplicationEventPublisher` and a complete `HostedProcessBootstrapContext` (including minted `instance_id`). It does **not** create `DiagnosticProblem` itself — projection is the composed publisher's responsibility.

---

## 4. Proposed contract

### 4.1 Decision A — What is the signal?

| Candidate | Verdict | Rationale |
|-----------|---------|-----------|
| **`DiagnosticProblem` / `Problem` directly** | **REJECT** | Requires B5 orchestrator, durable persistence, full `DiagnosticSignalSubjectScope` (`tenant_id`, `application_id`, `instance_id`). Creating Problems without B5 bypasses `ProblemLifecycleEngine` governance and fabricates diagnostic authority. |
| **`HostedApplicationEvent` / `APPLICATION_FAILED` alone** | **REJECT as pre-B5 primary** | `HostedApplicationEvent` schema **requires** `instance_id` (`validate_instance_id`). Emitting before B1 forces fabrication or weakens the canonical event contract used by HOST-DIAG-3. |
| **Observability export only (B4 without B5)** | **OPTIONAL degradation channel** | Valid for operator visibility; does **not** satisfy central `list_problems`. Documented in DG-001B1 option E. Not sufficient as the only contract. |
| **`HostedBootstrapFailureRecord` (new typed fact)** | **ACCEPT — primary pre-B5 signal** | Immutable, bounded, identity-honest fact capturing failure at a bootstrap readiness level. Not a Problem. Not a canonical hosting event until promoted. |

**Frozen answer:**

```text
Pre-B5 primary signal  = HostedBootstrapFailureRecord  (typed bootstrap failure fact)
Post-B5 canonical signal = HostedApplicationEvent APPLICATION_FAILED
Diagnostic outcome     = DiagnosticProblem (via HOST-DIAG-3 → ProblemLifecycleEngine only after B5)
```

Pre-B5 work records **facts**. Post-B5 work emits **events** that may become **Problems**. The two are linked by an explicit, typed **promotion** step — never by silently treating a fact as a Problem.

### 4.2 Core types (design — not implemented)

All types below are **contract targets** for DG-001B3. Names are illustrative; placement is `intergrax/hosting/` (Tier-1 hosting), not Diagnostics core.

#### 4.2.1 `BootstrapReadinessLevel`

Ordered enum mirroring the frozen B0–B7 ladder. Each failure record states **the highest readiness level not yet achieved** at detection time.

| Value | Meaning | Typical worker stage |
|-------|---------|-------------------|
| `B0_PROCESS` | Interpreter / import only | W0 |
| `B2_CONFIGURATION` | Settings / profile / authority / store resolution | W1–W4 |
| `B3_TENANT_BINDING` | Diagnostic tenant binding constructible | W5 (partial) |
| `B4_OBSERVABILITY` | Observability publisher constructible | rare partial W5 |
| `B5_DIAGNOSTICS` | Orchestrator + composed publisher ready | W6+ threshold |
| `B1_INSTANCE` | `instance_id` minted (orthogonal milestone; recorded separately) | W6 |

`B1_INSTANCE` is tracked via optional `InstanceIdentityRef` on the record, not by collapsing B1 into B5.

#### 4.2.2 `HostedBootstrapFailureStage`

Reuses **`HostedProcessBootstrapPhase`** values where applicable (`configuration`, `composition`, `dependency_resolution`, …). For pre-configuration failures (W0–W1), use a hosting-level stage:

| Stage | Maps to | When |
|-------|---------|------|
| `process_entry` | *(hosting extension)* | W0 import / interpreter |
| `environment_gate` | `configuration` | W1 env gate (non-exception exit) |
| `configuration` | existing enum | W2 |
| `composition` | existing enum | W3 |
| `dependency_resolution` | existing enum | W4, W5 diagnostics build |

**Rule:** extend `HostedProcessBootstrapPhase` with `PROCESS_ENTRY` and `ENVIRONMENT_GATE` only if needed; **one taxonomy** — no duplicate phase enums (DG-001D precedent).

#### 4.2.3 `BootstrapIdentitySnapshot`

Immutable snapshot of **what is honestly known** at failure time. Fields are optional except where noted.

```text
BootstrapIdentitySnapshot
├── application_id: str | None          # required from W2+ when manifest/settings resolved
├── process_role: str | None            # required when hosted process role is defined
├── instance_id: str | None             # ONLY after B1 mint — never synthesized
├── diagnostic_tenant_id: str | None    # ONLY after HostedDiagnosticTenantBinding constructed
├── process_os_id: int | None           # optional OS pid — not used for Problem grouping
└── bootstrap_attempt_id: str           # required — minted once per startup attempt (bounded uuid)
```

**`bootstrap_attempt_id`:** correlates deferred promotion, observability spans, and eventual `APPLICATION_FAILED.event_id` lineage without pretending to be `instance_id`.

#### 4.2.4 `HostedBootstrapFailureRecord` (primary signal)

```text
HostedBootstrapFailureRecord
├── schema_id / schema_version          # versioned contract envelope
├── record_id: str                      # unique per failure detection
├── bootstrap_attempt_id: str           # correlates one startup try
├── detected_at: datetime               # UTC, captured once
├── readiness_at_failure: BootstrapReadinessLevel
├── stage: HostedBootstrapFailureStage  # HostedProcessBootstrapPhase + extensions
├── identity: BootstrapIdentitySnapshot
├── failure_facts: HostedProcessBootstrapFailureFacts  # reuse existing bounded facts
│     ├── phase
│     ├── reason_code
│     ├── exception_type
│     └── process_role
├── severity: EventSeverity             # producer-classified, bounded
├── surface_kind: BootstrapSurfaceKind  # worker | hosted_foreground | supervisor | ...
└── promotion_state: PromotionState     # pending | promoted | observability_only | terminal
```

**Forbidden on the record:** raw exception message text, stack traces, arbitrary JSON bags, synthesized `instance_id` / `tenant_id`.

**`HostedProcessBootstrapFailureFacts` reuse:** at promotion time, the same facts map 1:1 into `hosted_process_bootstrap_failure_payload()` — no second payload shape.

#### 4.2.5 `BootstrapSurfaceKind`

```text
WORKER_BACKGROUND
HOSTED_FOREGROUND
SUPERVISOR_PRE_ENGINE
# future surfaces register via entry-point or static registry — not hard-coded in Diagnostics
```

### 4.3 Extension points (pluginable)

| Interface | Owner | Responsibility |
|-----------|-------|----------------|
| `BootstrapFailureDetector` | Platform hosting | Invokes guarded callback; on failure builds `HostedBootstrapFailureRecord` |
| `BootstrapFailureClassifier` | Product / surface plugin | Maps `Exception` → bounded `reason_code` (deterministic); default `bootstrap_unhandled_exception` |
| `BootstrapFailureReporter` | Platform hosting | Accepts record + `BootstrapReportingContext`; routes to channels |
| `BootstrapFailurePromotionPolicy` | Platform hosting | Decides if / when record may become `APPLICATION_FAILED` |
| `BootstrapFailurePromoter` | Platform hosting | Executes promotion using B5 publisher + B1 context |

**Registration model (DG-001B3):** product wires classifiers and optional extra reporters at composition time — no Diagnostics core imports, no service locator.

#### Reporter channels (ordered)

```text
1. Structured local log (always available)
2. Observability export (when B4 publisher exists)
3. Deferred promotion buffer (in-process, bounded queue — same process only)
4. Canonical APPLICATION_FAILED (only via Promoter when B5 ready)
```

Reporters **must not** call `DiagnosticOrchestrator` directly. Only the existing HOST-DIAG-3 composed publisher performs Problem projection.

### 4.4 Decision B — Identity contract

#### Required vs optional by readiness

| Field | W0 | W1 | W2–W4 | W5 fail | W6+ (B5 ok) |
|-------|:--:|:--:|:-----:|:-------:|:-----------:|
| `bootstrap_attempt_id` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `application_id` | — | partial | ✓ | ✓ | ✓ |
| `process_role` | — | — | ✓ | ✓ | ✓ |
| `diagnostic_tenant_id` | — | — | partial | partial* | ✓ |
| `instance_id` | — | — | — | — | ✓ (B1 mint) |
| `process_os_id` | optional | optional | optional | optional | optional |

\* W5 may construct `HostedDiagnosticTenantBinding` internally before orchestrator failure; record may include `diagnostic_tenant_id` **only if binding construction succeeded** before the failing step.

#### What must never be fabricated

| Artifact | Rule |
|----------|------|
| `instance_id` | Mint only via `HostedProcessBootstrapContext.create()` after product decides B1 point |
| `diagnostic_tenant_id` | Only from successful `HostedDiagnosticTenantBinding` construction |
| `DiagnosticProblem` / `ProblemId` | Only via HOST-DIAG-3 → `ProblemLifecycleEngine` after B5 |
| `tenant_id` from profile env id alone | Profile `profile_id` ≠ diagnostic tenant until bound |

#### Process identity

`process_os_id` is **operational correlation only**. It is not a substitute for `instance_id` and must not appear in `DiagnosticSignalSubjectScope`.

### 4.5 Decision C — Ownership

| Concern | Owner | Notes |
|---------|-------|-------|
| Failure detection | **Platform** (`BootstrapFailureDetector` / guarded primitive wrapper) | Single guard semantics; products compose callbacks |
| `bootstrap_attempt_id` mint | **Platform** | Once per `main()` / entrypoint invocation |
| `application_id` supply | **Product (Tier-3)** | From manifest / settings |
| `instance_id` mint | **Platform** (`HostedProcessBootstrapContext.create`) | Product chooses **when** to call create — not whether platform mints |
| `diagnostic_tenant_id` | **Product** via `HostedDiagnosticTenantBinding` | Platform validates non-empty |
| `reason_code` / `severity` classification | **Producer layer** (product classifier plugin with platform default) | Deterministic; no raw `str(exc)` |
| Record persistence (pre-B5) | **Reporter channel** | In-process buffer or observability — **not** Problem store |
| Promotion to `APPLICATION_FAILED` | **Platform `BootstrapFailurePromoter`** | Requires explicit B5 + B1 context |
| Problem projection | **HOST-DIAG-3 publisher** (unchanged) | `hosted_application_failure_to_problem_signal` |
| Problem lifecycle | **`ProblemLifecycleEngine`** (unchanged) | No alternate path |

### 4.6 Decision D — Lifecycle

```text
                    ┌─────────────────────────────────────┐
                    │  Bootstrap stage executes (W0–W5)   │
                    └─────────────────┬───────────────────┘
                                      │ failure detected
                                      v
                    ┌─────────────────────────────────────┐
                    │  Mint HostedBootstrapFailureRecord  │
                    │  (honest identity snapshot)       │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              v                       v                       v
     ┌────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
     │ LocalReporter  │    │ Observability    │    │ DeferredPromotion   │
     │ (always)       │    │ Reporter (B4+)   │    │ Buffer (in-process) │
     └────────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                                │
                                      B5 becomes available ─────┘
                                      (W5 success path OR reorder)
                                                v
                    ┌─────────────────────────────────────┐
                    │  B1: HostedProcessBootstrapContext    │
                    │      .create()                      │
                    └─────────────────┬───────────────────┘
                                      v
                    ┌─────────────────────────────────────┐
                    │  Promoter: record → APPLICATION_FAILED│
                    │  (same failure_facts payload)        │
                    └─────────────────┬───────────────────┘
                                      v
                    ┌─────────────────────────────────────┐
                    │  HOST-DIAG-3 composed publisher     │
                    │  observability → orchestrator.run   │
                    └─────────────────┬───────────────────┘
                                      v
                    ┌─────────────────────────────────────┐
                    │  ProblemLifecycleEngine.reconcile   │
                    │  → DiagnosticProblem / list_problems│
                    └─────────────────────────────────────┘

     If process exits before B5:
                    record stays observability_only | terminal
                    NO Problem created
```

#### Promotion rules (frozen)

| Condition | Promotion |
|-----------|-----------|
| B5 composed publisher available **and** B1 `instance_id` minted | **May promote** pending records from same `bootstrap_attempt_id` |
| B4 only | **Observability_only** — export record fields; no Problem |
| W0–W4 failure, process exits | **Terminal** — stderr/log truth preserved; no retroactive Problem |
| W1 env gate (`return 1`, no exception) | Record with `failure_facts.exception_type = "EnvironmentGateExit"` (bounded); still no Problem without B5 |

#### Relationship to `run_guarded_hosted_process_bootstrap`

Post-B5 guarded segments **continue to use** the existing primitive unchanged. Pre-B5 segments use a **sibling** `run_guarded_hosted_bootstrap_segment` (name TBD in DG-001B3) that:

1. Builds `HostedBootstrapFailureRecord` on failure.
2. Invokes wired reporters.
3. Re-raises or exits per product policy.
4. Does **not** emit `APPLICATION_FAILED` unless promotion preconditions are met.

This preserves DG-001B1 freeze: reuse guarded primitive for post-B5; sibling for pre-B5 — **not** LKW-local try/except.

### 4.7 W-stage mapping (reference worker)

| Stage | Readiness at failure | Primary signal | Promotion possible? |
|-------|---------------------|----------------|-------------------|
| W0 | `B0_PROCESS` | `HostedBootstrapFailureRecord` | No — terminal |
| W1 | `B2_CONFIGURATION` | Record (`environment_gate`) | No — terminal |
| W2 | `B2_CONFIGURATION` | Record | Only if product reorders to build B5 before exit* |
| W3 | `B2_CONFIGURATION` | Record | Same* |
| W4 | `B2_CONFIGURATION` | Record | **Yes** — B5 buildable after W4 success; promotion on W5 guard failure |
| W5 | `B3_TENANT_BINDING` / partial B5 | Record | **Yes** — primary DG-001B3 target |
| W6+ | `B5_DIAGNOSTICS` | `APPLICATION_FAILED` (existing) | N/A — already canonical |

\* Product reorder (mint B1 earlier, build B5 before W3) is an **optional future slice**; contract supports it but does not require it.

### 4.8 Recommended LKW implementation order (DG-001B3 input)

1. **W5 guard** — wrap `build_local_workspace_worker_bootstrap_diagnostics` after W4; on failure emit record; on success proceed to B1 + existing W7/W8 guards.
2. **W4→W5 promotion path** — when W5 fails, B5 stack was partially built; promoter emits `APPLICATION_FAILED` if composed publisher exists for failure-in-build edge cases (narrow).
3. **W3–W4 deferred promotion** — only after product accepts B5-before-W3 reorder or observability-only for W3.
4. **W1** — structured record + exit; remain terminal for Problems.

---

## 5. Alternatives considered

| Alternative | Outcome | Why rejected |
|-------------|---------|--------------|
| Create `DiagnosticProblem` directly from pre-B5 code | Rejected | Bypasses `ProblemLifecycleEngine`; requires fake identity or core API changes |
| Mint `instance_id` early (before W3) to unlock `APPLICATION_FAILED` | Deferred optional slice | Gives correlation but **still Category B** for Problems until B5; does not alone solve visibility |
| Extend `HostedApplicationEvent` to make `instance_id` optional | Rejected | Breaks HOST-DIAG-3 adapter assumptions; weakens canonical event contract |
| New `HostedApplicationEventType` for bootstrap | Rejected | DG-001B1 freeze: no new event types; `APPLICATION_FAILED` remains canonical |
| LKW-local try/except with ad-hoc logging | Rejected | DG-001B R1 freeze; not reusable |
| Observability-only (B4 without B5) as sole solution | Partial accept | Valid degradation channel, insufficient for enterprise contract |
| Persist pre-B5 records in Problem document store | Rejected | Conflates bootstrap facts with Problem lifecycle; requires persistence layer change |
| Single guard primitive forcing `APPLICATION_FAILED` always | Rejected | Cannot satisfy `instance_id` requirement pre-B1 |

---

## 6. Security / governance impact

| Risk | Mitigation |
|------|------------|
| **Tenant data leakage without ownership** | `diagnostic_tenant_id` only when `HostedDiagnosticTenantBinding` succeeded; never from raw profile env alone |
| **Cross-tenant Problem grouping** | Promotion uses existing HOST-DIAG-3 tenant binding — no new grouping path |
| **Sensitive exception text in central stores** | Record uses bounded `exception_type` + `reason_code` only — same rule as `HostedProcessBootstrapFailureFacts` |
| **Forged bootstrap facts** | Records minted only inside platform guard/detector; products supply identity inputs, not Problem IDs |
| **Replay / duplicate Problems** | Promotion sets `promotion_state=promoted`; idempotent on `bootstrap_attempt_id` + `record_id` per process |
| **Unauthorized diagnostics write** | Reporters cannot call `DiagnosticOrchestrator`; only composed publisher after B5 |

**Governance posture:** pre-B5 records are **operational facts** with explicit partial identity. They do not grant diagnostic authority until promotion through the existing spine.

---

## 7. Migration path

| Phase | Task | Scope |
|-------|------|-------|
| **DG-001B2** (this doc) | Freeze contract | Design only |
| **DG-001B3** | Producer implementation | `intergrax/hosting/` types + guarded segment wrapper; LKW W5 guard |
| **DG-001B4** (suggested) | Qualification proof | Subprocess proof: W5 failure → `list_problems` when promotion path active |
| **Optional** | B1 reorder slice | Mint `instance_id` before W3 for correlation; observability-only for W3 failures |
| **Docs** | Update prioritization / gap ledger | Close B0–B5 worker rank when W5 qualified |

**No migration of existing Problems** — W7/W8 behavior unchanged.

### 7.1 Files expected in DG-001B3 (not this task)

| Area | Illustrative paths |
|------|-------------------|
| Contract types | `intergrax/hosting/bootstrap_failure.py` (new) |
| Guard wrapper | extend `intergrax/hosting/process_bootstrap.py` |
| LKW wiring | `applications/local_workspace_application/host/background_worker_main.py` |
| Tests | `tests/unit/hosting/test_bootstrap_failure_record.py`, LKW conformance extension |

---

## 8. Open issues (document only — no implementation)

| Issue | Impact | Recommendation |
|-------|--------|----------------|
| W3 failure before document store exists | B5 not buildable | Accept terminal record or optional product reorder |
| W1 non-exception exit | No stack / exception type | Bounded `EnvironmentGateExit` classifier |
| Cross-process promotion | Deferred buffer is in-process only | Out of scope; supervisor surfaces use separate slice |
| Phase enum extension (`PROCESS_ENTRY`) | Hosting contract change | Add in DG-001B3 with single taxonomy rule |

**Diagnostics core:** no changes required.  
**New persistence layer:** not required for MVP — in-process promotion + observability sufficient.  
**Identity model:** no change to `validate_instance_id` / `HostedDiagnosticTenantBinding` — only honest optional fields on new record.

---

## 9. Test evidence

**No new runtime tests added** (design-only).

Existing tests executed (40 passed):

```text
tests/unit/applications/local_workspace_application/test_lkw_background_worker_bootstrap_conformance.py
tests/unit/hosting/test_guarded_process_bootstrap.py
```

Log: `.tmp/session/DG-001B2-WORKER-PRE-B5/pytest.log`

| Test area | Confirms design assumption |
|-----------|---------------------------|
| `test_worker_bootstrap_b6_failure_problem_visible_via_worker_read_side` | Post-B5 path: guarded primitive → Problem |
| `test_guarded_process_bootstrap.py` | `APPLICATION_FAILED` payload shape reusable at promotion |
| Absence of W1–W5 central tests | Pre-B5 Category B — contract addresses gap |

---

## 10. Confirmations

| Check | Status |
|-------|--------|
| Diagnostics core unchanged | **YES** |
| LKW unchanged | **YES** |
| Queue unchanged | **YES** |
| No bypass of Central Diagnostics | **YES** |
| No private API / reflection / dynamic contracts | **YES** |
| No branch / worktree / history rewrite | **YES** |
| No new producers or runtime classes in this task | **YES** |

---

## 11. Related documents

- [`DG_001B0_BOOTSTRAP_FAILURE_SURFACE_MAPPING.md`](../qualification/DG_001B0_BOOTSTRAP_FAILURE_SURFACE_MAPPING.md)
- [`DG_001B1_WORKER_PRE_B5_FAILURE_ARCHITECTURE_AUDIT_R1.md`](../qualification/DG_001B1_WORKER_PRE_B5_FAILURE_ARCHITECTURE_AUDIT_R1.md)
- [`DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md`](../qualification/DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md)
- [`DG_001B_FINAL_CLOSURE_R6.md`](../qualification/DG_001B_FINAL_CLOSURE_R6.md)
- [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md) — Problem lifecycle spine

---

## 12. Contract summary (quick reference)

```text
SIGNAL:     HostedBootstrapFailureRecord (pre-B5) → APPLICATION_FAILED (post-B5) → Problem
IDENTITY:   Honest optional fields; instance_id and diagnostic_tenant_id never fabricated
OWNERSHIP:  Platform detects/promotes; product supplies app/tenant; HOST-DIAG-3 projects Problems
LIFECYCLE:  Record → reporters → (optional) promotion at B5 → existing Problem spine
```
