# DG-001B1 — Worker pre-B5 failure architecture audit (R1)

> **Task:** `DG-001B1-WORKER-PRE-B5-FAILURE-ARCHITECTURE-AUDIT-R1`  
> **Mode:** architecture audit / qualification only — **no implementation**  
> **Branch:** `development`  
> **Start HEAD:** `28685d77efb0cacd71628dc8854d1c50d9e032e9`  
> **Audit date:** 2026-09-08  
> **Ancestor check:** `git merge-base --is-ancestor 7858e357a93de9c9cbfbf7c16943eb200bff78cf HEAD` → exit 0 (DG-001B0 base)

---

## 1. Verdict

```text
DG-001B1 WORKER PRE-B5 FAILURE ARCHITECTURE AUDIT R1 = PASS
```

**Purpose:** freeze the architectural boundary for LKW background worker failures that occur **before Central Diagnostics readiness (B5)** — without implementation, without Diagnostics core changes, and without extending DG-001B B6/B7 qualified coverage.

**DG-001 overall:** `PARTIALLY ADDRESSED` — worker **W1–W5** remain the largest open pre-execution blind spot inside the worker process; **W7–W8** remain **CLOSED / QUALIFIED** under DG-001B R6.

**Production changes:** none.

---

## 2. Scope

### 2.1 In scope

| Area | Evidence sources |
| ---- | ---------------- |
| LKW worker entrypoint ordering | `applications/local_workspace_application/host/background_worker_main.py` |
| B0–B7 readiness ladder | [`DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md`](DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md) §8 |
| Guarded bootstrap primitive + context contracts | `intergrax/hosting/process_bootstrap.py` |
| HOST-DIAG-3 composition boundary | `intergrax/applications/_shared/hosted_application_diagnostic_wiring.py` |
| Prior surface mapping | [`DG_001B0_BOOTSTRAP_FAILURE_SURFACE_MAPPING.md`](DG_001B0_BOOTSTRAP_FAILURE_SURFACE_MAPPING.md) §4.4 |
| DG-001B qualified post-B5 slice | [`DG_001B_FINAL_CLOSURE_R6.md`](DG_001B_FINAL_CLOSURE_R6.md) |

### 2.2 Out of scope

- Diagnostics core internals (`DiagnosticOrchestrator`, Problem lifecycle engine)
- Supervisor / hosted foreground surfaces (H0–H6) — reference only
- Public launcher (P0–P2), ASGI strict path (A0–A2)
- Celery / Nexus execution worker bootstrap (Category C)
- Remediation implementation, new producers, new event types
- Queue architecture, LKW production wiring changes

---

## 3. Central Diagnostics readiness model (B5 boundary)

Central `DiagnosticProblem` projection requires **all** of:

| Prerequisite | Symbol | Contract owner |
| ------------ | ------ | -------------- |
| Application identity | `application_id` | `normalize_application_id` — caller-supplied |
| Instance identity | `instance_id` | `validate_instance_id` — platform mint at B1 |
| Tenant authority | `HostedDiagnosticTenantBinding.tenant_id` | product-owned; non-empty |
| Observability publisher | `ObservabilityHostedApplicationEventPublisher` | B4 — export-first |
| Diagnostic stack | `DiagnosticOrchestrator` + durable Problem persistence | B5 |
| Canonical failure fact | `HostedApplicationEvent` / `APPLICATION_FAILED` with bounded `phase` | HOST-DIAG-3 input |

**Readiness threshold:** **B5** = B3 + B4 + B5 satisfied. Failures before this stack is constructed cannot produce safe `DiagnosticProblem` projection.

**Current LKW ordering (code-evidenced):**

```text
B0  logging.basicConfig
B2  LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS gate          [W1]
B2  LocalWorkspaceBackendSettings.from_env()         [W2]
B2  build_local_workspace_environment_profile()       [W2]
B2  activate_local_workspace_reference_production_authority()  [W3]
B2  resolve_lkw_runtime_document_store()             [W4]
B3–B5  build_local_workspace_worker_bootstrap_diagnostics()  [W5 success path]
B1  HostedProcessBootstrapContext.create()           [W6 — instance_id minted HERE]
B6  run_guarded_hosted_process_bootstrap(WORKER_CONSTRUCTION)  [W7 — qualified]
B7  run_guarded_hosted_process_bootstrap(STARTUP)     [W8 — qualified]
```

**Critical ordering fact:** `instance_id` mint (B1) and guarded bootstrap (B6/B7) occur **after** B3–B5 wiring. Failures at W1–W5 therefore precede both B1 and B5 in the current product entrypoint.

---

## 4. Worker lifecycle map

### 4.1 Stage-by-stage map

| Stage ID | B-level | Component | Available context | Available identity | Available dependencies | Possible failure modes | Current handling | Cat. |
| -------- | ------- | --------- | ----------------- | ------------------ | -------------------- | -------------------- | ---------------- | ---- |
| **W0** | B0 | Python interpreter / module import | OS process | **NO** stable identities | **NO** hosting contracts | ImportError, interpreter crash | OS / stderr; non-zero exit | **B** |
| **W1** | B2 | Message-bus env gate (`local_workspace_message_bus_enabled`) | env vars only | manifest `app_id` implicit; **NO** `instance_id` | **NO** publisher | Gate false → early exit | `logger.error`; `return 1` (no exception) | **B** |
| **W2a** | B2 | `_resolve_settings` / `LocalWorkspaceBackendSettings.from_env()` | parsed env | `application_id` via manifest constant | **NO** tenant binding; **NO** store | ValidationError, TypeError, missing env | Uncaught exception → stderr | **B** |
| **W2b** | B2 | `build_local_workspace_environment_profile(settings)` | settings + manifest | `application_id` YES; profile `tenant_id` as env id only | **NO** `HostedDiagnosticTenantBinding` | Profile build failure | Uncaught exception → stderr | **B** |
| **W3** | B2 | `activate_local_workspace_reference_production_authority()` | settings, profile, composition root | `application_id` YES; profile id known | composition stores **may** materialize on success; **NO** B5 stack | Governance/deploy failure, registry projection failure, store init failure | Uncaught exception → stderr | **B** |
| **W4** | B2 | `resolve_lkw_runtime_document_store(settings)` | settings; registry projection from W3 | same as W3 | document store path resolvable; **NO** orchestrator/publisher | Store factory / path resolution failure | Uncaught exception → stderr | **B** |
| **W5** | B3–B5 | `build_local_workspace_worker_bootstrap_diagnostics()` | registry projection, settings, profile, document store | `application_id` YES; tenant binding minted **inside** function | orchestrator stack build; observability exporter; composed publisher on success | Runtime build failure, orchestrator wiring failure, exporter resolution failure, publisher composition failure | Uncaught exception → stderr; **aborts before B1 mint** | **B** |
| **W6** | B1 | `HostedProcessBootstrapContext.create()` | post-W5 diagnostics bundle | **YES** — `instance_id` minted | composed publisher **YES** | Invalid `application_id` / `process_role` validation | `ValueError` → stderr | **A** *at boundary* (publisher exists; no guarded callback yet) |
| **W7** | B6 | `build_local_workspace_background_worker_wiring` (guarded) | full bootstrap context + B5 publisher | **YES** | queue/KV/kafka composition | `create_kafka_worker` TypeError, dependency resolution, harness wiring | `APPLICATION_FAILED` (`worker_construction`) → HOST-DIAG-3 → `list_problems` | **A** |
| **W8** | B7 | `worker.start()` (guarded) | same as W7 | **YES** | transport runtime | Kafka connect failure, consumer startup failure | `APPLICATION_FAILED` (`startup`) → HOST-DIAG-3 | **A** |

### 4.2 Lifecycle diagram (current)

```text
main()
  │
  ├─ W0 import ──────────────────────────────── Category B
  ├─ W1 env gate ────────────────────────────── Category B (exit 1)
  ├─ W2 settings/profile ────────────────────── Category B
  ├─ W3 authority activation ────────────────── Category B (heavy; stores on success)
  ├─ W4 document store resolve ──────────────── Category B
  ├─ W5 diagnostics build (B3–B5) ───────────── Category B on failure
  │       └─ success → composed publisher exists
  ├─ W6 context mint (B1) ───────────────────── Category A boundary
  ├─ W7 guarded worker construction (B6) ────── Category A (DG-001B R6)
  └─ W8 guarded worker startup (B7) ─────────── Category A (DG-001B R6)
```

**DG-001B1 focus:** W0–W5 only. W7–W8 are **out of remediation scope** (already qualified).

---

## 5. Failure surface matrix

| Stage | Failure example | `application_id` | `instance_id` | Diagnostic `tenant_id` | B4 observability | B5 orchestrator | `APPLICATION_FAILED` | `DiagnosticProblem` | Reporting today | Cat. |
| ----- | --------------- | :--------------: | :-----------: | :--------------------: | :--------------: | :-------------: | :------------------: | :-----------------: | --------------- | ---- |
| W0 | import error | — | — | — | — | — | — | — | stderr | **B** |
| W1 | message bus disabled | partial | — | — | — | — | — | — | log + exit 1 | **B** |
| W2 | invalid settings | ✓ | — | partial | — | — | — | — | stderr | **B** |
| W3 | deploy/activate failure | ✓ | — | partial | — | — | — | — | stderr | **B** |
| W4 | document store failure | ✓ | — | partial | — | — | — | — | stderr | **B** |
| W5 | orchestrator build failure | ✓ | — | ✓* | partial** | — | — | — | stderr | **B** |
| W6 | context validation | ✓ | ✓ | ✓ | ✓ | ✓ | —*** | ✓ | stderr | **A** |
| W7 | kafka worker wiring | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | central + observability | **A** |
| W8 | worker.start() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | central + observability | **A** |

\* W5 constructs `HostedDiagnosticTenantBinding` early inside `build_local_workspace_worker_bootstrap_diagnostics`; failure later in the same function leaves binding intent but **no** complete B5 stack.  
\** Observability exporter may partially resolve before orchestrator failure; composed publisher not returned to caller on failure.  
\*** No failure event on success path; validation failure is uncaught before guard.

---

## 6. Identity matrix

Per-stage answers to required identity / channel questions.

| Stage | Application identity? | Instance identity? | Tenant identity (diagnostic)? | Diagnostic publisher? | Observability channel? | Safe central report? |
| ----- | :-------------------: | :----------------: | :-----------------------------: | :-------------------: | :--------------------: | :------------------: |
| W0 | NO | NO | NO | NO | NO | NO |
| W1 | partial (manifest only) | NO | NO | NO | NO | NO |
| W2 | YES | NO | partial (profile id, not binding) | NO | NO | NO |
| W3 | YES | NO | partial | NO | NO | NO |
| W4 | YES | NO | partial | NO | NO | NO |
| W5 (fail) | YES | NO | partial → binding attempted | NO (incomplete) | NO | NO |
| W6 | YES | YES | YES | YES (composed) | YES | YES |
| W7–W8 | YES | YES | YES | YES | YES | YES |

**Frozen identity ownership (no ambiguity — audit does not STOP):**

| Artifact | Owner |
| -------- | ----- |
| `application_id` | Tier-3 product caller (`LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id`) |
| `instance_id` mint | `HostedProcessBootstrapContext.create` in `intergrax.hosting` — invoked by product **after** B5 wiring today |
| Diagnostic `tenant_id` | Product via `HostedDiagnosticTenantBinding` (`environment_profile.profile_id` for LKW) |
| Composed publisher | `intergrax.applications._shared.hosted_application_diagnostic_wiring` |
| Bootstrap failure fact | `intergrax.hosting.run_guarded_hosted_process_bootstrap` |
| Problem projection | HOST-DIAG-3 publisher — **not** diagnostics core |

**No fabrication rule:** pre-W5 stages must **not** synthesize `instance_id`, diagnostic tenant, or orchestrator state to force central reporting.

---

## 7. Failure classification

### Category A — Safe central diagnostic reporting possible

| Surface | Condition | Mechanism | Uzasadnienie |
| ------- | --------- | --------- | ------------ |
| W6 | Post-W5 success; context validation only | Composed publisher exists; could emit `APPLICATION_FAILED` if guarded — not wired today for W6 | B3–B5 complete before B1 mint; identity + tenant + orchestrator available |
| W7 B6 | DG-001B R6 qualified | `run_guarded_hosted_process_bootstrap` + HOST-DIAG-3 | Proven in R5/R6 qualification |
| W8 B7 | DG-001B R6 qualified | Same | Same |

**DG-001B1 does not expand Category A** to W1–W5 under current architecture.

### Category B — Insufficient context for safe central reporting

| Surface | Primary gap | Operator sees today | Uzasadnienie |
| ------- | ----------- | ------------------- | ------------ |
| W0 | No Python host identity contract | stderr / exit | B0 — no hosting spine |
| W1 | No publisher; no `instance_id`; gate returns without exception | log + exit 1 | Cannot call guarded primitive meaningfully |
| W2 | No tenant binding; no B5 | stderr | Profile id ≠ diagnostic contract |
| W3 | Heavy authority before B5; no `instance_id` | stderr | Stores may exist only **after** success; failure path has no orchestrator |
| W4 | Document store unresolved → B5 not buildable | stderr | B5 depends on resolved store |
| W5 | B5 build incomplete; no `instance_id` mint yet | stderr | Partial tenant intent inside function does not yield composed publisher to caller |

**Largest remaining worker blind spot:** **W1–W5** — aligns with DG-001B0 rank-1 recommendation and prioritization doc B0–B5 rank 4.

### Category C — Outside DG-001B / DG-001B1 remediation lane

| Surface | Rationale |
| ------- | --------- |
| Celery / Nexus worker bootstrap | Execution worker runtime; not LKW hosted process bootstrap |
| `intergrax/queueing/worker_bootstrap.py`, `intergrax/runtime/task/worker_bootstrap.py` | Tier-1 task admission bootstrap; no application hosting identity contract |
| Post-admission task/run failures | Execution identity path — separate DG-001 parent scope |
| Diagnostics projection failure after observability export | Non-fatal by HOST-DIAG-3 design |
| Public launcher / ASGI / supervisor surfaces | Separate qualified slices (DG-001C, DG-001D, DG-001A) |

---

## 8. Ownership analysis

### 8.1 Existing contracts (reuse-first)

| Contract | Location | Reusable for pre-B5 remediation? |
| -------- | -------- | ---------------------------------- |
| `HostedProcessBootstrapContext` | `intergrax/hosting/process_bootstrap.py` | **YES** — identity mint; **does not** carry tenant or publishers |
| `HostedProcessBootstrapPhase` | same | **YES** — maps W2→`configuration`, W3→`composition`, W4→`dependency_resolution` |
| `run_guarded_hosted_process_bootstrap` | same | **CONDITIONAL** — requires caller-supplied `event_publisher`; central Problem only with B5 composed publisher |
| `HostedDiagnosticTenantBinding` | `hosted_application_diagnostic_wiring.py` | **YES** — after profile known (W2b+) |
| `build_hosted_application_diagnostic_event_publisher` | same | **YES** — after document store + orchestrator deps resolvable (post-W4 success path) |
| `ObservabilityHostedApplicationEventPublisher` | `intergrax/hosting/eventing.py` | **PARTIAL** — observability-only Category A for **export**, not `DiagnosticProblem` |

**Missing contracts (documented gap — do not invent in audit):**

- No platform primitive for pre-B5 failure that produces `DiagnosticProblem` without full B5 stack
- No early-bootstrap context object combining identity + partial publishers (architecture doc R1 considered but **not** shipped as separate type)
- `HostedProcessBootstrapContext` intentionally excludes `tenant_binding` and publishers — product passes publishers to guard separately

### 8.2 Dependency paradox (frozen)

```text
B5 requires durable Problem persistence → typically document store (W4)
W3 authority activation may fail before stores exist
Therefore W3 failures cannot assume B5 resolvable without reordering or lighter diagnostic bootstrap
```

**Feasible partial path (architecture only):** after **W4 success**, B5 **is** buildable — W5 failure is the last pre-B1 blind spot where tenant + store exist but composed publisher never reaches caller.

### 8.3 Guarded primitive reuse decision (frozen)

| Option | Verdict | Rationale |
| ------ | ------- | --------- |
| **A — Reuse `run_guarded_hosted_process_bootstrap` unchanged for W2–W5** | **CONDITIONAL YES** | Only **after** composed B5 publisher exists; map phases to existing enum; **NO** diagnostics core changes |
| **B — Sibling early-bootstrap primitive** | **DEFER** | Not required to close audit; would duplicate guard semantics unless B5 precondition enforced |
| **C — LKW-local try/except** | **REJECTED** | Frozen forbidden in DG-001B R1 |
| **D — Mint B1 before W3** | **OPTIONAL future slice** | Gives `instance_id` for W3–W4 failures but **still Category B** for central diagnostics until B5 exists |
| **E — Observability-only pre-B5 reporting** | **OPTIONAL** | B4-without-B5 yields export visibility, **not** `list_problems`; acceptable documented degradation |

**Audit freeze:** W1–W5 remain Category B. Any future implementation slice must preserve HOST-DIAG-3 ordering and must not modify Diagnostics core.

---

## 9. Architectural recommendation

### 9.1 Frozen outcomes

| Decision | Outcome |
| -------- | ------- |
| Pre-B5 worker coverage today | **Category B** for W0–W5 — documented, not a DG-001B regression |
| DG-001B qualified scope | **Unchanged** — W7–W8 only |
| Diagnostics core changes | **NONE required** for future pre-B5 work |
| New event types | **NO** |
| Private API / reflection | **FORBIDDEN** |
| Identity owners | **Clear** — no STOP |
| New runtime flow | **Not required** to accept audit; optional product reorder is a **separate implementation slice** |

### 9.2 Recommended remediation direction (next slice — not this task)

Priority order for closing W1–W5 blind spot:

1. **W5 hardening** — guard `build_local_workspace_worker_bootstrap_diagnostics` **after** W4 with composed publisher callback scope; phase `dependency_resolution` or `composition`. Lowest paradox surface.
2. **W3–W4 phased guards** — only after B5 can be built from successful W4 outputs; mint B1 before guarded W3 only if observability-only degradation accepted for W3 failures.
3. **W1 env gate** — remain Category B or map to structured log fact; **do not** fabricate identity.
4. **W0** — remain Category B (true B0).

**Invariant preservation:** observability before diagnostics; producer owns failure fact; no diagnostics-core imports from LKW.

### 9.3 STOP conditions evaluated

| Condition | Result |
| --------- | ------ |
| Diagnostics core change required? | **NO** |
| Unknown identity owner? | **NO** |
| New runtime flow mandatory for audit? | **NO** |
| Scope creep into B2/B3 without decision? | **NO** — B2/B3 meanings frozen via B0–B7 ladder |

---

## 10. Proposed next steps

| Step | Task sketch | Target |
| ---- | ----------- | ------ |
| 1 | `DG-001B2-WORKER-PRE-B5-FAILURE-PRODUCER-R2` (implementation) | W5 guard + optional W3–W4 after B5 resolvable |
| 2 | Real qualification subprocess proof | W5 failure → `list_problems` when B5 wired before guard |
| 3 | Prioritization doc update | Close B0–B5 rank 4 for worker slice when qualified |
| 4 | **Do not** merge with DG-001A (default hosted) or DG-001C (public launcher) | Preserve slice boundaries |

---

## 11. Test evidence

**No new runtime tests added** (audit-only).

Existing tests run (40 passed):

```text
tests/unit/applications/local_workspace_application/test_lkw_background_worker_bootstrap_conformance.py
tests/unit/hosting/test_guarded_process_bootstrap.py
```

Log: `.tmp/session/DG-001B1-WORKER-PRE-B5/pytest.log`

| Test | Confirms |
| ---- | -------- |
| `test_worker_main_publisher_factory_is_host_diag_3` | W5 path uses HOST-DIAG-3 composition |
| `test_worker_bootstrap_b6_failure_problem_visible_via_worker_read_side` | W7 Category A (post-B5) |
| `test_worker_construction_failure_emits_application_failed_with_worker_construction_phase` | Guarded primitive + phase taxonomy |
| `test_guarded_process_bootstrap.py` | G1 contract |

**Gap confirmed by test absence:** no qualification test asserts central visibility for W1–W5 failures (consistent with Category B).

---

## 12. Confirmations

| Check | Status |
| ----- | ------ |
| Diagnostics core unchanged | **YES** |
| LKW unchanged | **YES** |
| Queue unchanged | **YES** |
| No bypass of Central Diagnostics | **YES** |
| No private API / reflection / dynamic contracts | **YES** |
| No branch / worktree / history rewrite | **YES** |
| No new producers or events in this task | **YES** |

---

## 13. Related documents

- [`DG_001B0_BOOTSTRAP_FAILURE_SURFACE_MAPPING.md`](DG_001B0_BOOTSTRAP_FAILURE_SURFACE_MAPPING.md) — parent mapping @ `7858e357a93de9c9cbfbf7c16943eb200bff78cf`
- [`DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md`](DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md) — B0–B7 ladder + R1 freeze
- [`DG_001B_FINAL_CLOSURE_R6.md`](DG_001B_FINAL_CLOSURE_R6.md) — W7/W8 qualified
- [`DG_001_REMAINING_PRE_EXECUTION_SURFACES_PRIORITIZATION.md`](DG_001_REMAINING_PRE_EXECUTION_SURFACES_PRIORITIZATION.md) — B0–B5 rank 4
- [`DIAGNOSTIC_GAP_LEDGER.md`](DIAGNOSTIC_GAP_LEDGER.md) — DG-001 parent
