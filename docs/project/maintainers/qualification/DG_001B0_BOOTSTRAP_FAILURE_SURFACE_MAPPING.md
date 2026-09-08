# DG-001B0 — Bootstrap Failure Surface Mapping

> **Task:** `DG-001B0-BOOTSTRAP-FAILURE-SURFACE-MAPPING`  
> **Mode:** architecture audit / qualification only — **no implementation**  
> **Branch:** `development`  
> **Start HEAD:** `f56ad8e38ff48097e2316b3347b2512c393b987d`  
> **Audit date:** 2026-09-08  
> **Ancestor check:** `git merge-base --is-ancestor 762feff42b9b3fbdceecefbd0ae0b95de9b6fc89 HEAD` → exit 0 (DG-001C qualified base)

---

## 1. Verdict

```text
DG-001B0 BOOTSTRAP FAILURE SURFACE MAPPING = PASS
```

**Purpose:** map every in-scope bootstrap boundary where Intergrax can terminate **before Central Diagnostics is fully ready** (B5: `DiagnosticOrchestrator` + durable Problem persistence + product tenant binding + composed publisher).

**DG-001 overall:** `PARTIALLY ADDRESSED` — B0–B5 prerequisite gaps remain the primary open blind spot; qualified sub-slices (DG-001B B6/B7, DG-001D, DG-001A boundary, DG-001C boundary) are documented as closed elsewhere.

**Production changes:** none.

---

## 2. Scope

### 2.1 In scope

| Area | Evidence sources |
| ---- | ---------------- |
| Bootstrap dependency ladder B0–B7 | [`DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md`](DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md) §8 |
| Guarded process bootstrap primitive | `intergrax/hosting/process_bootstrap.py` |
| Hosted foreground composition root | `intergrax/hosting/runner.py`, `applications/local_workspace_application/hosting/foreground.py` |
| Supervisor pre-engine boundary | `intergrax/hosting/supervisor/supervisor.py`, `failure_projection.py` |
| LKW worker entrypoint | `applications/local_workspace_application/host/background_worker_main.py` |
| LKW ASGI / reference production host | `applications/local_workspace_application/host/main.py` |
| Public launcher transport path | [`DG_001C_PUBLIC_LAUNCHER_BOOTSTRAP_QUALIFICATION.md`](DG_001C_PUBLIC_LAUNCHER_BOOTSTRAP_QUALIFICATION.md) |
| HOST-DIAG-3 composition contract | `intergrax/applications/_shared/hosted_application_diagnostic_wiring.py` |

### 2.2 Out of scope

- Diagnostics core internals (`DiagnosticOrchestrator`, Problem lifecycle engine)
- LKW production wiring changes, queue architecture, runtime task execution after admission
- Full repository scan; Celery/Nexus worker runtime bootstrap (Category C reference only)
- Remediation design or new producers/events

---

## 3. Central Diagnostics readiness model

Central Diagnostics projection (`DiagnosticProblem`) requires **all** of:

| Prerequisite | Symbol | Contract |
| ------------ | ------ | -------- |
| Application identity | `application_id` | `normalize_application_id` |
| Instance identity | `instance_id` | `validate_instance_id` |
| Tenant authority | `HostedDiagnosticTenantBinding.tenant_id` | product-owned; non-empty |
| Observability publisher | `ObservabilityHostedApplicationEventPublisher` or composed wrapper | export-first |
| Diagnostic stack | `DiagnosticOrchestrator` + durable `DocumentStore` Problem persistence | sync `orchestrator.run()` |
| Canonical failure fact | `HostedApplicationEvent` / `APPLICATION_FAILED` with bounded `phase` + `reason_code` | HOST-DIAG-3 projection input |

**Readiness threshold:** **B5** — B3 + B4 + B5 satisfied. B6/B7 failures are centrally visible **only when** B5 exists before the guarded callback (DG-001B qualified).

---

## 4. Bootstrap boundary map

Ordered by typical startup chronology. **Category** column uses §6 taxonomy.

### 4.1 Public / transport layer

| ID | Component | When it occurs | Identity available | Tenant binding | Publisher | Safe `DiagnosticProblem`? | Current reporting | Cat. |
| -- | --------- | -------------- | ------------------ | -------------- | --------- | ------------------------- | ----------------- | ---- |
| **P0** | Shell wrapper (`.bat` / `.sh`) | Before Python interpreter | **NO** | **NO** | **NO** | **NO** | Shell echo, exit code | **B** |
| **P1** | `uv` / project bootstrap | PYTHON-BOOTSTRAP | **NO** stable `application_id` / `instance_id` | **NO** | **NO** | **NO** | stderr, non-zero exit | **B** |
| **P2** | Proof orchestrator (`run-lkw-*-proof.py`) | After import; before host composition | **PARTIAL** — manifest `app_id` may be known; **no** `instance_id` | **PARTIAL** — profile may load; **no** `HostedDiagnosticTenantBinding` | **NO** | **NO** | `CoreProofError` KV stdout, logs | **B** |
| **P3** | Docker / child subprocess phases in proofs | External to Python host | **NO** / varies | **NO** | **NO** | **NO** | Child exit code, proof KV | **C** |

**Evidence:** DG-001C §4–§6; `tests/unit/applications/local_workspace_application/test_public_launcher_bootstrap_diagnostic_qualification.py`.

### 4.2 LKW ASGI / backend strict production

| ID | Component | When it occurs | Identity available | Tenant binding | Publisher | Safe `DiagnosticProblem`? | Current reporting | Cat. |
| -- | --------- | -------------- | ------------------ | -------------- | --------- | ------------------------- | ----------------- | ---- |
| **A0** | `create_app()` / `StrictProductionAsgiPlaceholder` | Uvicorn import without activated composition | manifest `app_id` only | **NO** | **NO** | **NO** | `HarnessHostRegistryAuthorityError` / stderr | **B** |
| **A1** | `run_reference_production()` lifecycle | Settings parse → deploy/activate → uvicorn | `application_id` YES after manifest; **no** `instance_id` | profile `tenant_id` as env id; **no** diagnostic binding | **NO** | **NO** | Python exception / uvicorn logs | **B** |
| **A2** | `create_local_workspace_process_app()` | Post-activate ASGI build | `application_id` YES | profile id known; **no** diagnostic binding | **NO** | **NO** | FastAPI/uvicorn stderr | **B** |

**Evidence:** `applications/local_workspace_application/host/main.py`; DG-001C §4 (hosted CLI rejects bare bootstrap).

### 4.3 Hosted foreground composition root

| ID | Component | When it occurs | Identity available | Tenant binding | Publisher | Safe `DiagnosticProblem`? | Current reporting | Cat. |
| -- | --------- | -------------- | ------------------ | -------------- | --------- | ------------------------- | ----------------- | ---- |
| **H0** | `resolve_hosted_application_definition(profile)` | Before supervisor | `application_id` from profile | **NO** | **NO** | **NO** | Exception propagates | **B** |
| **H1** | `_resolve_reference_paths` / runner factories | Before supervisor loop | `application_id` YES; **no** `instance_id` | **NO** | **NO** | **NO** | `HostedApplicationConfigurationError` | **B** |
| **H2** | Event-loop guard in `run_hosted_application` | Sync call from active loop | same as H0 | **NO** | **NO** | **NO** | `HostedApplicationConfigurationError` | **B** |
| **H3** | `run_local_workspace_hosted_application` without composition | Product guard | `application_id` N/A | **NO** | **NO** | **NO** | `HarnessHostRegistryAuthorityError` | **B** |
| **H4** | Default `event_publisher_factory` (`ObservabilityHostedApplicationEventPublisher`) | Supervisor constructed | `application_id` YES; `instance_id` minted in supervisor loop | **NO** diagnostic tenant | Observability **YES** (B4) | **NO** — no B5 | `APPLICATION_FAILED` → observability only; **no** `list_problems` | **B** |
| **H5** | Supervisor pre-engine (`engine_factory` / contract validation) | After `instance_id` minted; before `engine.run_until_stopped()` | **YES** | **YES** when product passes `HostedDiagnosticTenantBinding` + orchestrator | Composed **YES** when HOST-DIAG-3 wired | **YES** when B3–B5 wired | `APPLICATION_FAILED` (`engine_construction` / `engine_contract_validation`); DG-001D qualified | **A** *if wired* / **B** *default* |
| **H6** | `HostedApplicationEngine` lifecycle (startup hooks, runtime) | After engine constructed | **YES** | same as H5 | same as H5 | **YES** when B3–B5 wired | `APPLICATION_FAILED` via engine cleanup path | **A** *if wired* / **B** *default* |
| **H7** | `python -m local_workspace_application.hosting` stub | Operator invokes bare CLI | **NO** | **NO** | **NO** | **NO** | stderr message; exit 1 | **B** |

**Evidence:** `intergrax/hosting/runner.py`, `supervisor/supervisor.py`, `applications/local_workspace_application/hosting/foreground.py`; DG-001A, DG-001D final closure.

### 4.4 LKW background worker (B0–B7 ladder)

Current `background_worker_main.main()` ordering (code-evidenced):

```text
B0  logging.basicConfig
B2  message-bus env gate (return 1, no exception)
B2  LocalWorkspaceBackendSettings.from_env()
B2  build_local_workspace_environment_profile()
B2  activate_local_workspace_reference_production_authority()   ← heavy; stores materialized
B2  resolve_lkw_runtime_document_store()
B3–B5  build_local_workspace_worker_bootstrap_diagnostics()
B1  HostedProcessBootstrapContext.create()  ← instance_id minted HERE (after B5 wiring)
B6  run_guarded_hosted_process_bootstrap(WORKER_CONSTRUCTION)
B7  run_guarded_hosted_process_bootstrap(STARTUP)
```

| ID | Level | Component | When it occurs | Identity | Tenant | Publisher | Safe `DiagnosticProblem`? | Current reporting | Cat. |
| -- | ----- | --------- | -------------- | -------- | ------ | --------- | ------------------------- | ----------------- | ---- |
| **W0** | B0 | Process start | Interpreter entry | **NO** | **NO** | **NO** | **NO** | OS / stderr | **B** |
| **W1** | B2 | `LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS` gate | Before authority | manifest `app_id` only | profile id parseable; **no** binding | **NO** | **NO** | `logger.error`; `return 1` | **B** |
| **W2** | B2 | Settings / env profile parse | `_resolve_settings`, `build_local_workspace_environment_profile` | `application_id` via manifest | profile id | **NO** | **NO** | Uncaught exception / stderr | **B** |
| **W3** | B2 | `activate_local_workspace_reference_production_authority()` | Registry projection, deploy/activate, governance | `application_id` YES | profile id; **no** diagnostic binding | **NO** | **NO** | Uncaught exception / stderr | **B** |
| **W4** | B2 | `resolve_lkw_runtime_document_store()` | Before diagnostics wiring | YES | profile id | **NO** | **NO** | Uncaught exception / stderr | **B** |
| **W5** | B3–B5 | `build_local_workspace_worker_bootstrap_diagnostics()` | Orchestrator + composed publisher construction | YES | **YES** (`HostedDiagnosticTenantBinding`) | Composed diagnostic publisher | **N/A** — success path only; failure here aborts before B1 mint | Uncaught exception | **B** |
| **W6** | B1 | `HostedProcessBootstrapContext.create()` | After W5 succeeds | **YES** (`instance_id` minted) | **YES** | **YES** | **YES** — but no guarded callback yet | N/A on success | **A** *at boundary* |
| **W7** | B6 | `build_local_workspace_background_worker_wiring` (guarded) | Inside guarded callback | **YES** | **YES** | **YES** | **YES** | `APPLICATION_FAILED` → observability → Problem (DG-001B R6) | **A** |
| **W8** | B7 | `worker.start()` (guarded) | Transport startup | **YES** | **YES** | **YES** | **YES** | Same as W7 | **A** |

**Critical ordering note:** failures at **W1–W4** occur **before** `instance_id` mint and **before** B5 publisher exists — historically the largest worker blind spot. DG-001B **does not** claim coverage for W1–W4.

**Evidence:** `background_worker_main.py`; `process_bootstrap.py`; DG-001B R6 closure.

### 4.5 Platform guarded bootstrap primitive

| ID | Primitive | Role | Preconditions for central report |
| -- | --------- | ---- | -------------------------------- |
| **G1** | `run_guarded_hosted_process_bootstrap` | Sync callback guard; emits `APPLICATION_FAILED` on exception | B1 context + B4 publisher minimum; **Problem projection requires B5 composed publisher** |
| **G2** | `HostedProcessBootstrapPhase` taxonomy | Bounded `phase` in payload | `configuration`, `composition`, `dependency_resolution`, `worker_construction`, `startup`, `engine_construction`, `engine_contract_validation` |

Primitive **never** creates `DiagnosticProblem` itself — projection is publisher responsibility (HOST-DIAG-3).

---

## 5. Identity availability matrix

Rows = bootstrap boundaries; columns = identity / authority artifacts at failure time.

| Boundary | `application_id` | `instance_id` | `tenant_id` (diagnostic) | `HostedDiagnosticTenantBinding` | Observability publisher (B4) | `DiagnosticOrchestrator` (B5) | `APPLICATION_FAILED` emit | `DiagnosticProblem` safe |
| -------- | :--------------: | :-----------: | :----------------------: | :-----------------------------: | :--------------------------: | :---------------------------: | :-----------------------: | :----------------------: |
| P0 PRE-PYTHON | — | — | — | — | — | — | — | — |
| P1 PYTHON-BOOTSTRAP | — | — | — | — | — | — | — | — |
| P2 Proof orchestrator | partial | — | partial | — | — | — | — | — |
| A0–A2 ASGI strict | ✓ | — | partial (profile) | — | — | — | — | — |
| H0–H3 Pre-supervisor hosted | ✓ | — | — | — | — | — | — | — |
| H4 Default hosted runner @ failure | ✓ | ✓ | — | — | ✓ | — | ✓ | — |
| H5 Pre-engine (HOST-DIAG-3 wired) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| H6 Engine lifecycle (wired) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| W1–W4 Worker pre-B5 | ✓ | — | partial | — | — | — | — | — |
| W5 Diagnostics build fail | ✓ | — | ✓ | ✓ | — | — | — | — |
| W6 Context mint | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| W7–W8 Guarded B6/B7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Legend: ✓ = available at failure boundary; — = not available; partial = may exist in config but not diagnostic contract.

---

## 6. Failure surface classification

### Category A — Can report centrally today

Prerequisites satisfied: identity + ownership + HOST-DIAG-3 contract.

| Surface | Condition | Mechanism |
| ------- | --------- | --------- |
| H5 Supervisor pre-engine | Product wires `event_publisher_factory` → `build_hosted_application_diagnostic_event_publisher` | `APPLICATION_FAILED` → HOST-DIAG-3 → `list_problems` (DG-001D) |
| H6 Engine lifecycle failure | Same wiring | Engine `_publish_lifecycle_failed_event` → HOST-DIAG-3 |
| W7 B6 worker construction | LKW worker always composes B3–B5 before guard | `run_guarded_hosted_process_bootstrap` + composed publisher (DG-001B) |
| W8 B7 worker startup | Same | Same |

### Category B — Cannot report centrally yet

Missing identity, tenant authority, publisher, or B5 stack.

| Surface | Primary gap | Operator sees |
| ------- | ----------- | ------------- |
| P0, P1 | No Python host / no identity | Shell / stderr |
| P2 | No publisher; no `instance_id` | Proof KV / logs |
| A0–A2 | No hosting event spine | ASGI / uvicorn errors |
| H0–H3 | Pre-`instance_id` or pre-publisher | Python exceptions |
| H4 Default hosted | B5 not wired (by design — DG-001A) | Observability only |
| H7 Bare hosting CLI | No composition | stderr |
| **W1–W4** | **Pre-B5; pre-`instance_id` (W1–W4)** | **stderr / exit code** |
| W5 (failure during diagnostics build) | B5 incomplete | stderr |

**Largest remaining Category B cluster:** worker **W1–W4** and hosted **H0–H4** — aligns with prioritization doc rank 4 (B0–B5 prerequisite gaps).

### Category C — Outside DG-001B / this mapping's remediation lane

| Surface | Rationale |
| ------- | --------- |
| P3 External docker/CI children | Platform controller / proof transport; no generic hosting producer |
| Celery / Nexus `create_nexus_celery_worker_app` | Execution worker bootstrap; post-admission task surface (DG-001 parent scope excludes runtime execution) |
| `intergrax/queueing/bootstrap.py`, registry bootstraps | Tier-1 library init; no application identity contract |
| `platform_proofs/.../composition/bootstrap_runtime.py` | Scenario-specific proof composition |
| Diagnostics projection failure after observability export | Non-fatal by design; observability truth preserved |
| Post-startup task/run failures | Execution identity path — separate from pre-execution DG-001 |

---

## 7. Cross-surface comparison (reporting mechanism)

| Mechanism | Surfaces | Reaches Central Diagnostics? |
| --------- | -------- | ---------------------------- |
| Shell exit / echo | P0 | **NO** |
| Python stderr / uncaught exception | P1, A*, H0–H3, W1–W4 | **NO** |
| Structured log | W1 (env gate) | **NO** |
| Proof KV (`failure_reason=`) | P2 | **NO** |
| `HostedApplicationExitRecord` only | Historical pre-DG-001D | **Superseded** for H5 when wired |
| `ObservabilityHostedApplicationEventPublisher` | H4, partial W* | Observability **YES**; Problems **NO** |
| `HostedApplicationDiagnosticEventPublisher` | H5–H6 wired, W7–W8 | **YES** |
| `run_guarded_hosted_process_bootstrap` emit | W7–W8, extensible to earlier phases **if** B5 pre-positioned | Event **YES**; Problem **only with B5 publisher** |

---

## 8. Recommendations (next steps)

Ordered by diagnostic blind-spot severity; **no implementation in DG-001B0**.

| Priority | Slice | Target boundaries | Approach sketch (architecture only) |
| :------: | ----- | ----------------- | ----------------------------------- |
| **1** | **B0–B2 worker/host pre-identity** | W1–W4, optionally H0–H2 | Early `instance_id` mint (B1 before heavy B2) + phased guarded segments once minimal B5 resolvable; accept observability-only for true B0 |
| **2** | **Default hosted HOST-DIAG-3** | H4 | Product documentation + deployment profiles wire factory (DG-001A — intentional default; not platform auto-wire) |
| **3** | **Public launcher typed producer** | P0–P2 | Separate slice per DG-001C R1 §16 item 3 — only after identity + tenant resolvable in-process |
| **4** | **ASGI strict path** | A0–A2 | Optional hosted-adjacent guard if foreground/hosting convergence required; currently Category B by design |

**Invariant preservation:** no diagnostics-core changes; no new event types; no private API; producer owns failure fact; observability before diagnostics.

**Suggested next qualification task:**

```text
DG-001B1-WORKER-PRE-B5-FAILURE-ARCHITECTURE-AUDIT-R1
```

Freeze whether W1–W4 should share `run_guarded_hosted_process_bootstrap` or a sibling early-bootstrap primitive before implementation.

---

## 9. Test evidence

**No new runtime tests added** (audit-only).

Referenced existing qualification tests:

| Module | Confirms |
| ------ | -------- |
| `tests/unit/applications/local_workspace_application/test_public_launcher_bootstrap_diagnostic_qualification.py` | P0–P2 Category B |
| `tests/unit/hosting/test_guarded_process_bootstrap.py` | G1 primitive |
| `tests/unit/hosting/supervisor/test_supervisor_pre_engine_failure.py` | H5 producer |
| `tests/unit/applications/local_workspace_application/test_lkw_background_worker_bootstrap_conformance.py` | W5–W8 wiring |
| `scripts/proof/dg001b_r5_bootstrap_failure_qualification.py` | W7 real qualification |
| `scripts/proof/dg001d_r4_supervisor_failure_qualification.py` | H5 real qualification |

---

## 10. Confirmations

| Check | Status |
| ----- | ------ |
| Diagnostics core unchanged | **YES** |
| LKW production wiring unchanged | **YES** |
| Queue unchanged | **YES** |
| No bypass of Central Diagnostics | **YES** |
| No private API / reflection / dynamic contracts | **YES** |
| No branch / worktree / reset | **YES** |
| No new producers or events in this task | **YES** |

---

## 11. Related documents

- [`DIAGNOSTIC_GAP_LEDGER.md`](DIAGNOSTIC_GAP_LEDGER.md) — DG-001 parent
- [`DG_001_PRE_EXECUTION_STARTUP_FAILURE_VISIBILITY_AUDIT.md`](DG_001_PRE_EXECUTION_STARTUP_FAILURE_VISIBILITY_AUDIT.md) — R1 root audit
- [`DG_001_REMAINING_PRE_EXECUTION_SURFACES_PRIORITIZATION.md`](DG_001_REMAINING_PRE_EXECUTION_SURFACES_PRIORITIZATION.md) — B0–B5 rank 4
- [`DG_001B_FINAL_CLOSURE_R6.md`](DG_001B_FINAL_CLOSURE_R6.md) — W7/W8 qualified
- [`DG_001D_FINAL_CLOSURE.md`](DG_001D_FINAL_CLOSURE.md) — H5 qualified
- [`DG_001A_HOSTED_DEFAULT_HOST_DIAG_3_QUALIFICATION.md`](DG_001A_HOSTED_DEFAULT_HOST_DIAG_3_QUALIFICATION.md) — H4 boundary
- [`DG_001C_PUBLIC_LAUNCHER_BOOTSTRAP_QUALIFICATION.md`](DG_001C_PUBLIC_LAUNCHER_BOOTSTRAP_QUALIFICATION.md) — P0–P2 boundary
