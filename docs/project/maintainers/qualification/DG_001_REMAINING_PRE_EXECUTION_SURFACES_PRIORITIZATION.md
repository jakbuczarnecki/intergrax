# DG-001 — Remaining pre-execution surfaces prioritization

> **Task:** DG-001-REMAINING-PRE-EXECUTION-SURFACES-PRIORITIZATION  
> **Mode:** audit / prioritization only — **no implementation**  
> **Parent:** DG-001 — PRE-EXECUTION / OPERATOR STARTUP FAILURE VISIBILITY (`PARTIALLY ADDRESSED`)  
> **Qualified sub-slice:** DG-001B worker bootstrap B6/B7 = **CLOSED / QUALIFIED** (R6 @ `aac75b8e8dfc82192595eaacd0a561249f47f171`)  
> **Start HEAD:** `7720341bb17b1f6f1fa340c233e06ea80d118323`  
> **Audit date:** 2026-09-07
>
> **Post-closure reconciliation (2026-09-08):** **DG-001D = CLOSED / QUALIFIED** — [`DG_001D_FINAL_CLOSURE.md`](DG_001D_FINAL_CLOSURE.md). Sections §2–§3 below remain **historical pre-R2 audit evidence** for prioritization rationale; do not treat DG-001D row status as current operator truth.

---

## 1. Verdict

```text
DG-001 REMAINING PRE-EXECUTION SURFACES PRIORITIZATION = PASS
```

Closure ancestor check: `git merge-base --is-ancestor aac75b8e8dfc82192595eaacd0a561249f47f171 HEAD` → **exit 0**.

---

## 2. Remaining surfaces (explicit)

| ID | Surface | Status |
|----|---------|--------|
| **DG-001A** | Hosted/default production wiring — HOST-DIAG-3 publisher coverage | Open |
| **DG-001C** | Public proof / launcher PYTHON-BOOTSTRAP before host publisher | Open |
| **DG-001D** | Supervisor / pre-engine failure before `HostedApplicationEngine` | **CLOSED / QUALIFIED** (final closure; table is historical pre-R2 audit) |
| **B0–B5** | Bootstrap failures before diagnostic prerequisites exist | Open |

**Not merged into DG-001B.** DG-001B qualifies only guarded B6/B7 after B3–B5 exist.

---

## 3. Evidence table

| Surface | Real failure example | Current canonical evidence | Current operator visibility | First missing boundary | Reusable fix candidate | Priority |
| ------- | -------------------- | -------------------------- | --------------------------- | ---------------------- | ---------------------- | -------- |
| **DG-001D** | `HostedApplicationSupervisor` `engine_factory` raises → `HostedApplicationExitRecord` (`exit_kind=supervisor_error`, `reason_code=engine factory failed`); no `APPLICATION_FAILED` | `intergrax/hosting/supervisor/supervisor.py` `_build_engine` / `classify_exception`; unit tests in `tests/unit/hosting/supervisor/` | Supervisor result / logs only; **no** `list_problems` | Pre-engine supervisor path does not publish hosting lifecycle failure event | Supervisor publishes `APPLICATION_FAILED` via existing `event_publisher` + optional `HostedApplicationDiagnosticEventPublisher`; bounded phase taxonomy extension (`process_launch` / `application_bootstrap`) | **1 — highest** |
| **DG-001C** | Historical `ModuleNotFoundError: local_workspace_application` at PYTHON-BOOTSTRAP (ledger DG-001); launcher fixed via `uv run --project` (`run-lkw-core-platform-proof-windows.bat`) but diagnostics unchanged | Ledger + R1 audit §7; `.bat` transport-only; failure before `HostedApplicationEventPublisher` | Shell stderr / exit code only | No Python module host / no publisher construction | Product/bootstrap composition publishes bounded failure once `application_id` + tenant binding resolvable; reuse `HostedApplicationEvent` when identity exists | **2** |
| **DG-001A** | Default `run_hosted_application` → `ObservabilityHostedApplicationEventPublisher` only; engine `APPLICATION_FAILED` reaches observability, not Central Diagnostics | `intergrax/hosting/runner.py` `_default_runner_factories`; LKW foreground optional wiring in `hosting/foreground.py` | Observability export when engine fails; **no** Problem without product HOST-DIAG-3 | Default `event_publisher_factory` not diagnostic-composed | Product entrypoints wire `build_hosted_application_diagnostic_event_publisher` with `HostedDiagnosticTenantBinding` + durable `DocumentStore` | **3** |
| **B0–B5** | LKW `activate_local_workspace_reference_production_authority()` fails before `build_local_workspace_worker_bootstrap_diagnostics`; worker `main()` env gate (`LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS`) exits before B3 | `background_worker_main.py`; DG-001B architecture §8 ladder | stderr / process exit | B3–B5 not constructed | Guarded bootstrap segments per phase once identity/tenant resolvable; or accept observability-only for pre-B5 (documented) | **4 — prerequisite gaps** |

---

## 4. Boundary matrix (B0–B7)

Evidence from code (`process_bootstrap.py`, `background_worker_main.py`, `runner.py`, `supervisor.py`, DG-001B R6).

| Surface | App identity | Instance identity | Tenant available | Observability available | Diagnostics available |
| ------- | ------------ | ----------------- | ---------------- | ----------------------- | --------------------- |
| **B0** | NO | NO | NO | NO (no publisher) | NO |
| **B1** | YES when caller supplies `application_id` (`HostedProcessBootstrapContext.create`) | YES — minted at B1 | NO | NO | NO |
| **B2** | YES | YES (if B1 done) | PARTIAL — profile/settings may exist; no diagnostic binding | NO (unless B4 wired) | NO |
| **B3** | YES | YES | YES when product constructs `HostedDiagnosticTenantBinding` (e.g. LKW `environment_profile.profile_id`) | NO until B4 | NO until B5 |
| **B4** | YES | YES | YES when B3 wired | YES — `ObservabilityHostedApplicationEventPublisher` | NO |
| **B5** | YES | YES | YES | YES | YES — `HostedApplicationDiagnosticEventPublisher` + `DiagnosticOrchestrator` |
| **B6** | YES | YES | YES (LKW worker wired) | YES | **QUALIFIED** (DG-001B R5-R1-A) |
| **B7** | YES | YES | YES (LKW worker wired) | YES | **QUALIFIED** (DG-001B R5-R1-A) |

**Supervisor pre-engine (DG-001D adjunct):** `application_id` YES (`definition.application_id`); `instance_id` YES (minted before `engine_factory` in `HostedApplicationSupervisor.run`); tenant **available to product** via profile binding but **not wired** in default runner; observability publisher **present** but **not used** for pre-engine failure; diagnostics **NO**.

---

## 5. Default hosted path coverage

```text
DEFAULT HOSTED PATH COVERAGE
```

| Path | Publisher | HOST-DIAG-3 / Central Diagnostics | Evidence |
| ---- | --------- | ----------------------------------- | -------- |
| Platform `run_hosted_application` default | `ObservabilityHostedApplicationEventPublisher` | **NO** — observability-only | `runner.py` `_default_runner_factories` |
| Platform supervisor + engine startup failure (wired observability) | Observability export of `APPLICATION_FAILED` | **NO** unless product overrides `event_publisher_factory` | `HostedApplicationEngine._publish_lifecycle_failed_event` |
| LKW foreground `run_local_workspace_hosted_application` | Optional HOST-DIAG-3 when **both** `diagnostic_orchestrator` + `diagnostic_tenant_binding` passed | **OPTIONAL** — not default | `hosting/foreground.py` |
| LKW `python -m local_workspace_application.hosting` production CLI | N/A — exits 1 without composition | **NO** | `hosting/__main__.py` |
| LKW background worker `background_worker_main` | `build_hosted_application_diagnostic_event_publisher` | **YES** (B6/B7 qualified) | `background_worker_main.py` |
| Other Tier-3 apps (uvicorn/docker CMD direct) | None / app-local logging | **NO** | docker `CMD ["uvicorn", ...]` pattern |
| LKW test `hosted_process_launcher.py` | Default (no diagnostic args) | **NO** | test-only entry |
| Public proof `.bat` → `uv run --project` → proof Python | N/A before host module | **NO** | `run-lkw-core-platform-proof-windows.bat` |

**Reconciled statement:** R6 default-path caveat remains **accurate**. LKW background worker is wired; generic foreground and non-LKW products remain observability-only or without hosting publisher.

---

## 6. Public launcher diagnostic coverage

**Chain:** `shell (.bat)` → `uv run --project applications/local_workspace_application python <proof>.py` → module import → proof environment → host composition.

| Stage | Failure class | Canonical platform event | Identity available |
| ----- | ------------- | ------------------------ | ------------------ |
| PRE-PYTHON (`uv` missing, bad path) | Shell exit | NO | NO |
| PYTHON-BOOTSTRAP (`ModuleNotFoundError`, import errors) | Process stderr | NO | NO stable `application_id` / `instance_id` |
| Post-import proof orchestration | Python exception / docker exit | NO (unless host publisher exists) | PARTIAL after manifest/profile load |
| Host composition + publisher | `APPLICATION_FAILED` possible | YES when HOST-DIAG-3 wired | YES |

**Historical evidence:** Ledger DG-001 documents `ModuleNotFoundError: local_workspace_application` before workload start. **Launcher was fixed** (`uv --project` in `.bat`); **diagnostic visibility was not fixed** — DG-001C remains valid.

---

## 7. Supervisor / pre-engine diagnostic coverage

**In-process canonical supervisor:** `HostedApplicationSupervisor` (APP-HOST-5C) inside `run_hosted_application` — **not** a separate OS child-process spawner.

| Failure | Producer today | `APPLICATION_FAILED` | Central Diagnostics |
| ------- | -------------- | -------------------- | ------------------- |
| `engine_factory` exception | `HostedApplicationExitClassifier` → `HostedApplicationExitRecord` | **NO** | **NO** |
| Engine contract mismatch (`instance_id`, digest) | Same — exit record only | **NO** | **NO** |
| Engine `start()` failure | `HostedApplicationEngine` lifecycle | **YES** | Only if HOST-DIAG-3 wired |
| External docker/process CMD failure (child never runs Python) | Container runtime / shell | **NO** | **NO** — platform controller or launcher owner |

**Semantic owner for DG-001D:** **host/supervisor process** (the process that mints `instance_id` and owns `event_publisher`). External orchestrators (docker, CI) are out of generic hosting scope unless a future platform controller slice is added.

---

## 8. DG-001D — diagnostic authority

```text
Where must diagnostic authority live if the child application cannot start?
```

| Level | DG-001D applicability |
| ----- | ---------------------- |
| Application process | **NO** — engine may not exist |
| Host process | **YES** — primary target for in-process supervisor |
| Supervisor | **YES** — `HostedApplicationSupervisor` already holds `event_publisher`, mints `instance_id`, knows `application_id` |
| External platform controller | **PARTIAL** — docker/CI child spawn failures; not covered by current supervisor; separate from generic hosting slice |

**Frozen candidate (not implemented):**

```text
Supervisor / process controller
        │
        ├── canonical application identity
        ├── minted application instance identity
        ├── product diagnostic tenant binding
        │
        ▼
run_guarded_hosted_process_launch(...)   [future sibling — see §10]
        │
        ├── observability
        └── HOST-DIAG-3
                │
                ▼
Central Diagnostic Engine
```

---

## 9. Event contract audit (`HostedApplicationEvent` / `APPLICATION_FAILED`)

| Question | Answer |
| -------- | ------ |
| Can supervisor emit application failure for a process it attempted to start? | **YES semantically** — not implemented today |
| Does `instance_id` exist before successful engine startup? | **YES** — `validate_instance_id(self.instance_id_generator())` before `_build_engine` |
| Who mints it? | `HostedApplicationSupervisor.instance_id_generator` (default `uuid4`) |
| Same canonical application instance identity? | **YES** — `application_id` from definition; `instance_id` per attempt |
| Is lifecycle `FAILED` correct? | **YES** for pre-engine bootstrap failures |
| Separate event type required? | **NO** — reuse `HostedApplicationEvent` / `APPLICATION_FAILED` with bounded phase payload |
| Phase taxonomy extension required? | **YES bounded** — add phases such as `process_launch`, `application_bootstrap` (or supervisor-specific bounded values in existing `phase` field); do not invent new event type |

**Preference honored:** reuse `HostedApplicationEvent`; projection via existing `hosted_application_failure_to_problem_signal` requires `phase` + `reason_code`.

---

## 10. Subject model

**Preferred:** `APPLICATION_INSTANCE` — **sufficient** for supervisor pre-engine failures when `application_id` + `instance_id` are minted (proven in supervisor loop).

**Do not fabricate** TaskId/RunId/execution identity.

**New subject kind:** **NOT required.**

---

## 11. Tenant authority

| Surface | Who knows diagnostic tenant before child/engine starts? |
| ------- | ------------------------------------------------------- |
| DG-001D (supervisor) | **Product** — deployment profile / `HostedDiagnosticTenantBinding` from environment profile; supervisor can receive composed publisher with tenant; default runner does not |
| DG-001C (launcher) | **NOT at PYTHON-BOOTSTRAP**; after profile/manifest load product may know `profile_id` |
| DG-001A (default hosted) | Product must wire binding; platform does not derive tenant |
| B0–B2 | **NO** diagnostic tenant |
| B3+ | Product-owned `HostedDiagnosticTenantBinding` — **forbidden** to derive from execution tenant |

---

## 12. Publisher location

| Surface | Publisher location |
| ------- | ------------------ |
| DG-001B (qualified) | Inside worker process after B3–B5 |
| DG-001D (target) | **Supervisor/host process** — `HostedApplicationSupervisor.event_publisher` already injected; can compose `HostedApplicationDiagnosticEventPublisher` when product supplies tenant + orchestrator |
| DG-001A | Product `event_publisher_factory` override on `run_hosted_application` |
| DG-001C | Bootstrap composition root after identity + tenant resolvable |

**`HostedApplicationDiagnosticEventPublisher` in supervisor process:** **YES — composable** (same pattern as worker). **Missing abstraction:** guarded publish on supervisor pre-engine failure path (no copy/paste publisher logic).

**Persistence:** must project to same Central Diagnostic Engine, Problem lifecycle, durable `DocumentStore`, operator read-side — **no** supervisor-local problem store.

**Observability ordering:** failure fact → observability → Central Diagnostics — supervisor must use canonical `ObservabilityHostedApplicationEventPublisher` first.

---

## 13. `run_guarded_hosted_process_bootstrap` reuse (R2)

| Outcome | Verdict |
| ------- | ------- |
| **A** — reusable unchanged | **NO** — semantic scope is sync bootstrap **callback** inside an already-live process |
| **B** — hosting-level generalization | **PARTIAL** — failure payload + `APPLICATION_FAILED` emission pattern reusable |
| **C** — sibling guarded process-launch primitive | **SELECTED** — `run_guarded_hosted_process_launch(...)` or supervisor-integrated equivalent for async pre-engine `engine_factory` boundary; do not stretch bootstrap primitive |

---

## 14. Priority model (weighted qualitative)

Dimensions: failure-before-diagnostics (5), blast-radius (5), frequency (3), operator-blindness (5), reusability (4), architectural-prerequisite (5), implementation-risk (−3), product-specificity (−5). Scores 1–5 per dimension (higher = more urgent); risk/specificity inverted.

| Surface | Score | Rank |
| ------- | ----: | ---- |
| **DG-001D** | **~113** | 1 |
| **DG-001A** | **~80** | 2 |
| **DG-001C** | **~69** | 3 |

**Ordering:** `DG-001D > DG-001A > DG-001C` — matches expected unless evidence disproves.

**Rule applied:** largest diagnostic blind spot + generic platform leverage + fewest prerequisites → **DG-001D** (earliest boundary with zero guaranteed Central Diagnostics producer for pre-engine failures).

---

## 15. Hard-contract constraints (audit flags)

Future implementation must preserve: no `Any`, no loose containers, no reflection, no dynamic imports, no `type: ignore`, no private-member coupling.

**Existing friction:** `runner.py` `_RunnerFactories.create_event_publisher: Callable[[], Any]` and `event_publisher: Any` in `_build_engine_factory` — candidate for typed hardening in implementation slice; not blocking prioritization.

---

## 16. Reusability and pluginability

| Test | DG-001D next slice |
| ---- | ------------------ |
| Second application without LKW? | **YES** |
| Behind `HostedApplicationEventPublisher` / `ObservabilityExporter` / `DiagnosticOrchestrator` / `DocumentStore`? | **YES** |
| Product-specific mechanism required? | **NO** |
| Queue architecture changes? | **NONE** |
| Diagnostics core changes? | **NONE** (composition/wiring only) |
| New event type? | **NO** |
| New subject kind? | **NO** |

---

## 17. Non-claims

- Does **not** close DG-001 parent.
- Does **not** implement supervisor, launcher, or wiring changes.
- Does **not** redesign process manager / worker scheduler.
- Does **not** claim B0–B5 failures are centrally visible today.
- Does **not** claim public launcher failures reach Central Diagnostics.

---

## 18. Next task

```text
DG-001D-SUPERVISOR-PRE-ENGINE-FAILURE-ARCHITECTURE-AUDIT-R1
```

Architecture audit R1 for supervisor pre-engine failure representation — freeze producer contract, phase taxonomy, tenant wiring, and guarded-launch primitive shape before implementation.

---

## 19. Related documents

- [`DIAGNOSTIC_GAP_LEDGER.md`](DIAGNOSTIC_GAP_LEDGER.md) — DG-001 parent row
- [`DG_001_PRE_EXECUTION_STARTUP_FAILURE_VISIBILITY_AUDIT.md`](DG_001_PRE_EXECUTION_STARTUP_FAILURE_VISIBILITY_AUDIT.md) — R1 root audit
- [`DG_001B_FINAL_CLOSURE_R6.md`](DG_001B_FINAL_CLOSURE_R6.md) — DG-001B closure
- [`DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md`](DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md) — B0–B7 ladder
