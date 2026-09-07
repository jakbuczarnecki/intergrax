# DG-001B Worker Pre-Execution Bootstrap Failure Visibility — Final Closure (R6)

**Verdict:** CLOSED / QUALIFIED

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `e3200afc91667fff26466ae589c3f85415c37693`

**Review HEAD (pre-docs):** `e3200afc91667fff26466ae589c3f85415c37693`

**Task:** `DG-001B-FINAL-CLOSURE-REVIEW-R6` — review-only final closure; no production or test changes.

**Canonical qualification authority:** attempt `dg001b-r5-r1-a-20260907132655`, commit `aafeff9569b1270936da0fbc384d783e14fb3a72`

---

## 1. Verdict

```text
DG-001B WORKER PRE-EXECUTION / BOOTSTRAP FAILURE VISIBILITY = CLOSED / QUALIFIED
```

R6 confirms that the full qualification lineage (R1 architecture freeze → R2 guarded primitive → R2-R1 hard-contract correction → R3 HOST-DIAG-3 composed projection → R3-R1/R2 test-harness hardening → R4 LKW conformance wiring → R4-R1 observability + persistence topology correction → R5 real functional qualification → R5-R1 generic injection hardening + corrected real qualification) satisfies all frozen enterprise invariants **INV-B1..B15**.

Historical R5-A remains **FUNCTIONALLY PASS / SUPERSEDED FOR FINAL ARCHITECTURAL QUALIFICATION** — not reclassified.

**DG-001 overall:** `PARTIALLY ADDRESSED` (parent row unchanged).

**DG-003 operator story projection:** `OPEN` — separate gap; not closed by DG-001B.

---

## 2. Scope

DG-001B scope is exactly:

```text
worker/application bootstrap failure
after diagnostic prerequisites B3–B5 exist,
specifically guarded B6/B7 surfaces
```

**Qualified failure stages:**

| Phase | Enum | Qualification |
| ----- | ---- | ------------- |
| B6 | `worker_construction` | **Real qualified** (R5-R1-A) |
| B7 | `startup` | **Structurally + conformance qualified** (R2/R3/R4 unit + integration) |

**Does not include:** B0 process launch, B1 identity-before-context, B2 configuration resolution, B3 tenant binding construction, B4 observability composition, B5 diagnostic orchestrator composition, supervisor pre-engine, public launcher / `.bat` bootstrap, pre-Python failures, or universal hosting coverage.

---

## 3. Why this is platform-wide (not LKW diagnostics)

DG-001B is a **qualified platform capability**:

```text
HostedProcessBootstrapContext
+ run_guarded_hosted_process_bootstrap
+ HostedApplicationEvent (APPLICATION_FAILED)
+ HOST-DIAG-3 composed publisher
+ APPLICATION_INSTANCE diagnostic subject
```

LKW (`local_workspace`) was the **real qualification surface only** — it proved conformity of canonical production composition with real MongoDB, Elasticsearch, and separate-process operator read. No LKW-specific logic lives in `intergrax/runtime/diagnostics`.

Another Tier-3 application can reuse DG-001B by supplying:

- `run_guarded_hosted_process_bootstrap` (hosting primitive),
- product `HostedDiagnosticTenantBinding`,
- `build_hosted_application_diagnostic_event_publisher` (canonical composition),
- canonical manifest `application_id` + minted `instance_id`.

**No LKW import required.**

---

## 4. Qualification lineage

| Revision | Purpose | Final status |
| -------- | ------- | ------------ |
| **R1** | Architecture freeze | **QUALIFIED** — [`DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md`](DG_001B_WORKER_BOOTSTRAP_TYPED_FAILURE_PRODUCER_ARCHITECTURE.md) |
| **R2** | Guarded bootstrap primitive | **QUALIFIED** after R2-R1 hard-contract correction — `intergrax/hosting/process_bootstrap.py` (`9ce7decd8`, `66dbe9dad`) |
| **R3** | HOST-DIAG-3 composed projection | **QUALIFIED** after harness corrections — `test_hosted_application_diagnostic_integration.py` |
| **R4** | LKW conformance wiring | **QUALIFIED** after R4-R1 observability + persistence topology correction — `fcd023c8f`, `8aabea069` |
| **R5-A** | First real proof | **FUNCTIONALLY PASS / ARCHITECTURAL CORRECTION REQUIRED** — qualification fault embedded in production LKW; immutable |
| **R5-R1-A** | Corrected real proof | **QUALIFIED** — [`DG_001B_REAL_CONTROLLED_BOOTSTRAP_FAILURE_QUALIFICATION_R5.md`](DG_001B_REAL_CONTROLLED_BOOTSTRAP_FAILURE_QUALIFICATION_R5.md) (`aafeff9569b1270936da0fbc384d783e14fb3a72`) |
| **R6** | Final closure review | **CLOSED / QUALIFIED** (this document) |

R5-A is **not** rewritten into an uninterrupted PASS chain.

---

## 5. Frozen invariants matrix

| ID | Invariant | Evidence | Status |
| -- | --------- | -------- | ------ |
| **INV-B1** | Generic pre-execution bootstrap failure primitive owned by `intergrax.hosting`; product only wires | `intergrax/hosting/process_bootstrap.py`; LKW wires via `background_worker_main.py` | **PASS** |
| **INV-B2** | Canonical failure event: `HostedApplicationEvent`, `event_type=APPLICATION_FAILED`, `lifecycle_state=FAILED`; no separate worker diagnostic contract | `run_guarded_hosted_process_bootstrap` emits bounded `HostedApplicationEvent` | **PASS** |
| **INV-B3** | Non-execution subject: `DiagnosticSubjectKind.APPLICATION_INSTANCE`; no TaskId/RunId/AttemptId/ExecutionId | R5-R1-A identities; conformance tests | **PASS** |
| **INV-B4** | Identity: canonical manifest `application_id`, `instance_id` minted once by `HostedProcessBootstrapContext`, `process_role=background_worker`; same context for B6 and B7 | `HostedProcessBootstrapContext.create`; `background_worker_main.py` single context | **PASS** |
| **INV-B5** | Product-owned tenant: `HostedDiagnosticTenantBinding` from environment profile; no execution tenant derivation | `build_local_workspace_worker_bootstrap_diagnostics` uses `environment_profile.profile_id` | **PASS** |
| **INV-B6** | Accurate failure phase: B6→`worker_construction`, B7→`startup`; no phase collapsing | Separate guarded calls with distinct `HostedProcessBootstrapPhase` | **PASS** |
| **INV-B7** | Original exception → diagnostic publication attempted → original exception re-raised | `test_failure_re_raises_original_exception_object`; `test_publisher_failure_does_not_replace_original_bootstrap_exception` | **PASS** |
| **INV-B8** | Observability export first, diagnostic projection second; R5-R1 uses real configured exporter (Elasticsearch), not NoOp | `HostedApplicationDiagnosticEventPublisher.publish`; R5-R1-A E5 PASS | **PASS** |
| **INV-B9** | Central Diagnostic Engine spine: publisher → `PlatformProblemSignal` → `DiagnosticOrchestrator` → `ProblemLifecycleEngine`; no LKW-local engine | `hosted_application_failure_to_problem_signal`; no LKW imports in `intergrax/runtime/diagnostics` | **PASS** |
| **INV-B10** | Worker diagnostic composition and operator read-side share same canonical `DocumentStore` authority | R4-R1 conformance: `test_worker_bootstrap_diagnostic_and_worker_runtime_share_canonical_document_store`; R5-R1-A Mongo replica-set | **PASS** |
| **INV-B11** | Separate process → canonical `DiagnosticReadService` → persisted Problem visible | R5-R1-A Process B reader subprocess; E8 PASS | **PASS** |
| **INV-B12** | Real integration qualification: real child, production composition, Mongo, Elasticsearch, controlled B6, real Problem, real operator read | R5-R1-A attempt `dg001b-r5-r1-a-20260907132655` | **PASS** |
| **INV-B13** | Production: generic `BackgroundWorkerConstructor`; qualification owns `ControlledFailingBackgroundWorkerConstructor` in `scripts/proof/`; no DG001B fault env/enum in production | Grep audit: no `DG001B`/`worker_construction_fault` in production paths; `test_lkw_background_worker_constructor_seam.py` | **PASS** |
| **INV-B14** | Queue architecture independence — no queue redesign required for diagnostic contract | Bootstrap failure at construction seam before queue worker start | **PASS** |
| **INV-B15** | `runtime/diagnostics` does not import LKW; LKW is consumer/proof surface | Import boundary tests; grep audit | **PASS** |

**Result:** **15 / 15 PASS**

---

## 6. Real R5-R1-A evidence (immutable)

Attempt: **`dg001b-r5-r1-a-20260907132655`**

Commit: **`aafeff9569b1270936da0fbc384d783e14fb3a72`**

| Fact | Value |
| ---- | ----- |
| `application_id` | `local_workspace` |
| subject kind | `APPLICATION_INSTANCE` |
| real child process | **YES** |
| real Mongo | **YES** (replica-set) |
| real Elasticsearch | **YES** |
| separate reader process | **YES** |
| execution identity | **NONE** |
| identity fidelity | **100%** |
| FP | **0** |
| FN | **0** |
| child exit | **!= 0** (exit **1**) |
| stage | **`worker_construction`** |
| diagnostic tenant | `local_workspace.product` |
| `instance_id` | `499cb2e3-b0f2-4c0d-9366-95530db40dec` |
| `event_id` | `evt-672f3085-c43b-4d97-a76e-1d920ab6cdad` |
| `ProblemId` | `problem_5f1387c57fb5487989b1137172af96d3` |
| occurrence id | `application_instance:local_workspace:499cb2e3-b0f2-4c0d-9366-95530db40dec` |

**Drift check:** `git diff aafeff9569b1270936da0fbc384d783e14fb3a72..HEAD` on qualification paths → **empty**. R5-R1 proof reused without requalification.

Evidence artifacts: `.tmp/session/dg001b-r5/dg001b-r5-r1-a-20260907132655/`

---

## 7. Reusability / pluginability / modularity

| Assessment | Result |
| ---------- | ------ |
| Reusable without LKW | **YES** |
| Pluginability at boundaries (`ObservabilityExporter`, `DocumentStore`, `BackgroundWorkerConstructor`, `HostedApplicationEventPublisher`, analyzers/grouping) | **PRESERVED** |
| Module separation | **PASS** |

Expected separation (all verified):

```text
hosting              → bootstrap failure fact/event
applications/_shared → hosted diagnostic composition
runtime/diagnostics  → interpretation/problem lifecycle
product (LKW)        → tenant/environment composition
provider adapters    → Mongo/Elasticsearch/etc.
```

No vendor dependency leaks into generic diagnostic contract.

---

## 8. Failure isolation

| Property | Semantics | Evidence |
| -------- | --------- | -------- |
| Diagnostic projection failure | Must not replace bootstrap exception | `run_guarded_hosted_process_bootstrap` re-raises original after publish attempt; projection wrapped in `except Exception` in publisher only |
| Observability failure | Follows platform export policy; does not rewrite bootstrap truth | `ObservabilityHostedApplicationEventPublisher` export policy; bootstrap truth in `HostedApplicationEvent` independent of export success |

---

## 9. Security / data minimization

Canonical persisted failure facts remain bounded:

```text
phase, reason_code, exception_type, process_role
```

Raw exception message **not** in diagnostic state — `test_failure_payload_excludes_raw_exception_message`; R5-R1 sentinel safety proof (secret absent from canonical Problem state).

---

## 10. Persistence / operator read / grouping

| Capability | Status | Basis |
| ---------- | ------ | ----- |
| Durable persistence | **QUALIFIED** | R5-R1-A Mongo; inherits D1 Problem lifecycle |
| Separate-process operator read | **QUALIFIED** | R5-R1-A Process B |
| Problem grouping across instances | **QUALIFIED** | R3 `test_bootstrap_recurrence_groups_across_instances` — same defect groups under one Problem; `instance_id` outside structural fingerprint |
| Scale boundary | **Inherited** from S1 Problem persistence qualification |
| Recovery boundary | **Inherited** from D1; R5-R1 adds cross-process read |

---

## 11. RuntimeEvent / execution identity independence

DG-001B operates **without RuntimeEvent** because failure is pre-execution. Absence of RuntimeEvent does **not** prevent `APPLICATION_INSTANCE` diagnosis.

Execution identity is **structurally not applicable** — not missing data to invent. No Task/Run fabrication.

---

## 12. Hard-contract audit (DG-001B introduced scope)

Production files touched R2–R5-R1 audited for forbidden patterns (`Any`, loose dicts, `type: ignore`, dynamic introspection, qualification coupling):

| File | DG-001B hard-contract |
| ---- | --------------------- |
| `intergrax/hosting/process_bootstrap.py` | **CLEAN** |
| `intergrax/applications/_shared/hosted_application_diagnostic_wiring.py` | **CLEAN** (pre-existing `object.__setattr__` on tenant normalize only) |
| `applications/.../background_worker_constructor.py` | **CLEAN** |
| `applications/.../background_worker_main.py` | **CLEAN** |
| `applications/.../background_worker_factory.py` | **CLEAN** for DG-001B patterns |

No DG-001B-introduced hard-contract violations found.

---

## 13. Residual repository debt

| Debt | Classification | Blocks DG-001B? |
| ---- | -------------- | --------------- |
| Repository Pyright baseline (~8 scoped repository-health errors per maintainer baseline; 2 pre-existing in `background_worker_factory.py` settings typing) | Pre-existing; not introduced by DG-001B | **NO** |
| Full repository Pyright clean | Not claimed | N/A |
| DG-003 rich operator story | Open separate gap | N/A |

---

## 14. Non-claims

DG-001B does **not** prove:

- B0 process launch failures
- B1 identity/bootstrap failures before context
- B2 configuration resolution failures
- B3 tenant binding construction failures
- B4 observability composition failures
- B5 diagnostic orchestrator composition failures
- Supervisor pre-engine failure
- Public launcher / `.bat` bootstrap failure
- Pre-Python failures
- All hosting surfaces globally

**Forbidden overclaim:** “Central Diagnostics now catches all startup failures.”

**Correct claim:** Central Diagnostics is qualified for real hosted worker pre-execution B6/B7 bootstrap failures once B3–B5 diagnostic prerequisites exist.

---

## 15. Remaining DG-001 surfaces

| Slice | Label | Status |
| ----- | ----- | ------ |
| **DG-001A** | Hosted production wiring / default coverage for surfaces still without HOST-DIAG-3 | Open |
| **DG-001C** | Public proof / launcher pre-publisher bootstrap (PYTHON-BOOTSTRAP) | Open |
| **DG-001D** | Supervisor / pre-engine process failure | Open |
| Earlier B0–B5-only failures | Pre-diagnostic-prerequisite bootstrap | Open |

**Default-path caveat (audited R6):** Statement remains **accurate**. Default platform foreground runner (`run_hosted_application` / `_default_runner_factories`) uses `ObservabilityHostedApplicationEventPublisher` only unless product overrides with diagnostic wiring. LKW **background worker** is wired (R4/R5); generic default paths and non-LKW products may still lack central diagnostic publisher.

---

## 16. Qualification level (honest)

| Dimension | Level |
| --------- | ----- |
| DG-001B capability | **CLOSED / QUALIFIED** — problem discoverability with bounded causal facts (DQ-3/DQ-4 equivalent for `APPLICATION_INSTANCE`) |
| Full rich operator causal story | **DG-003 OPEN** — not DQ-5 |

Problem discoverability = qualified. Full operator story projection = separate DG-003 gap.

---

## 17. Ledger reconciliation

Parent row **DG-001** status: **`PARTIALLY ADDRESSED`**

Qualified sub-slice: **DG-001B worker bootstrap visibility = CLOSED / QUALIFIED**

See [`DIAGNOSTIC_GAP_LEDGER.md`](DIAGNOSTIC_GAP_LEDGER.md) DG-001 row update.

---

## 18. R6 regression

**Drift:** none on qualification paths since `aafeff9569b1270936da0fbc384d783e14fb3a72`.

**Focused regression (R6):**

```bash
uv run pytest \
  tests/unit/hosting/test_guarded_process_bootstrap.py \
  tests/unit/hosting/architecture/test_process_bootstrap_import_boundaries.py \
  tests/unit/applications/_shared/test_hosted_application_diagnostic_integration.py \
  tests/unit/applications/local_workspace_application/test_lkw_background_worker_bootstrap_conformance.py \
  tests/unit/applications/local_workspace_application/test_lkw_background_worker_constructor_seam.py \
  tests/unit/scripts/proof/test_dg001b_r5_bootstrap_failure_qualification.py \
  -q
```

**Result:** **71 passed**, 0 failed, 0 skipped

**`--ignore`:** NO  
**New skips:** NONE

Evidence: `.tmp/session/dg001b-r6/pytest.log`

---

## 19. Production change gate

```bash
git diff e3200afc91667fff26466ae589c3f85415c37693 -- intergrax applications
```

**Expected R6 production diff:** **NONE** (docs-only closure)

---

## 20. Final closure statement

```text
DG-001B FINAL CLOSURE REVIEW R6 = CLOSED / QUALIFIED ✅

PLATFORM-WIDE GUARDED BOOTSTRAP FAILURE MECHANISM = QUALIFIED ✅

APPLICATION_INSTANCE DIAGNOSTICS = QUALIFIED ✅

EXECUTION IDENTITY REQUIREMENT = NONE ✅

REAL WORKER PROCESS PROOF = QUALIFIED ✅

REAL OBSERVABILITY EXPORT = QUALIFIED ✅

REAL DURABLE PROBLEM = QUALIFIED ✅

SEPARATE-PROCESS OPERATOR READ = QUALIFIED ✅

GENERIC REUSABLE COMPOSITION = QUALIFIED ✅

PLUGINABILITY = PRESERVED ✅

MODULARITY = PRESERVED ✅

QUEUE ARCHITECTURE COUPLING = NONE ✅

LKW-SPECIFIC DIAGNOSTIC LOGIC = NONE ✅

RAW FAILURE SECRET = NOT PERSISTED ✅

DG-001B = CLOSED / QUALIFIED ✅

DG-001 = PARTIALLY ADDRESSED

LKW = QUALIFICATION SURFACE ONLY
```

**Recommended next Diagnostic Engine task:** remaining **DG-001** pre-execution surface (DG-001A/C/D or B0–B5) **or** **DG-003 OPERATOR DIAGNOSTIC STORY PROJECTION** — per ledger priority; do not automatically return to LKW.
