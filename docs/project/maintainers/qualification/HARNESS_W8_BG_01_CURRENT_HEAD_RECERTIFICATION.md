# HARNESS-W8 / BG-01 — Current-HEAD Background Execution Convergence Recertification

**Status:** READY FOR AUDIT (Cursor session evidence — not closure)  
**Baseline START_HEAD:** `5d5442dae7769671a7532a973428c8203b492ec6` (`development`)  
**Scope:** W8 semantic scope = **BG-01** only; **SCHED-01** not entered.

## W8 reconciliation

| Workstream | State |
| ---------- | ----- |
| EBH-4 | CLOSED (independently accepted) |
| HARNESS-W7 / HOST-01 | CLOSED @ `5d5442dae7769671a7532a973428c8203b492ec6` |
| HARNESS-W8 | BG-01 current-HEAD recertification (**this record**) |
| SCHED-01 | NOT ENTERED (depends on BG-01) |
| HARNESS-FINAL | NOT ENTERED |

**Why HARNESS-W8 = BG-01:** Canonical Harness SSOT ordered workstreams: HOST-01 → BG-01 → SCHED-01. HOST-01 is independently closed; W8 does not absorb scheduling (SCHED-01), CE-01, PLUG-01, or ART-01.

## Background execution graph (current HEAD)

```text
Application / agent
        ↓
TaskQueue / MessageBus
        ↓
transport delivery (lease / ack / retry)
        ↓
background identity / re-entry admission
        ↓
NexusWorkerRuntime
        ↓
HostTaskExecutionPort.execute(...)
        ↓
Governance / root admission (DefaultRootExecutionLauncher)
        ↓
Execution Engine
        ↓
private Nexus
```

## Closed-world BG inventory (§5)

| Surface | Contract | Transport owner | Identity owner | Execution owner | Persistence owner | Governance owner | Status |
| ------- | -------- | --------------- | -------------- | --------------- | ----------------- | ---------------- | ------ |
| enqueue | `TaskRequest` / `TaskQueue.enqueue` | provider / queue | caller supplies tenant; bootstrap mints canonical IDs | n/a | optional idempotency store | n/a | CURRENT |
| provider transport | `TaskQueue` SPI | provider | transport_task_id only | n/a | provider | n/a | replaceable (BG-Q8) |
| worker intake | encoded payload → `NexusWorkerRuntime` | queue worker | `BackgroundExecutionIdentity` + `BackgroundTransportExecutionRef` | `HostTaskExecutionPort` | KV identity persistence | re-entry admission | CURRENT |
| redelivery | broker duplicate delivery | provider retry policy | identity reconciliation via admission | EE terminal truth | identity KV | re-entry gate | CURRENT |
| idempotency | idempotency key + logical handler | transport + store | canonical task/run | handler invoke once | idempotency store | n/a | CURRENT |
| retry | transport max_retries / visibility | dispatcher / provider | AttemptId owned by EE | EE attempt lifecycle | n/a | n/a | separated (BG-Q6) |
| resume | checkpoint bytes on identity | transport | same canonical identity | `HostTaskExecutionPort` + `resume_checkpoint` | checkpoint store | root admission | CURRENT |
| result propagation | `result_codec` / queue result | queue | correlates to transport handle | EE terminal projection | n/a | n/a | CURRENT |
| terminal handling | `TERMINAL_ALREADY_RECORDED` disposition | transport may redeliver | EE owns terminal | no fresh side effect | terminal store | admission | CURRENT |

**Transport identity** (`provider_task_id`, queue handle) ≠ **canonical execution identity** (`tenant_id`, `TaskId`, `RunId`, `AttemptId`, `ExecutionId`). Bridge = `BackgroundExecutionIdentity` + bootstrap/re-entry admission.

## Qualification drift fixed (§6–§10)

| Issue | Before | After |
| ----- | ------ | ----- |
| Stale `execution_wiring` | `governed_contractor_application.host.execution_wiring.build_governed_contractor_host_task_execution` in `test_bg_01_gates.py` | `tests.fixtures.harness_host_task_execution.build_harness_environment_host_task_execution` over governed environment profile |
| Harness env materialization | Raw `NexusLoop` passed where EE expects `EnvironmentOrchestrationMaterialization` | Test fixture wraps `EnvironmentOrchestrationMaterialization(_backend=nexus_loop)` (qualification-only) |
| Missing root authority import | `REFERENCE_ROOT_EXECUTION_AUTHORITY_ADMISSION` used without import in 3 BG unit modules | Explicit import from `testing_support.reference_root_execution_authority_admission` |

## BG-Q1..Q15

| Q | Verdict | Evidence |
| --- | ------- | -------- |
| BG-Q1 | PASS | `tests/qualification/bg_01/test_bg_01_gates.py` — port invocation + production surface scan |
| BG-Q2 | PASS | identity forwarded on Task + host kwargs |
| BG-Q3 | PASS | tenant mismatch → `ValueError`, port not awaited |
| BG-Q4 | PASS | `DefaultRootExecutionLauncher.launch` exactly once (governed harness host stack) |
| BG-Q5 | PASS | idempotent logical task single handler invoke |
| BG-Q6 | PASS | transport retry separate; redelivery does not reconcile new attempt |
| BG-Q7 | PASS | terminal redelivery safe disposition |
| BG-Q8 | PASS | custom `_FakeTaskQueue` plugin path |
| BG-Q9 | PASS | intake import layer gate |
| BG-Q10 | PASS | resume uses host execution + checkpoint kwargs |
| BG-Q11 | PASS | no transport lease as execution timeout |
| BG-Q12 | PASS | no legacy alternate adapters in composition roots |
| BG-Q13 | PASS | contracts vendor-neutral |
| BG-Q14 | PASS | forbidden dynamic ABI gate |
| BG-Q15 | PASS | worker layer gate |

**Catalog:** `tests/qualification/bg_01/catalog.py` — **22 passed** (`uv run pytest -p no:xdist tests/qualification/bg_01/ -q`).

## Identity matrix

| ID kind | Owner | Notes |
| ------- | ----- | ----- |
| provider / transport task id | TaskQueue provider | opaque delivery handle |
| `tenant_id` | canonical identity | fail-closed match payload + worker |
| `TaskId` / `RunId` / `AttemptId` / `ExecutionId` | Execution Engine | continuity through bootstrap → worker → port |

## Retry / redelivery matrix

| Mechanism | Owner | Must not |
| --------- | ----- | -------- |
| broker redelivery / visibility | transport | mint fresh canonical AttemptId by itself |
| `AttemptId` transitions | Execution Engine | be driven by dispatcher retry counters |
| terminal execution | Execution Engine | allow duplicate side effects on redelivery |

## Tenant 16Q

| # | Verdict |
| --- | ------- |
| 1 explicit tenant at enqueue | PASS |
| 2 tenant in `TaskRequest` | PASS |
| 3 tenant in transport ref | PASS |
| 4 tenant in persisted BG identity | PASS |
| 5 worker tenant == identity tenant | PASS |
| 6 payload tenant == identity tenant | PASS |
| 7 mismatch rejected before execution | PASS (BG-Q3 + unit redelivery tests) |
| 8 redelivery preserves tenant | PASS |
| 9 retry does not widen tenant | PASS |
| 10 missing tenant → no global default | PASS |
| 11 checkpoint tenant matches identity | PASS |
| 12 resume preserves tenant | PASS |
| 13 custom queue cannot rewrite tenant | PASS (BG-Q8 path uses same admission) |
| 14 child execution cannot widen tenant | N/A — WITH EVIDENCE (BG scope: host-task worker path; child fencing covered by EE gates outside this record) |
| 15 cross-tenant reentry rejected | PASS (re-entry admission + identity tests) |
| 16 adversarial wrong-tenant proof | PASS (`test_bg_q3_tenant_mismatch_blocks_execution`) |

## Typing classification (semantic vs transport)

| Weak-looking type | Semantic or transport? | Justification | Action |
| ----------------- | ---------------------- | ------------- | ------ |
| `Dict[str, Any]` on `NexusTaskWorkerOutput.result_payload` | transport | encoded result map at queue boundary | none |
| `execute_payload` return `Dict[str, Any]` | transport | worker result serialization | none |
| `identity_persistence.py` DocumentStore vs ConditionalDocumentStore (pyright) | persistence adapter | pre-existing KV wiring typing | report only; no BG semantic change |
| `dispatcher.py` optional retry policy (pyright) | transport | optional provider policy object | report only; pre-existing |

## Replaceability (BG-Q8 / FRZ-RPL scoped)

Custom `_FakeTaskQueue` in BG gates: enqueue → dispatch → `NexusWorkerRuntime` / `HostTaskExecutionPort` without core branching. Contributes scoped evidence toward FRZ-RPL-01..03; **not** global FRZ PASS.

## Pyright (session)

```text
uv run pyright intergrax/runtime/background_execution intergrax/runtime/task/nexus_worker_execution.py intergrax/runtime/task/queued_host_task_execution_adapter.py intergrax/queueing/contracts/task_queue.py intergrax/queueing/worker/execution.py intergrax/queueing/worker/dispatcher.py tests/fixtures/harness_host_task_execution.py
```

**Result:** 4 errors (dispatcher optional member access ×3; identity_persistence DocumentStore protocol ×1) — **pre-existing on START_HEAD**; **0 new** from BG-01 qualification edits.

## Test runs (4 invocations, `-p no:xdist`)

| # | Command | Result |
| --- | ------- | ------ |
| 1 | `tests/qualification/bg_01/` | **22 passed** |
| 2 | `test_background_execution_bootstrap.py` + `test_ue_9a_*` + `test_ue_11e_*` | **21 passed** |
| 3 | `test_nexus_worker_governance_admission.py` | **4 passed** (requires `celery` in venv for `worker_bootstrap` import) |
| 4 | `test_ue_11gp_production_host_execution_gate.py` + `test_ebh_4_r1_nexus_encapsulation_gate.py` + `test_npsc4_2_residual_compatibility_gate.py` | **39 passed, 1 failed** — `test_npsc42_orchestration_backend_access_confined_to_allowlist` (EE allowlist drift on current HEAD; not BG-01 regression) |

## Enterprise audit matrix

| Invariant | Verdict | Evidence |
| --------- | ------- | -------- |
| Contracts over implementations | PASS | TaskQueue / port contracts |
| Hard boundaries | PASS | worker → port only |
| Exactly-one execution owner | PASS | EE via port |
| Exactly-one background intake owner | PASS | `background_execution/*` + worker intake |
| Transport ≠ Execution | PASS | BG-Q6/Q11 |
| Retry ≠ execution attempt | PASS | BG-Q6 |
| Delivery ≠ permission | PASS | BG-Q4 + governance tests |
| Strong typing | PASS (scoped) | semantic `Any` only at transport payloads |
| Vendor-neutral contracts | PASS | BG-Q9/Q13 |
| TaskQueue replaceability | PASS | BG-Q8 |
| No dynamic probing | PASS | BG-Q14 |
| Nexus private | PASS | EBH-4 gate green in run #4 |
| Governance ≠ Execution | PASS | launcher spy BG-Q4 |
| Tenant continuity | PASS | 16Q |
| Idempotency | PASS | BG-Q5 |
| Redelivery safety | PASS | BG-Q7 + UE-9A/11E |
| Resume continuity | PASS | BG-Q10 |
| Fail closed | PASS | BG-Q3 |
| Regression protection | PARTIAL | npsc42 allowlist gate failure tracked |

## FRZ evidence (scoped contribution only)

**global FRZ PASS delta = 0** · **new FRZ-TEN PASS delta = 0**

Candidate IDs exercised by BG-01 gates/tests (not promoted to global PASS): FRZ-EXE-01, FRZ-EXE-02, FRZ-EXE-03, FRZ-EXE-07, FRZ-GOV-01, FRZ-GOV-02, FRZ-GOV-09, FRZ-CTR-01, FRZ-CTR-04, FRZ-CTR-05, FRZ-TYP-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-06, FRZ-RPL-01, FRZ-RPL-02, FRZ-RPL-03, FRZ-REG-01, FRZ-REG-02, FRZ-REG-03, FRZ-REG-06, FRZ-REG-08, FRZ-REG-09, FRZ-TEN-01, FRZ-TEN-02, FRZ-TEN-10, FRZ-TEN-11, FRZ-TEN-12.

## Non-blocking future debt

cron/misfire/durable schedules → **SCHED-01**; universal multi-tenant production background qualification → **TENANT-X** / **PROD-Q**; capacity/DR → **STATE-X** / **QUAL-X**.

## Unresolved findings

| Class | Item |
| ----- | ---- |
| TRACKED FREEZE DEBT | `test_npsc42_orchestration_backend_access_confined_to_allowlist` — orchestration backend accessor allowlist vs current EE composition files |
| ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED | Local venv lacked `celery` until `uv pip install celery>=5.3` for run #3 collection |

**IN-SCOPE BLOCKER = 0** · **unclassified findings = 0**

## Recommendation

```text
BG-01 = READY FOR AUDIT
HARNESS-W8 = READY FOR AUDIT
HARNESS-FINAL = NEXT / NOT ENTERED
GOV-X2 = NOT ENTERED
SCHED-01 = NOT ENTERED (dependency/future workstream)
```
