# PROD-Q-R1-STRICT-HTTP-INVOCATION-IDENTITY-QUALIFICATION — Evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (bounded parent **PROD-Q-R1** / DIM20 input; **does not** close **PROD-Q-R1**, **R7-Q** dim **20**, **PROD-Q**, or promote global **FRZ-***) |
| **Task** | **PROD-Q-R1-STRICT-HTTP-INVOCATION-IDENTITY-QUALIFICATION** |
| **Parent** | **PROD-Q-R1** — **NOT CLOSED** |
| **Dependency** | **R7-Q** §C **DIM20** (parent) |
| **Branch** | `development` |
| **START_HEAD** | `e5e5d326184b23d3eb93786e60db7a4c2154499d` |
| **Production code delta** | **0** (qualification tests + evidence only) |

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

---

## 1. Controlling stage

| Check | Result |
|-------|--------|
| **PROD-Q-R1** parent | **NOT CLOSED** |
| **Mandatory parent work** | Strict HTTP invocation-identity qualification (this task) after **PIS-STRICT**, **DISPATCH-Q1**, **P3-F** / **IG-F01** |
| **R7-Q** | **NOT QUALIFIED** — dim **20** remains parent-gated |
| **Prerequisites** | **DISPATCH-Q1** ACCEPTED (scoped); **P3-F** READY FOR AUDIT; **ADR-03** RATIFIED |

---

## 2. HTTP / invocation contract (actual guarantee)

| Question | Answer @ HEAD |
|----------|----------------|
| Caller-supplied stable `invocation_id` on strict HTTP? | **No** — `GovernedContractorRunRequestV1` has no `invocation_id` / `task_id` / `run_id` fields (**H01** static) |
| Who mints `invocation_id` for governed external work? | **GCA orchestrator** / explicit `ProviderInvocation` on sanctioned adapter path (`inv-{uuid4}`) — **not** HTTP |
| HTTP retry preserves same run/execution? | **No** — each `POST /run` builds a new `Task`; response `task_id` / `run_id` come from host execution (**H04**) |
| Same payload ⇒ same invocation? | **No** — independent HTTP requests ⇒ distinct execution identities; payload equality does **not** imply shared `invocation_id` |
| Cross-request business idempotency? | **Not promised** — `idempotency_key` in adapter metadata is **not** a platform-wide HTTP↔invocation registry |
| Same canonical `invocation_id` replay | **Ordinary gate** — second dispatch denied (`dispatch_not_admitted`); **IG-F01** + **H02** on lifecycle SQLite PIS |
| Ambiguous / UNKNOWN outcome | **No ordinary redispatch** (**H03**); sanctioned **GR-7** repeat only when policy allows (**H03b**) |
| Provider exactly-once | **Not claimed** |

**Reflex HTTP path note:** `external_contractor.adapt` via `ExternalContractorAdapterAgent` uses step-metadata adapter entry **without** pre-minted `ProviderInvocation`. **Sanctioned production external-work proof** uses `build_governed_external_work_production_runtime` + `GovernedProviderInvocationDispatchGate` on the **same** strict-session `SQLiteProviderInvocationStore` (**H01** / **H08**).

---

## 3. Closed-world owners

| Concern | Owner |
|---------|--------|
| HTTP ingress / tenant auth | **GCA** strict serving + P3-A2 API-key binding |
| Task / run minting | **Host** `HostTaskExecutionPort` / execution facade |
| Durable PIS | **ER** `SQLiteProviderInvocationStore` (lifecycle-owned) |
| Ordinary dispatch | **GCA** `GovernedProviderInvocationDispatchGate` |
| Recovery repeat | **GR-7-A7** `ProviderInvocationRecoveryRepeatPort` (orthogonal to gate) |

---

## 4. Qualification matrix (H01–H08)

| ID | Criterion | Verdict | Module |
|----|-----------|---------|--------|
| **H01** | Strict HTTP admission + sanctioned external work (1 provider call, PIS row) | **PASS** | `test_h01_*` |
| **H02** | Same `invocation_id` gate replay denied | **PASS** | `test_h02_*` |
| **H03** | UNKNOWN — no unsafe ordinary redispatch | **PASS** | `test_h03_*` |
| **H03b** | Policy-allowed recovery repeat (sanctioned port) | **PASS** | `test_h03b_*` |
| **H04** | Independent identical HTTP payloads → distinct `task_id`/`run_id` | **PASS** | `test_h04_*` |
| **H05** | Tenant HTTP deny + PIS isolation | **PASS** | `test_h05_*` |
| **H06** | Invocation conflict fail-closed | **PASS** | `test_h06_*` |
| **H07** | Concurrent gate — ≤1 execute | **PASS** | `test_h07_*` |
| **H08** | Strict session durable PIS + gate wiring | **PASS** | `test_h08_*` |

**Provider dispatch counts (qualification run):** per-test `DeterministicExternalWorkFake.create_calls` — baseline external-work **1** per sanctioned create; gate replay **0** additional; recovery repeat **1** via repeat port only (**H03b**).

---

## 5. Test execution

```text
uv run pytest -p no:xdist \
  tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py
```

| Metric | Value |
|--------|-------|
| **Result** | **19 passed**, 0 failed, 0 skipped (**R1-R2** adds **R07–R09**; **R06** = **H01–H08** unchanged) |
| **Execution SHA** | See §11 **FINAL_COMMIT** |
| **Log** | `.tmp/session/PROD-Q-R1-STRICT-HTTP-R1-R2/pytest-r2.log` |

---

## 6. FRZ (local contribution)

| FRZ | Contribution |
|-----|----------------|
| **FRZ-PRD-08** | Strict-path dispatch + HTTP execution identity boundaries — evidence **READY FOR AUDIT** (global **OPEN**) |
| **FRZ-REL-03 / REL-05** | Intent-before-dispatch; no ordinary replay on ambiguous state |
| **FRZ-SEC-*** / **FRZ-GOV-*** | HTTP tenant deny slice (**H05**) — partial |

---

## 7. Findings

| Class | Item |
|-------|------|
| **IN-SCOPE BLOCKER** | **PROD-Q-R1** parent closure (remaining R1 children) |
| **TRACKED FREEZE DEBT** | Global **FRZ-*** PASS |
| **ENVIRONMENT/TEST ISSUE** | None @ this run |

---

## 8. Corrective lineage — PROD-Q-R1-STRICT-HTTP-R1

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-STRICT-HTTP-R1** |
| **Finding** | Autouse `_isolate_production_process_lifecycle_slot` read `production_process_lifecycle._active_production_process_lifecycle` / `_process_closed` |
| **Fix** | Owned-session registry + public `StrictProductionServingSession.close()` teardown; no pre-test silent slot clearing |
| **Production delta** | **0** |
| **Historical execution** | `cb276479cf76fe0bc269f448a868ea2026ebdae9` (pre-R1 module run) — preserved |
| **R1 replay** | 11/11 @ FINAL_COMMIT on `development` (see Package P reconciliation §5.2 / §8) |

---

## 9. Git report (@ FINAL_COMMIT)

| Field | Value |
|-------|-------|
| **origin** | `origin/development` |
| **START_HEAD** | `e5e5d326184b23d3eb93786e60db7a4c2154499d` |
| **Pre-R1 AUDITED_HEAD** | `cb276479cf76fe0bc269f448a868ea2026ebdae9` |
| **STRICT-HTTP-R1 implementation** | `ea2e929b352c6e09c13d4cec817ab0c64cf87396` |
| **STRICT-HTTP-R1-R1 corrective / FINAL_COMMIT** | `afc466d669b2e119e0de1d90d0c40c6debad848d` (registry cleanup safety **F01–F03**; see §10) |
| **Prior evidence pin** | `075005b7de1c740dc3017337a11f40ae6591e471` / `1aa8725cbfbbff0c21aab50c62094b680f56511b` (preserved lineage) |

---

## 10. Corrective lineage — PROD-Q-R1-STRICT-HTTP-R1-R1

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-STRICT-HTTP-R1-R1** |
| **Parent** | **PROD-Q-R1-STRICT-HTTP-R1** |
| **START_HEAD** | `1aa8725cbfbbff0c21aab50c62094b680f56511b` |
| **Production delta** | **0** |

### F01–F03 root causes

| ID | Root cause | Correction |
|----|------------|------------|
| **F01** | Registry `pop()` before `session.close()` — failed close dropped ownership evidence | Close in LIFO order; **pop only after successful** envelope `close()`; failed close **retains** registration |
| **F02** | `if "already closed" in str(exc): continue` — forbidden string lifecycle classification | Removed; ASGI lifespan may close via public `close()` — **qualification ownership envelope** records successful close and releases registry **without** message inspection |
| **F03** | `BaseException` bucket could mishandle interrupts / typing at teardown boundary | Teardown collects **`Exception`** only; `KeyboardInterrupt` / `SystemExit` propagate; single failure re-raised; `ExceptionGroup` when multiple |

### Ownership after R1-R1 (superseded by R1-R2 §11)

- Historical: ownership envelope — **removed** in **PROD-Q-R1-STRICT-HTTP-R1-R2** (typing boundary defect).

### Regression tests (R01–R05)

| ID | Intent |
|----|--------|
| **R01** | Successful cleanup clears registry |
| **R02** | Failed close preserves evidence and blocks subsequent test |
| **R03** | `already closed` message does not suppress failure |
| **R04** | Reverse-order multi-session close |
| **R05** | No silent global `production_process_lifecycle` slot reset in support module |

### Remaining limitations (R1-R1)

- Superseded by §11.

---

## 11. Corrective lineage — PROD-Q-R1-STRICT-HTTP-R1-R2

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-STRICT-HTTP-R1-R2** |
| **Parent** | **PROD-Q-R1-STRICT-HTTP-R1** |
| **START_HEAD** | `a1e2228f9304bf368d0aebd63265385097459554` |
| **FINAL_COMMIT** | *(set at push — operator GitHub SHA audit)* |
| **Production delta** | **0** |

### Root cause

| ID | Issue | Correction |
|----|-------|------------|
| **T01** | `_QualificationOwnedStrictProductionServingSession.__getattr__` bridged incompatible semantic types | Removed wrapper; registry holds **concrete** `StrictProductionServingSession` for fixtures |
| **T02** | `register_owned_*` declared concrete session but returned envelope | Returns **same** session instance passed to `create_governed_contractor_strict_production_asgi_app` (**R07**) |
| **T03** | Regression doubles used `# type: ignore[arg-type]` | Narrow **`_OwnedStrictHttpInvocationSessionCloseTarget` Protocol** + `register_owned_strict_http_invocation_close_target_for_regression` (**R01–R04** only) |
| **T04** | ASGI lifespan public `close()` vs registry teardown | Release when `StrictProductionServingSession._closed` is already **True** after lifespan — **no** exception-message classification (**R09**, **R03** preserved) |

### Ownership after R1-R2

- Production-facing ASGI composition receives **genuine** `StrictProductionServingSession` (**R07**).
- Registry tracks session identity; teardown uses public `close()` or `_closed` observation on **concrete** session only (qualification boundary — not a production API).
- Failed `close()` retains registration (**R02**); registry slice has no reflection / typing suppression (**R08**).

### Regression tests (R01–R09)

| ID | Intent |
|----|--------|
| **R01–R05** | Unchanged intent (R1-R1) |
| **R06** | **H01–H08** unchanged |
| **R07** | Fixture session is concrete `StrictProductionServingSession` |
| **R08** | Registry support slice: no `__getattr__` / `getattr` / `type: ignore` / `cast` / `Any` |
| **R09** | Post–`TestClient` lifespan close, registry teardown succeeds without false failure |

### Remaining limitations

- No production public “session closed” observable beyond failed second `close()` — qualification uses `_closed` on **concrete** session at teardown boundary only (aligned with other PROD-Q strict HTTP gates).
- No production API to introspect unknown leaked global lifecycle slots (unchanged).
