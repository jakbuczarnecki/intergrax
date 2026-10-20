# PROD-Q-R1-STRICT-HTTP-INVOCATION-IDENTITY-QUALIFICATION — Evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (bounded parent **PROD-Q-R1** / DIM20 input; **does not** close **PROD-Q-R1**, **R7-Q** dim **20**, **PROD-Q**, or promote global **FRZ-***) |
| **Task** | **PROD-Q-R1-STRICT-HTTP-INVOCATION-IDENTITY-QUALIFICATION** |
| **Parent** | **PROD-Q-R1** — **NOT CLOSED** |
| **Dependency** | **R7-Q** §C **DIM20** (parent) |
| **Branch** | `development` |
| **START_HEAD** | `8cf15daa9fffd61b579d5d9d4732e5eca87560f2` |
| **Production code delta** | **Bounded** — `StrictSanctionedExternalWorkFollowOnExecutionAdmissionPort` (follow-on HTTP identity reuse); cancel `side_effect_scope_digest` from prior CREATE quote/metadata (**no** new dispatcher/PIS) |

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
| **Result** | Historical **19 passed** @ R1-R2 FINAL_COMMIT; **R1-R3** adds **R10** (module **20** tests — **no** full-module replay @ R1-R3 without operator authorization) |
| **Execution SHA** | See §12 **FINAL_COMMIT** (R1-R3) |
| **Log** | R1-R2: `.tmp/session/PROD-Q-R1-STRICT-HTTP-R1-R2/pytest-r2.log`; R1-R3 bounded: `.tmp/session/PROD-Q-R1-STRICT-HTTP-R1-R3/pytest-r.log`, `pytest-r10.log`, `test_h01` smoke |

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
| **FINAL_COMMIT** | `577805b6c11c55fb3bcf040f3a1b6ab00df1f0bb` |
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

### Remaining limitations (superseded by R1-R3 §12)

- R1-R2 `_closed` observation at qualification boundary — **removed** in **PROD-Q-R1-STRICT-HTTP-R1-R3**.

---

## 12. Corrective lineage — PROD-Q-R1-STRICT-HTTP-R1-R3

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-STRICT-HTTP-R1-R3** |
| **Parent** | **PROD-Q-R1-STRICT-HTTP-R1** |
| **START_HEAD** | `011d058c723ffd3d49f022a4d29e83f8d6ab5462` |
| **FINAL_COMMIT** | `e1f2f75e6b087720a8e816efd756895e8e13f5cd` |
| **Production delta** | **0** |

### Root cause

| ID | Issue | Correction |
|----|-------|------------|
| **P01** | R1-R2 teardown read `StrictProductionServingSession._closed` (private field) | Test-owned bookkeeping: record **only** after **successful** `TestClient` / ASGI lifespan exit via `strict_http_qualification_asgi_lifespan` |
| **P02** | **R09** asserted `fixture.session._closed` | Assert qualification ASGI-lifespan completion record + registry release — **no** private session field reads |
| **P03** | Direct `TestClient` at H04/H05/R09 bypassed shared lifespan contract | All bounded module `TestClient` entry points route through `strict_http_qualification_asgi_lifespan` (incl. `post_strict_http_run`) |

### Ownership after R1-R3

- Production `StrictProductionServingSession.close()` remains the **sole** lifecycle close authority (ASGI lifespan unchanged).
- Qualification bookkeeping is **test-local** (`_sessions_with_completed_qualification_asgi_lifespan`); it **does not** call `close()` and is cleared when registry releases the session.
- Registry teardown skips redundant `close()` only when ASGI lifespan completion was **explicitly recorded** after full successful `TestClient` exit (**R09**).
- Lifespan exit failure or `TestClient` body exception → **no** record (**R10**); registry retains ownership; fail-closed teardown unchanged (**R02**).

### Regression tests (R01–R10)

| ID | Intent |
|----|--------|
| **R01–R09** | Unchanged intent (R1-R2); **R09** uses public bookkeeping only |
| **R10** | Failed ASGI lifespan `close()` does not record completed lifecycle |
| **R06** | **H01–H08** unchanged (smoke: **H01** @ R1-R3 bounded run only) |

### Bounded pytest (@ R1-R3 agent session)

| Selection | Result | Log |
|-----------|--------|-----|
| `-k test_r0` (**R01–R05**, **R07–R09**) | **8 passed**, exit **0**, ~19s | `.tmp/session/PROD-Q-R1-STRICT-HTTP-R1-R3/pytest-r.log` |
| `::test_r10_*` | **1 passed**, exit **0**, ~15s | `pytest-r10.log` |
| `::test_h01_baseline_strict_http_and_sanctioned_external_work` | **1 passed**, exit **0**, ~16s | (session smoke) |

**Not executed @ R1-R3:** full **20/20** module run (operator authorization required per task budget).

### Remaining limitations

- No production public “session already closed” observable — qualification must not invent one; ASGI-lifespan completion is recorded only at test boundary after successful `TestClient` shutdown.
- No production API to introspect unknown leaked global lifecycle slots (unchanged).

---

## 13. Corrective lineage — PROD-Q-R1-STRICT-HTTP-R1-R4

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-STRICT-HTTP-R1-R4** |
| **Parent** | **PROD-Q-R1-STRICT-HTTP-R1** |
| **START_HEAD** | `f8edb258fb214f399022854ff17a17d85293e922` |
| **FINAL_COMMIT** | `566228eb6830254286373d2bbabed4f22d97ffce` |
| **Production delta** | **0** |

### Root cause

| ID | Issue | Correction |
|----|-------|------------|
| **P04** | `@contextmanager` body exception skipped post-`with TestClient` bookkeeping — successful ASGI shutdown not recorded; registry redundant `close()` could add misleading teardown failure | Capture body exception inside `TestClient`; on successful `TestClient` `__exit__`, record completion then re-raise body exception; dual failure → `ExceptionGroup` without completion record |

### Exception-path matrix (qualification ASGI lifespan)

| Body | Shutdown | Completion record | Propagated error |
|------|----------|-------------------|------------------|
| OK | OK | **Yes** | — |
| Fail | OK | **Yes** | Original body exception |
| OK | Fail | **No** | Shutdown exception |
| Fail | Fail | **No** | `ExceptionGroup` (body + shutdown) |
| Session not owned | — | **No close/reset via this helper** | unchanged |

### Regression tests (R01–R12)

| ID | Intent |
|----|--------|
| **R01–R10** | Unchanged intent (R1-R3) |
| **R11** | Body failure + successful shutdown → completion recorded; registry teardown without false failure |
| **R12** | Body + shutdown failure → no completion record; both exceptions preserved |

### Bounded pytest (@ R1-R4 agent session)

| Selection | Result | Log |
|-----------|--------|-----|
| `-k "test_r09_asgi_lifespan_close or test_r10_asgi_lifespan_exit or test_r11_asgi_lifespan_body or test_r12_asgi_lifespan_body"` | **4 passed**, exit **0**, ~16s | `.tmp/session/PROD-Q-R1-STRICT-HTTP-R1-R4/pytest-r09-r12.log` |

**Not executed @ R1-R4:** full **22/22** module run (operator authorization required per task budget).

### Ownership after R1-R4

- Completion record still means **successful public `TestClient` / ASGI context exit** only — not merely entering `finally`.
- **R1-R3** limitation “body exception → no record” — **superseded** for successful-shutdown path (**R11**).

---

## 14. Corrective lineage — PROD-Q-R1-STRICT-HTTP-R1-R5

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-STRICT-HTTP-R1-R5** |
| **Parent** | **PROD-Q-R1-STRICT-HTTP-R1** |
| **START_HEAD** | `a9a189f1899af1f05ea96f8cf3b4c6065760c405` |
| **FINAL_COMMIT** | `4fac9482edd60120cfcec3a498709e2c4335b040` |
| **Production delta** | **0** |

### Root cause

| ID | Issue | Correction |
|----|-------|------------|
| **P05** | `strict_http_qualification_asgi_lifespan` aggregated `BaseException` members into `ExceptionGroup`, which accepts only `Exception` — fatal shutdown/body paths could raise `TypeError` and mask real failures | Track body failures as `Exception` only; re-raise non-`Exception` `BaseException` without aggregation or completion bookkeeping |

### Exception-path matrix (qualification ASGI lifespan)

| Body | Shutdown | Completion record | Propagated error |
|------|----------|-------------------|------------------|
| OK | OK | **Yes** | — |
| `Exception` | OK | **Yes** | Original body exception |
| OK | `Exception` | **No** | Shutdown exception |
| `Exception` | `Exception` | **No** | `ExceptionGroup` (body + shutdown) |
| Fatal `BaseException` | — | **No** | Fatal propagates (no `ExceptionGroup`) |
| `Exception` | Fatal `BaseException` | **No** | Fatal propagates (no `ExceptionGroup`) |
| Session not owned | — | **No close/reset via this helper** | unchanged |

### Regression tests (R01–R13)

| ID | Intent |
|----|--------|
| **R01–R12** | Unchanged intent (R1-R4) |
| **R13** | Body `Exception` + fatal shutdown during `TestClient` unwind → `KeyboardInterrupt` propagates; **no** completion record; **no** invalid `ExceptionGroup` / `TypeError` (fake `TestClient` — no full ASGI replay) |

### Bounded pytest (@ R1-R5 agent session)

| Selection | Result | Log |
|-----------|--------|-----|
| `::test_r13_asgi_lifespan_fatal_shutdown_propagates_without_invalid_exception_group` | **1 passed**, exit **0**, ~15s | `.tmp/session/PROD-Q-R1-STRICT-HTTP-R1-R5/pytest-r13.log` |

**Not executed @ R1-R5:** full module run or **R09–R12** replay (operator authorization / task budget).

---

## 15. DIM20 scoped contribution — non-CREATE strict HTTP (H09–H11)

| ID | Operation | Verdict | Pytest node |
|----|-----------|---------|-------------|
| **H09** | Strict HTTP **CREATE** (regression) | **PASS** | `test_h09_http_strict_post_traverses_reflex_sanctioned_pis_and_provider` |
| **H10** | Strict HTTP **ACCEPT_QUOTE** | **PASS** | `test_h10_http_strict_post_accept_quote_traverses_reflex_sanctioned_pis_and_provider` |
| **H11** | Strict HTTP **CANCEL** | **PASS** | `test_h11_http_strict_post_cancel_traverses_reflex_sanctioned_pis_and_provider` |
| **H10n/H11n** | Missing prereq / unsupported op / wrong tenant | **PASS** | `test_h10n_*`, `test_h11n_*`, `test_h10_h11n_unsupported_sanctioned_operation_*` |

**Bounded pytest (DIM20):**

```text
uv run pytest -p no:xdist \
  tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py \
  -k "h10 or h11" -q
```

**Negatives A/D/E (sanctioned port missing, PIS admission failure, equivalent replay):** unchanged coverage via **H01–H08**, **PIS-STRICT**, **H06** — no duplicate non-CREATE HTTP tests.

**FINAL_COMMIT (code):** `08f1ef069073fbb91944807777e9b744ca76ebd4` · **evidence tip:** `7a2c4d5d303e7949f89bc9ff75a093c7dee043fe`

**Observed limit:** In-hook provider `cancel_work` assertions that raise inside the dispatch `execute()` closure surface as `side_effect_authorization_failed`; post-hoc PIS/outcome assertions prove intent-before-effect for **H11**.

---

## 16. DIM20 follow-on admission fail-closed static audit

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-R7-Q-DIM20-FOLLOW-ON-ADMISSION-FAIL-CLOSED-QUALIFICATION** |
| **START_HEAD / AUDITED_HEAD (production)** | `4272f885ce558e5e3249d6f67f5f622bbe8833c6` |
| **Production delta** | **0** — no defect; downstream guards sufficient |
| **Status** | **READY FOR AUDIT** (does not close **PROD-Q-R1**, **R7-Q** dim **20**, **PROD-Q**, or global **FRZ-***) |

### 16.1 Audit question (`try_resolve` → `None` → `inner.execute`)

| `None` cause | Intended? | Downstream effect |
|--------------|-----------|-------------------|
| No `META_SANCTIONED_MUTATION_OPERATION` / empty | **Yes** — ordinary CREATE / non-follow-on HTTP | `inner.execute` mints new execution; Tier-2 `resolve_sanctioned_operation_from_metadata` defaults **CREATE** |
| Operation **CREATE** | **Yes** | Same as non-follow-on |
| Invalid operation enum | **Yes** — not follow-on bind | `inner.execute`; Tier-2 `ValueError` → `domain_job` `_fail_closed_domain_payload` |
| **ACCEPT_QUOTE** / **CANCEL** but missing/invalid `META_PRIOR_EXECUTION_ID`, tenant/principal, or store record | **Yes** — admission cannot bind CREATE identity | `inner.execute` on **new** HTTP task/run; Tier-2 still reads **prior** `execution_id` from metadata → `GovernedSanctionedExternalWorkOperationPort._resolve_prior_create_result` → `resolve_create_adapter_result_for_follow_on` → **`None`** → deny `prior_create_snapshot_missing` — **no** `orchestrator.accept`/`cancel`, **no** PIS dispatch, **no** provider mutation |
| Valid follow-on metadata + matching `get_create_follow_on_record` | **No** `None` | Admission rebinds `task_id`/`run_id`/`attempt_id`/`execution_id` + `restore_existing_execution=True` → sanctioned path on authoritative CREATE execution (**H10** / **H11**) |

**Conclusion:** `None` is **not** universally an error; it is the correct out-of-scope signal for CREATE and for admission-layer non-bind. Invalid **ACCEPT_QUOTE**/**CANCEL** must not rely on admission alone — **orchestrator + sanctioned port** enforce prior-create authority before dispatch.

### 16.2 Closed-world call graph (strict serving)

```text
POST /run → GCA task build → StrictProductionServingTenantAuthorityTaskExecutionPort
  → StrictSanctionedExternalWorkFollowOnExecutionAdmissionPort (optional bind)
  → HostTaskExecutionPort (inner)
  → Tier-2 domain_job (require_sanctioned_external_work_mutations=True)
  → GovernedSanctionedExternalWorkOperationPort
  → GovernedExternalWorkOrchestrator (+ PIS / GovernedProviderInvocationDispatchGate on admit paths only)
  → ExternalWorkIntegration (provider)
```

**Second execution authority:** none observed — strict reflex path does not fall back to `adapt_from_step_metadata` when `require_sanctioned_external_work_mutations` is set.

### 16.3 Fail-closed matrix (invalid follow-on)

| Variant | HTTP | Provider | PIS dispatch | Authority escalation |
|---------|------|----------|--------------|----------------------|
| Accept, no prior CREATE (**H10n**) | 200 (request accepted) | **0** `accept_calls` | **No** admitted mutation | **No** |
| Cancel, no prior CREATE (**H11n**) | 200 | **0** `cancel_calls` | **No** | **No** |
| Accept, wrong tenant vs CREATE (**H10n** tenant) | **403** | **0** | **No** (tenant port before follow-on bind) | **No** |
| Accept / cancel, same tenant, wrong principal vs CREATE (**PI-01** / **PI-02**) | 200 | **0** `accept_calls` / **0** `cancel_calls` | **No** new PIS intent | **No** (no CREATE execution rebind; `task_id` ≠ CREATE `task_id`) |
| Unsupported sanctioned op (**H10/H11n** unsupported) | 200 | **0** all fakes | **No** | **No** |

### 16.4 Owners / contracts (unchanged)

| Layer | Owner | Contract |
|-------|-------|----------|
| Follow-on HTTP identity bind | **GCA** `StrictSanctionedExternalWorkFollowOnExecutionAdmissionPort` | `HostTaskExecutionPort` wrapper |
| Prior-create authority | **GCA** `GovernedExternalWorkOrchestrator.resolve_create_adapter_result_for_follow_on` | `GovernedExternalWorkCreateFollowOnRecordV1` + execution store |
| Tier-2 reflex | **external_contractor_adapter** `domain_job` | `SanctionedExternalWorkOperationPort` |
| Dispatch / PIS | **GCA** orchestrator + **ER** lifecycle store / gate | existing PIS-STRICT / DISPATCH-Q1 evidence |

### 16.5 Bounded pytest (@ DIM20 follow-on admission audit)

| Node ID | Result |
|---------|--------|
| `::test_h10n_accept_without_prior_create_fail_closed_zero_provider_calls` | **PASS** |
| `::test_h11n_cancel_without_prior_create_fail_closed_zero_provider_calls` | **PASS** |
| `::test_h10n_wrong_tenant_accept_denied_before_provider` | **PASS** |

```text
uv run pytest -p no:xdist <three node IDs above> -q
```

Log: `.tmp/session/dim20-non-create-qual/pytest-h10-h11-5.log` — **3 passed**, ~20s.

**Not re-run:** full **H10**/**H11** positive chain (existing §15 evidence); **B6** acceptance.

### 16.6 ECAP (relevant scope)

| Area | Disposition |
|------|-------------|
| **A1–A8** | **PASS** (scoped) — single sanctioned composition owner; hard boundaries; fail-closed deny semantics on invalid follow-on |
| **B1–B5** | **PASS** (scoped) — §15 + this static audit + three negative nodes |
| **B6** | **PENDING** — formal independent verdict not in scope |

### 16.7 FRZ (scoped evidence contribution only — global rows remain **OPEN**)

| ID | Scoped note |
|----|-------------|
| **FRZ-SEC-04** | Invalid follow-on denied before provider; wrong tenant **403** at serving binding |
| **FRZ-PRD-03** | Unsupported / missing prior-create → deny, not silent success |
| **FRZ-PRD-07** | Qualification fixture only; no lab/production boundary change |
| **FRZ-PRD-08** | No bypass via `inner.execute` without orchestrator admit |
| **FRZ-REL-03** / **FRZ-REL-05** | Follow-on idempotency/cancel remain orchestrator-owned; admission does not mint alternate invocation |
| **FRZ-OWN-*** / **FRZ-BND-*** / **FRZ-CTR-*** | Unchanged ownership per §3 / ADR-01 |
| **FRZ-REG-02** / **03** / **09** | Existing **H10n**/**H11n** nodes retained; no production change |

### 16.8 Unresolved / limits

| ID | Class | Note |
|----|-------|------|
| — | — | **No IN-SCOPE BLOCKER** from follow-on `None` → `inner` delegation |
| **B6** | **TRACKED** | Parent formal verdict pending |
| Admission vs deny reason | **Guarantee limit** | HTTP 200 on **H10n**/**H11n** reflects request/task acceptance; operation denial is in domain summary / orchestrator reason — not a new HTTP model |

**Audit recommendation (non-binding):** Documented `None` semantics (§16.1) for independent reviewers; **no** duplicate fail-closed gate at admission layer required @ `4272f885…`.

---

## 17. DIM20 CANCEL outcome qualification (P09 scoped)

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-R7-Q-DIM20-CANCEL-OUTCOME-QUALIFICATION** |
| **START_HEAD** | `0c39f8f0f41bf3cb3e9ad479130243b80fee24eb` |
| **Production delta** | **0** — qualification tests + evidence only |
| **Status** | **READY FOR AUDIT** (does not close **PROD-Q-R1**, **R7-Q** dim **20**, **PROD-Q**, or global **FRZ-***) |

### 17.1 Existing contracts (no new cancellation protocol)

| Concept | Canonical type / owner |
|---------|------------------------|
| Sanctioned HTTP **CANCEL** | `SanctionedExternalWorkMutationOperation.CANCEL` → `SanctionedExternalWorkCancelRequest` (`intergrax/contracts/sanctioned_external_work_operation.py`) |
| Provider observation → ERL outcome | `ExternalWorkSideEffectObservation` → `project_external_work_side_effect_to_effect_outcome` (`agents/external_contractor_adapter/external_effect_outcome_projection.py`) |
| Provider refusal | `ExternalWorkErrorCode.CANCELLATION_REJECTED` → `ExternalEffectOutcome.FAILURE` → `ProviderInvocationStatus.FAILED` |
| Provider uncertainty | `ExternalWorkErrorCode.PROVIDER_OUTCOME_UNCERTAIN` → `ExternalEffectOutcome.UNKNOWN` → `ProviderInvocationStatus.UNKNOWN` → `GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN` (`applications/governed_contractor_application/host/lifecycle_states.py`) |
| Intent-before-effect / no outcome retry | `GovernedProviderInvocationDispatchGate` + `persist_provider_invocation_outcome` (`applications/governed_contractor_application/host/provider_invocation_lifecycle.py`) |
| Orchestrated cancel | `GovernedExternalWorkOrchestrator.cancel` — `CANCELLED` host state **only** after successful attestation on **ALLOW** + proof path; failed/unknown PIS outcomes do **not** claim physical cancellation (`applications/governed_contractor_application/host/orchestrator.py`) |

**Not claimed:** guaranteed physical termination at provider; automatic ordinary redispatch after failed/unknown cancel; new recovery owner or alternate dispatch path.

### 17.2 Qualification matrix (C01–C04)

| ID | Scenario | Verdict | Evidence |
|----|----------|---------|----------|
| **C01** | Provider confirms cancel (happy path) | **PASS** (code/evidence review — **H11** not re-run) | §15 **`test_h11_http_strict_post_cancel_traverses_reflex_sanctioned_pis_and_provider`** @ `08f1ef069073fbb91944807777e9b744ca76ebd4` — intent before effect, `ProviderInvocationStatus.SUCCEEDED`, exactly **+1** `cancel_calls` |
| **C02** | Provider refuses cancel (`CANCELLATION_REJECTED`) | **PASS** | **`test_h11c_http_strict_cancel_provider_rejection_persists_failed_not_success`** — PIS `FAILED` + `error_code=cancellation_rejected`; **not** `SUCCEEDED`; single provider dispatch; `evaluate_provider_invocation_recovery` ≠ `IDEMPOTENT_REPEAT` for `external_work.cancel_work.v1` |
| **C03** | Provider outcome uncertain (`PROVIDER_OUTCOME_UNCERTAIN`) | **PASS** | **`test_h11u_http_strict_cancel_provider_uncertain_persists_unknown_not_cancelled`** — PIS `UNKNOWN`; no success/cancelled attestation path; recovery ≠ `IDEMPOTENT_REPEAT` (aligned with **H03** UNKNOWN redispatch posture) |
| **C04** | Failure before provider effect (admission / missing prereq) | **PASS** (existing negatives + static audit) | §16 **`test_h11n_cancel_without_prior_create_fail_closed_zero_provider_calls`**; GR-7-A3 **`test_intent_persistence_failure_zero_provider_calls`** (`applications/governed_contractor_application/tests/host/test_gr7_a3_durable_provider_invocation.py` @ `0c39f8f0…` tree) — zero `cancel_work`, no false terminal cancel outcome |

### 17.3 Bounded pytest (@ this task)

```text
uv run pytest -p no:xdist \
  tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py::test_h11c_http_strict_cancel_provider_rejection_persists_failed_not_success \
  tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py::test_h11u_http_strict_cancel_provider_uncertain_persists_unknown_not_cancelled \
  -q
```

Log: `.tmp/session/dim20-cancel-outcome-qual/pytest-h11c-h11u-3.log` — **2 passed**, ~13s.

**Not re-run:** **H11** positive chain; **H11n**/**H10n** negatives (§16); full PROD-Q module; **B6**.

### 17.4 Owners / call graph (unchanged)

```text
POST /run → GCA strict execution → sanctioned follow-on admission → Tier-2 domain_job
  → GovernedSanctionedExternalWorkOperationPort → GovernedExternalWorkOrchestrator.cancel
  → ExternalWorkAdapter.cancel_and_map → PIS gate → ExternalWorkIntegration.cancel_work
```

### 17.5 ECAP (scoped)

| Area | Disposition |
|------|-------------|
| **A1–A8** | **PASS** (scoped) — single orchestrator/PIS owner; contracts over ad-hoc status; fail-closed on missing prereq |
| **B1–B5** | **PASS** (scoped) — §15 **H11** + §16 negatives + **H11c**/**H11u** |
| **B6** | **PENDING** |

### 17.6 FRZ (scoped contribution only)

| ID | Scoped note |
|----|-------------|
| **FRZ-REL-03** | Recovery evaluation blocks ordinary `IDEMPOTENT_REPEAT` on failed/unknown cancel PIS rows (**H11c**/**H11u**) |
| **FRZ-REL-05** | Refusal ≠ success; uncertainty → `UNKNOWN` / `EXECUTION_OUTCOME_UNKNOWN`, not `CANCELLED` |
| **FRZ-SEC-04** | Sanctioned port + policy gate before `cancel_work` (unchanged) |
| **FRZ-PRD-03** / **08** / **10** / **12** | Truthful PIS outcomes; no bypass; operator-visible failure/unknown; reconciliation/recovery remains ERL-owned |
| **FRZ-REG-02** / **03** / **09** | New regression nodes **H11c**/**H11u** |

### 17.7 Unresolved / limits

| ID | Class | Note |
|----|-------|------|
| — | — | **No IN-SCOPE BLOCKER** |
| **P09** (Package P CREATE slice) | **TRACKED FREEZE DEBT** | Cancel depth documented here for **R7-Q** / DIM20; does not retroactively close Package P row |
| **B6** | **TRACKED** | Formal independent verdict pending |
| HTTP 200 on operational deny | **Guarantee limit** | Same as §16.8 — task/request acceptance vs domain/orchestrator outcome |
| Terminal diagnostic bridge | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Qual sessions may log `policy_bundle_id` reconstruction noise; does not affect PIS/outcome assertions in **H11c**/**H11u** |

---

## 18. DIM20 follow-on principal isolation (PI-01 / PI-02)

| Field | Value |
|-------|-------|
| **Task** | **PROD-Q-R1-R7-Q-DIM20-FOLLOW-ON-PRINCIPAL-ISOLATION** |
| **START_HEAD** | `99e75eb71e0159e86914e40c910d1189b5c46227` |
| **Production delta** | **0** — existing `try_resolve_sanctioned_follow_on_execution_admission` principal binding sufficient |
| **Status** | **READY FOR AUDIT** (does not close **PROD-Q-R1**, **R7-Q** dim **20**, **PROD-Q**, or global **FRZ-***) |

### 18.1 Invariant

Same **tenant**, valid prior strict HTTP **CREATE** (principal **A**), follow-on **ACCEPT_QUOTE** or **CANCEL** with principal **B** in metadata must **not**: rebind authoritative CREATE execution; admit PIS dispatch for accept/cancel; invoke provider accept/cancel; report domain `used`/`admitted` success for the unauthorized follow-on.

**Canonical guard:** `applications/governed_contractor_application/host/strict_sanctioned_follow_on_http_execution_admission.py` — `record.principal_id` must match `META_PRINCIPAL_ID`; on mismatch admission returns `None` → downstream `prior_create_snapshot_missing` deny (§16.1).

### 18.2 Evidence matrix

| ID | Scenario | Verdict | Pytest node |
|----|----------|---------|-------------|
| **PI-01** | **ACCEPT_QUOTE**, same tenant, wrong principal | **PASS** | `test_pi01_wrong_principal_same_tenant_accept_denied_zero_provider_calls` |
| **PI-02** | **CANCEL**, same tenant, wrong principal | **PASS** | `test_pi02_wrong_principal_same_tenant_cancel_denied_zero_provider_calls` |

### 18.3 Bounded pytest (@ this task)

```text
uv run pytest -p no:xdist \
  tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py::test_pi01_wrong_principal_same_tenant_accept_denied_zero_provider_calls \
  tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py::test_pi02_wrong_principal_same_tenant_cancel_denied_zero_provider_calls \
  -q
```

Log: `.tmp/session/dim20-principal-isolation/pytest-pi01-pi02.log` — **2 passed**, ~14s @ execution **HEAD** (see **FINAL_COMMIT** below).

**Observed limits:** HTTP **200** on denied follow-on matches §16.8 / §17.7 (request/task acceptance vs domain denial). Principal **B** injected only via `external_work.principal_id` metadata override; CREATE remains principal **A** (`http-inv-qual-principal`). No second API-key principal map added.

**FINAL_COMMIT (code + evidence):** `54a4e6c1ab62236f04ba1e21a4867b62e0d1259e`

### 18.4 ECAP (scoped)

| Area | Disposition |
|------|-------------|
| **A1–A8** | **PASS** (scoped) — single admission owner; fail-closed principal binding; no shadow execution rebind |
| **B1–B5** | **PASS** (scoped) — **PI-01** / **PI-02** executable negatives atop §15–§17 |
| **B6** | **PENDING** |

### 18.5 FRZ (scoped contribution only)

| ID | Scoped note |
|----|-------------|
| **FRZ-SEC-04** | Unauthorized principal cannot drive provider accept/cancel on another principal's CREATE |
| **FRZ-PRD-03** / **08** | No false operational success; orchestrator deny path preserved |
| **FRZ-REL-03** | No alternate PIS intent for unauthorized follow-on |
| **FRZ-TEN-*** (applicable)** | Tenant unchanged; isolation is principal-scoped within tenant |
| **FRZ-OWN-*** / **FRZ-BND-*** / **FRZ-CTR-*** | Unchanged per §16.4 |
| **FRZ-REG-02** / **03** / **09** | Regression nodes **PI-01** / **PI-02** |

### 18.6 Unresolved

| ID | Class | Note |
|----|-------|------|
| — | — | **No IN-SCOPE BLOCKER** |
| **B6** | **TRACKED** | Formal independent verdict pending |
