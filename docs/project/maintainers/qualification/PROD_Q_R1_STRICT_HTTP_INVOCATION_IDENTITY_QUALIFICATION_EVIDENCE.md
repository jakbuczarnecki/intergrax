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
| **Result** | **11 passed**, 0 failed, 0 skipped |
| **Log** | `.tmp/session/PROD-Q-R1-STRICT-HTTP-INVOCATION-IDENTITY-QUALIFICATION/pytest.log` |

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

## 8. Git report (@ FINAL_COMMIT)

| Field | Value |
|-------|-------|
| **origin** | `origin/development` |
| **START_HEAD** | `e5e5d326184b23d3eb93786e60db7a4c2154499d` |
| **FINAL_COMMIT** | *(see commit pushing this record)* |
