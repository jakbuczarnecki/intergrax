# PROD-Q-R1-R7-Q-P3-F — Strict PIS / repeat dispatch adversarial qualification evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (scoped **P3-T-08**; does **not** close **P3**, **PROD-Q-R1-R7-Q** dim **20**, **PROD-Q-R1**, **PROD-Q**, or promote global **FRZ-***) |
| **Child** | **PROD-Q-R1-R7-Q-P3-F** |
| **Task** | **P3-T-08** — repeat dispatch / recovery re-entry on strict production external-work path |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** |
| **Dispatch prerequisite** | **PROD-Q-R1-DISPATCH-Q1** — operator **ACCEPTED (scoped)** @ `f96884fdac73bcc8e12cb98ee3460c9e5148b393` |
| **Branch** | `development` |
| **START_HEAD / source SHA** | `11a1203e93dca70ca385faa182ac4f4fe258470e` |
| **Tests executed in this task** | **0** (static continuity + accepted evidence reuse; **DEFAULT ZERO PYTEST** policy) |

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą finalnego zamknięcia zadania.**

---

## 1. Preflight

| Check | Result |
|-------|--------|
| Branch | `development` |
| HEAD vs **EXPECTED_START_HEAD** `11a1203e93dca70ca385faa182ac4f4fe258470e` | **match** |
| Working tree @ qualification start | **clean** |
| **PROD-Q-R1-ADR-03** | **RATIFIED** — §2.3 (`insert_invocation_if_absent` + dispatch only on `INSERTED`) |
| **PROD-Q-R1-DISPATCH-Q1** | **ACCEPTED (scoped)** — dispatch prerequisite for P3-F |
| **PROD-Q-R1-PIS-01 / PIS-02** | **READY FOR AUDIT** (prior slices; not replayed @ this HEAD run) |
| **P3-P0** mandatory scope **P3-F** | In scope — dim **20** adversarial slice |
| Intervening dispatch invalidation @ `11a1203e…` | **none** — gate, store, composition modules continuous with DISPATCH-Q1 static audit |
| **PROD-Q** / global FRZ | **BLOCKED** — no FRZ PASS promoted |

---

## 2. Scope (P3-T-08)

**Question:** Do PIS + GCA dispatch contracts remain enforced through the **strict production composition** and **`ExternalWorkAdapter._tracked_execute`**, not only through directly constructed gate objects?

**Verdict:** **YES for bounded contract + static continuity — READY FOR AUDIT** (operator independent audit required).

**Explicitly not claimed:** provider-side exactly-once; end-to-end business-operation idempotency for a new orchestration attempt with a new `invocation_id`; full strict-session HTTP → provider duplicate-dispatch proof without deferred integration.

**Production code delta:** **none**.

---

## 3. Canonical strict execution graph (@ `11a1203e…`)

```text
open_strict_production_serving_session
  intergrax/applications/_shared/strict_production_serving_launcher.py
  └─ open_production_process_lifecycle(provider_invocation_db=…)
       intergrax/applications/_shared/production_process_lifecycle.py
       └─ open_provider_invocation_store_at → SQLiteProviderInvocationStore (owned)
       └─ ProductionProcessComposition.provider_invocation_store

create_governed_contractor_strict_production_asgi_app(session)
  applications/governed_contractor_application/host/strict_production_serving.py
  └─ create_governed_contractor_process_app(process_composition=session.composition)

GCA host runtime (strict composition builder / reference lifecycle tests)
  └─ build_governed_external_work_production_runtime
       applications/.../host/production_external_work_composition.py
       ├─ requires provider_invocation_store, is_durable is True
       ├─ GovernedProviderInvocationDispatchGate(store, clock)
       └─ ExternalWorkAdapter(
            invocation_dispatch=gate,
            reliability_aware_invocation_dispatch=gate)

Meaningful side effect path
  agents/external_contractor_adapter/external_work_adapter.py
  └─ authorize_and_execute(…, execute=_tracked_execute)
  └─ _tracked_execute:
       requires provider_invocation when dispatch ports set
       → dispatch_after_intent_persisted(provider_invocation, _dispatch_provider)
  └─ GovernedProviderInvocationDispatchGate.dispatch_after_intent_persisted
       applications/.../host/provider_invocation_lifecycle.py
       └─ admit_provider_invocation_intent(store, invocation)
            intergrax/runtime/enterprise_reliability/provider_invocation_lifecycle.py
            └─ store.insert_invocation_if_absent(invocation)
                 SQLiteProviderInvocationStore (BEGIN IMMEDIATE)
       └─ only INSERTED → execute() → provider integration
  └─ post-execute: orchestrator persist_provider_invocation_outcome (no provider retry)

Recovery (orthogonal to ordinary dispatch)
  GR-7-A7 evaluate/decide + ProviderInvocationRecoveryRepeatPort
  intergrax/runtime/enterprise_reliability/provider_invocation_recovery.py
```

| Role | Owner |
|------|--------|
| **Invocation identity** | `GovernedExternalWorkOrchestrator._new_invocation` → `invocation_id=f"inv-{uuid4().hex}"` (`orchestrator.py`); adapter requires caller-supplied `ProviderInvocation` when dispatch wired |
| **Durable intent** | `SQLiteProviderInvocationStore` / `ProviderInvocationStore` contract |
| **Dispatch admission** | `admit_provider_invocation_intent` + `GovernedProviderInvocationDispatchGate` |
| **External effect boundary** | Provider integration inside `_dispatch_provider` after gate admission |
| **Recovery authority** | GR-7-A7 ports — **not** the ordinary dispatch gate |
| **SQLite connection lifecycle** | `open_production_process_lifecycle` owns store; `lifecycle.close_process_resources()` closes (`PIS-02`) |
| **Strict factory invariants** | D4 `_require_venv_production_adapter`; non-durable PIS rejected at `build_governed_external_work_production_runtime` |
| **Bypass (non-sanctioned)** | `ExternalWorkAdapter` with `invocation_dispatch=None` calls provider directly — **not** wired on production builder; `offline_demo.py` is non-production |

---

## 4. Adversarial matrix F-01…F-10

Legend: **PR** = PROVEN AND REUSABLE · **PS** = PROVEN BY STATIC CONTINUITY · **NI** = NOT PROVEN — INTEGRATION REQUIRED

| ID | Invariant | Classification | Evidence |
|----|-----------|----------------|----------|
| **F-01** | Durable intent before `execute()` | **PR** + **PS** | Gate: `admit_provider_invocation_intent` before `execute()` (`provider_invocation_lifecycle.py` L166–201). **PIS-N5** `test_pis_n5_failed_intent_persistence_prevents_execute`. Adapter: `_tracked_execute` always routes through gate when ports set (`external_work_adapter.py` L1053–1071). |
| **F-02** | Same `invocation_id` → `dispatch_not_admitted`, no second `execute()` | **PR** + **PS** | **PIS-N11** `test_pis_n11_gate_reentry_no_second_execute`; message `provider_invocation_intent_already_recorded_dispatch_not_admitted`. Adapter maps `ProviderInvocationPersistenceError` to fail-closed result (L1115–1124). **NI:** orchestrator → adapter second side-effect with same id through strict ASGI. |
| **F-03** | Concurrent dispatch — ≤1 admission | **PR** | **PIS-N12** `test_pis_n12_concurrent_gate_at_most_one_execute`; store **PIS-N4** concurrent `insert_invocation_if_absent`. |
| **F-04** | Uncertain store commit — no execute | **PR** | **PIS-N13**, **N21–N26** uncertain-commit suite @ `test_prod_q_r1_pis_01_provider_invocation_store.py`. |
| **F-05** | Intent without outcome — default re-entry must not silently re-invoke provider | **PR** + **PS** | **PIS-N28** `test_pis_n28_no_second_execute_when_durable_intent_without_outcome`; **PIS-02-G** `test_pis_02_g_unresolved_intent_blocks_dispatch_after_restart` (lifecycle-owned store + gate). Outcome class **uncertain** per GR-7-A7 — not invented. **NI:** strict session replay without direct gate construction. |
| **F-06** | Reopen SQLite after restart — same record blocks repeat dispatch | **PR** + **PS** | **PIS-N6**, **PIS-N27**; **PIS-02-E** restart reads committed intent; **PIS-02-G** reopen + gate negative. |
| **F-07** | `IDEMPOTENT_REPEAT` via GR-7 / `RepeatPort`, not gate weakening | **PR** | **PIS-N14** / **GR-7-A7** `test_gr7_a7_provider_recovery.py` (`test_failed_eligible_policy_repeat_executes_one_mutation`, `test_crash_ambiguity_no_automatic_repeat`, `test_repeat_creates_new_physical_invocation_id`). Gate still rejects **EQUIVALENT_NOOP** on ordinary path. |
| **F-08** | Strict composition wires durable PIS + canonical gate into real adapter | **PS** + **PR** | `production_external_work_composition.py` L291–327; **test_gr7_a8** `test_production_composition_declares_aware_dispatch_binding`; **test_gr7_a3** `test_production_rejects_non_durable_in_memory_store`; **PIS-02-A/B** lifecycle store shared with composition. |
| **F-09** | New `invocation_id` ≠ old invocation — no silent business idempotency | **PS** | `_new_invocation` mints fresh uuid per orchestration call (`orchestrator.py` L1111+). Platform does **not** equate new id to prior business operation unless separate effect-contract / idempotency semantics (**NOT PROVEN** end-to-end). |
| **F-10** | No sanctioned strict entrypoint bypasses dispatch admission | **PS** | Single strict external-work factory: `build_governed_external_work_production_runtime`. Strict ASGI uses `session.composition` from lifecycle — no alternate host adapter factory without gate. Demo/offline paths excluded from strict production claim. **NI:** prove absence of duplicate execute via full `open_strict_production_serving_session` → HTTP → provider mutation counter. |

**Mission mapping (§1):** Items **1–4** → F-01–F-04; **5** → F-05–F-06; **6** → F-07; **7** → F-08 + F-10.

**Defect class @ HEAD:** **none** — no **FAIL — PROVEN DEFECT** on inspected paths after ADR-03 §2.3 wiring.

---

## 5. Reuse of qualified evidence

| Source | Reuse in P3-F |
|--------|----------------|
| **PIS-N1…N14**, **N27**, **N28** | Gate + SQLite store contracts — **PR** (DISPATCH-Q1 matrix @ `f96884fd…`, continuity @ `11a1203e…`) |
| **PIS-02** lifecycle binding | D2 opener / strict composition builder — `test_prod_q_r1_pis_02_provider_invocation_lifecycle_binding.py` |
| **GR-7-A7** | Recovery separation — `test_gr7_a7_provider_recovery.py` |
| **DISPATCH-Q1** | Parent dispatch prerequisite + ownership graph |
| **D4** strict launcher | `open_strict_production_serving_session` + `create_governed_contractor_strict_production_asgi_app` wiring @ `test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` (topology/preconditions — not duplicate-dispatch E2E) |

No duplicate PIS store tests under P3-F filename.

---

## 6. Concurrency and crash-window analysis

| Window | Ordinary dispatch @ platform | Limit |
|--------|------------------------------|-------|
| Two threads, same `invocation_id`, gate entry | One `INSERTED`, one `dispatch_not_admitted` | **PR** (N12) |
| Commit uncertainty after intent write | Persistence error; no `execute()` | **PR** (N13) |
| Crash after provider effect, outcome missing | Same `invocation_id` → gate blocks; recovery = GR-7 ambiguity | **PR**; strict-session E2E **NI** |
| Process restart, same DB | `EQUIVALENT_NOOP` on re-admit | **PR** (N27, PIS-02-G) |
| New orchestration after failure | New `invocation_id` — separate external effect authorization | **F-09** honesty |

---

## 7. Same-invocation vs new-invocation

| Case | Behavior |
|------|----------|
| Retry with **same** durable `invocation_id` on ordinary gate path | **dispatch_not_admitted** — no second provider `execute()` (**PR**) |
| **New** `_new_invocation()` | New row eligible for first dispatch — **not** automatic dedup of business operation (**F-09**) |
| Approved **IDEMPOTENT_REPEAT** | New physical invocation via repeat port (**F-07**) |

---

## 8. Real vs mocked boundaries

| Layer | This qualification |
|-------|---------------------|
| SQLite PIS + gate | Real store in unit/contract tests (historical **PR**) |
| `ExternalWorkAdapter` + production gate | **PS** source continuity; GR-7-A8-R3 tests use recording dispatch ports, not full SQLite+gate+orchestrator loop |
| Strict physical serving / HTTP / real provider | **Not executed** — **DEFERRED** (§10) |

---

## 9. Tests executed

| Field | Value |
|-------|-------|
| **Count** | **0** |
| **Duration** | **0 s** |
| **Policy** | Task §7 DEFAULT ZERO PYTEST; reuse accepted historical nodes only |

---

## 10. Deferred integration-only assertion

**Status:** **DEFERRED — OPERATOR-SCHEDULED QUALIFICATION**

**Intent:** Prove duplicate external provider mutation cannot occur through the **sanctioned** chain  
`open_strict_production_serving_session` → `create_governed_contractor_strict_production_asgi_app` → governed external-work orchestration → `ExternalWorkAdapter._tracked_execute` → gate, when the durable store already holds intent for `invocation_id` without outcome.

**Suggested node ID (not implemented in this slice):**  
`tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_f_strict_repeat_dispatch_negative.py::test_strict_session_external_work_no_second_provider_mutation_same_invocation_id`

**Dependencies:** D4 strict session fixture (`test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` helpers); deterministic external-work fake with mutation counter; bounded HTTP or direct orchestrator invoke — **no** real external provider; **no** multi-process replay unless separately authorized.

**Budget:** operator-scheduled; exceeds default 30 s agent ceiling if full ASGI + activation included.

---

## 11. Enterprise architecture audit (@ `11a1203e…`)

| Check | Result |
|-------|--------|
| Exactly-one `ProviderInvocationStore` authority | Contract + single SQLite per lifecycle |
| Atomic INSERTED / EQUIVALENT_NOOP | `insert_invocation_if_absent` |
| Exactly-one ordinary dispatch gate semantics | `GovernedProviderInvocationDispatchGate` + shared `admit_provider_invocation_intent` |
| Exactly-one strict external-work composition owner | `production_external_work_composition.py` |
| No duplicate execute for same `invocation_id` (ordinary path) | Gate fail-closed — **PR** |
| No unsafe automatic crash recovery dispatch | GR-7 ambiguity; gate blocks re-entry — **PR** |
| Ordinary dispatch vs GR-7 repeat separated | **PR** |
| Tenant / execution scope | Unchanged vs P3-A track; orthogonal to this slice |
| Strong typing / replaceability | D4 strict pin unchanged |
| Governance ≠ execution | Authorization boundary → `_tracked_execute` → gate → provider |
| No shadow idempotency subsystem | None on strict path |
| No lab/reference fallback in strict production | D4 adapter pin |
| Lifecycle cleanup / connection ownership | PIS-02 close semantics |
| Evidence SHA traceability | START_HEAD + reused evidence SHAs in §5 |

---

## 12. FRZ contributions (criteria remain **OPEN**)

| FRZ | Checklist wording (canonical) | P3-F contribution only |
|-----|------------------------------|-------------------------|
| **FRZ-PRD-08** | production-only bypass count = 0 | Static evidence: strict production external-work builder requires durable PIS + injects gate; no sanctioned bypass on `build_governed_external_work_production_runtime`. Integrated bypass count through full strict graph — **NI** (§10). |
| **FRZ-REL-03** | idempotency ownership explicit | Dispatch admission owned by store atomic fact + GCA gate; recovery owned by GR-7-A7 — documented in §3–§4. |
| **FRZ-REL-05** | external operation termination semantics explicit | Unresolved intent without outcome → no ordinary re-dispatch; ambiguity not collapsed into synthetic outcome (**F-05**). |
| **FRZ-REG-02** | every corrected blocker has regression gate | Reuses existing PIS-01/02 + GR-7-A7 gates — **contribution only**. |
| **FRZ-REG-03** | negative tests prove gates detect violations | Reuses `dispatch_not_admitted` negatives — **contribution only**; §10 deferred strict-session negative. |
| **FRZ-REG-09** | critical invariant protected by code/test, not docs only | Gate/store invariants **PR**; strict-session end-to-end negative **NI**. |

No global FRZ promotion.

---

## 13. P3 acceptance readiness

| Item | Status |
|------|--------|
| **P3-T-08** / **P3-F** scoped qualification | **READY FOR AUDIT** |
| **P3** parent | **NOT CLOSED** — other P3 slices remain |
| **R7-Q** dim **20** | **NOT CLOSED** — integrated gap §10 |
| **PROD-Q-R1** | **NOT CLOSED** |
| **Next mandatory stage** | Operator audit of this record; schedule §10 integration if dim **20** closure requires strict-session proof |

---

## 14. Git report

| Field | Value |
|-------|-------|
| **START_HEAD** | `11a1203e93dca70ca385faa182ac4f4fe258470e` |
| **FINAL_COMMIT** | *(single evidence commit — see post-push `origin/development`)* |
| **Tests executed** | **0** |
