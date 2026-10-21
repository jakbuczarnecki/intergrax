# PROD-Q-R1-R7-Q — Stable CREATE identity scoped qualification (IMPL-01)

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (scoped qualification contribution; **does not** close **IMPL-01**, **R7-Q**, **PROD-Q**, or promote global **FRZ-***) |
| **Task** | **PROD-Q-R1-R7-Q-STABLE-CREATE-IDENTITY-SCOPED-QUALIFICATION** |
| **Parent** | **PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01** — **NOT CLOSED** |
| **Controlling decision** | [`PROD_Q_R1_R7_Q_STABLE_CREATE_INVOCATION_IDENTITY_DECISION_01.md`](PROD_Q_R1_R7_Q_STABLE_CREATE_INVOCATION_IDENTITY_DECISION_01.md) §6 |
| **Branch** | `development` |
| **START_HEAD** | `1bf97ce8a1f59241a59d834ec009ccc270029702` |
| **Execution SHA** | `1bf97ce8a1f59241a59d834ec009ccc270029702` |
| **Production code delta** | **0** (qualification execution + evidence only) |

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

---

## 1. Mandate and exclusions

| Check | Result |
|-------|--------|
| Identity redesign | **Not performed** |
| New PIS / alternate authority | **Not introduced** |
| Full **PROD-Q** | **Not executed** |
| Cross-process GCA restart (OD-R2-03 production) | **Out of scope** — Q2 uses **in-process reopen** of dual SQLite backends only |
| Package P **B6** ledger | **Unchanged** — operator scoped acceptance preserved |

---

## 2. Test execution (single process, `-p no:xdist`)

```text
uv run pytest -p no:xdist \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_impl_01_stable_create_invocation_identity.py \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2.py \
  tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py \
  -k "impl01 or q2_ or test_h09 or test_h02 or test_h03 or test_h05 or test_h06 or test_h01_baseline or test_h10n or test_h11n or test_h10_h11n or test_h10n_wrong_tenant"
```

| Metric | Value |
|--------|-------|
| **Result** | **23 passed**, 31 deselected, 0 failed |
| **Log** | `.tmp/session/STABLE-CREATE-IDENTITY-QUAL/pytest.log` |

---

## 3. Qualification matrix — unit (orchestrator + PIS gate)

| Scenario | Code owner | Test node | Execution SHA | Result | Provider calls | Remaining risk |
|----------|------------|-----------|---------------|--------|----------------|----------------|
| Stable `gca-pinv:*` minted on first CREATE | GCA orchestrator + `derive_governed_external_work_create_provider_invocation_id` | `test_impl01_orchestrator_create_mints_stable_invocation_id` | `1bf97ce8…` | **PASS** | 1 | — |
| Same canonical identity — no second dispatch | GCA + gate + adapter | `test_impl01_same_create_retry_no_second_provider_dispatch` | `1bf97ce8…` | **PASS** | 1 total; retry 0 | — |
| In-process PIS reopen — stable id unchanged | ER SQLite PIS + GCA | `test_impl01_sqlite_pis_restart_same_create_no_second_dispatch` | `1bf97ce8…` | **PASS** | 1 | Not cross-process restart |
| New authorized CREATE (distinct `execution_id`) | GCA | `test_impl01_different_execution_id_admits_second_create` | `1bf97ce8…` | **PASS** | 2 | — |
| Legacy `inv-*` when GER survives | GCA `_resolve_create_provider_invocation_id` | `test_impl01_legacy_random_invocation_id_preserved_when_ger_survives` | `1bf97ce8…` | **PASS** | per GER | — |

---

## 4. Qualification matrix — dual SQLite Q2 (OD-R2-03 lab / in-process reopen)

| Scenario | Code owner | Test node | Execution SHA | Result | Provider calls | Remaining risk |
|----------|------------|-----------|---------------|--------|----------------|----------------|
| First CREATE persists PIS + GCA | GCA + ER PIS + GCA SQLite | `test_q2_p01_first_create_persists_pis_intent_outcome_and_gca_records` | `1bf97ce8…` | **PASS** | 1 | — |
| Reopen both DBs — retry CREATE | Same | `test_q2_p02_reopen_both_backends_retry_create_zero_additional_provider_calls` | `1bf97ce8…` | **PASS** | 1 total | **In-process reopen only** |
| GCA records removed, PIS retained | GCA + PIS | `test_q2_n01_remove_gca_execution_records_retain_pis_no_second_provider_call` | `1bf97ce8…` | **PASS** | 1 | — |
| PIS intent without outcome — no ordinary redispatch | Gate | `test_q2_n02_pis_intent_without_outcome_denies_ordinary_redispatch` | `1bf97ce8…` | **PASS** | 1 | — |
| GER without F02 — no CREATE redispatch | GCA orchestrator | `test_q2_n03_ger_without_f02_denies_follow_on_and_create_redispatch` | `1bf97ce8…` | **PASS** | 1 | — |
| Corrupt F02 fail-closed | GCA store | `test_q2_n04_corrupt_f02_fail_closed` | `1bf97ce8…` | **PASS** | 0 | — |
| Reopen equivalent CREATE — `EQUIVALENT_NOOP`, no new intent | PIS `insert_invocation_if_absent` | `test_q2_n05_reopen_equivalent_create_stable_id_no_new_pis_intent` | `1bf97ce8…` | **PASS** | 1 | `EQUIVALENT_NOOP` ≠ provider success |

---

## 5. Qualification matrix — strict HTTP (sanctioned port + admission negatives)

| Scenario | Code owner | Test node | Execution SHA | Result | Provider calls | Remaining risk |
|----------|------------|-----------|---------------|--------|----------------|----------------|
| Strict HTTP → sanctioned external work | GCA strict serving + production runtime | `test_h01_baseline_strict_http_and_sanctioned_external_work` | `1bf97ce8…` | **PASS** | 1 | — |
| Same canonical `invocation_id` gate replay | GCA gate | `test_h02_same_canonical_invocation_gate_replay_denied` | `1bf97ce8…` | **PASS** | 1 | — |
| UNKNOWN — no ordinary redispatch | Gate + orchestration | `test_h03_unresolved_external_effect_no_ordinary_redispatch` | `1bf97ce8…` | **PASS** | 1 | — |
| Tenant HTTP + PIS isolation | Strict HTTP auth + PIS | `test_h05_tenant_isolation_http_and_pis` | `1bf97ce8…` | **PASS** | per tenant | — |
| Conflicting invocation identity | PIS conflict | `test_h06_conflicting_invocation_identity_fail_closed` | `1bf97ce8…` | **PASS** | 0 | — |
| **H09** CREATE strict POST → reflex → PIS → provider | Full strict chain | `test_h09_http_strict_post_traverses_reflex_sanctioned_pis_and_provider` | `1bf97ce8…` | **PASS** | 1 | — |
| Accept without prior CREATE | Admission | `test_h10n_accept_without_prior_create_fail_closed_zero_provider_calls` | `1bf97ce8…` | **PASS** | 0 | — |
| Cancel without prior CREATE | Admission | `test_h11n_cancel_without_prior_create_fail_closed_zero_provider_calls` | `1bf97ce8…` | **PASS** | 0 | — |
| Unsupported sanctioned op | Admission | `test_h10_h11n_unsupported_sanctioned_operation_fail_closed` | `1bf97ce8…` | **PASS** | 0 | — |
| Wrong-tenant accept denied | HTTP auth | `test_h10n_wrong_tenant_accept_denied_before_provider` | `1bf97ce8…` | **PASS** | 0 | — |

*Note:* `-k "test_h03"` also selected `test_h03b_sanctioned_recovery_repeat_when_policy_allows` (**PASS**) — sanctioned GR-7 repeat orthogonal to ordinary gate; not a CREATE-identity defect signal.

---

## 6. Audit — `dispatch_not_admitted` message substring (`orchestrator._finalize_side_effect`)

| Failure class | Adapter `reason` | Typical `error_message` | Substring `dispatch_not_admitted`? | Orchestrator branch @ HEAD |
|---------------|------------------|-------------------------|-------------------------------------|----------------------------|
| Real duplicate / `EQUIVALENT_NOOP` | `provider_invocation_intent_persistence_failed` | `provider_invocation_intent_already_recorded_dispatch_not_admitted` | **Yes** | `_handle_create_duplicate_invocation_admission` |
| Ordinary PIS/SQLite persistence fault | same | store/infra messages (closed-world inventory) | **No** | `EXECUTION_FAILED` / `provider_invocation_intent_persistence_failed` |
| Governance / auth deny | `side_effect_authorization_failed` | authorization text | **No** | Not duplicate-admission path |
| Unrelated adapter fault | various | action-specific | **No** | Not duplicate-admission path |

**Verdict:** **No confirmed misclassification defect** at execution SHA — only `GovernedProviderInvocationDispatchGate` emits the controlled substring. **Remaining risk:** orchestrator couples duplicate admission to **text** inside a shared adapter `reason`; future messages containing the substring could mis-route. **Minimal typed hardening (not applied this task):** dedicated `ProviderInvocationDispatchNotAdmittedError` (contracts) → adapter `reason="provider_invocation_dispatch_not_admitted"` → orchestrator matches typed `reason` (no new authority).

---

## 7. ECAP and FRZ

| Reference | Contribution |
|-----------|--------------|
| **ECAP-P4** | Agent does not self-accept **FRZ-*** PASS |
| **ECAP-P2** | Historical stage closure ≠ current-HEAD freeze PASS |
| **FRZ-PRD-08 / FRZ-REL-03/05** | Evidence **READY FOR AUDIT** for stable CREATE identity slice — global rows **OPEN** |
| **FRZ-REG-09** | Traceability via this record + session log |

---

## 8. Exit disposition

**STABLE CREATE IDENTITY — READY FOR INDEPENDENT AUDIT**

Does **not** close **IMPL-01**, **R7-Q**, **PROD-Q**, or **Package P B6**.
