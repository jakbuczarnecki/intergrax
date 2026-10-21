# PROD-Q-R1-R7-Q-IMPL-01-R2-OD-R2-03 — RESTART SAFETY & OPERATOR ADMISSION

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** |
| **Parent** | PROD-Q-R1-R7-Q — IN PROGRESS / NOT QUALIFIED |
| **START_HEAD** | `15a8fd145711dd3b0d6c4fe94cb8c81bd1198a60` |
| **Evidence SHA** | `15a8fd145711dd3b0d6c4fe94cb8c81bd1198a60` (batched IMPL-01 GCA restart durability @ same SHA) |
| **Controlling decision** | OD-R2-03 @ [`PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_STRICT_COMPOSITION_AUTHORITY_DECISION_R2.md`](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_STRICT_COMPOSITION_AUTHORITY_DECISION_R2.md) §15 |
| **Does not close** | OD-R2-03 · IMPL-01-R2 · R7-Q · PROD-Q · any global **FRZ-*** PASS |

## Qualification level legend

| Level | Meaning |
|-------|---------|
| **REPO-QUALIFIED** | Exact-SHA pytest / architecture gate on `development` |
| **TARGET-ENVIRONMENT-VERIFIED** | Operator replay on declared deployment/storage |
| **PRODUCTION-ACCEPTED** | Independent B6 sign-off + deployment evidence |

`is_durable=True` and laboratory SQLite reopen **do not** imply **PRODUCTION-ACCEPTED**.

## Closed-world owner graph (unchanged)

| Concern | Authority |
|---------|-----------|
| Process composition + GCA bundle | `ProductionProcessComposition` |
| Lifecycle / cleanup | `ProductionProcessLifecycle` |
| Sanctioned mutations | `GovernedSanctionedExternalWorkOperationPort` → `GovernedExternalWorkOrchestrator` |
| PIS | `SQLiteProviderInvocationStore` + `GovernedProviderInvocationDispatchGate` |
| F02 follow-on | `GovernedExecutionStore` (Option B) |
| Physical provider dispatch | `GovernedProviderInvocationDispatchGate` |

**No second** PIS, GCA owner, recovery engine, or dispatch path introduced in this task.

## Four GCA store roles (laboratory SQLite)

| Role | Implementation | REPO-QUALIFIED evidence | Production |
|------|----------------|-------------------------|------------|
| Execution + GER + F02 | `SQLiteGovernedExternalWorkHostStoreBackend` | Q2 + DUR-PROC-01/02 + `test_prod_q_r1_r7_q_gca_sqlite_durable_recovery.py` | **NOT PROVEN** |
| Receipt | same bundle | gca_sqlite round-trip | **NOT PROVEN** |
| Policy bundle artifact | same bundle | validate + reopen | **NOT PROVEN** |
| Continuation | same bundle | reopen tests | **NOT PROVEN** |

## Durability evidence (this package)

| ID | Scope | Result | Limitation |
|----|-------|--------|------------|
| **DUR-PROC-01** | Fresh Python interpreter reopen after graceful backend close | **REPO-QUALIFIED** — `test_dur_proc_01_fresh_python_process_graceful_restart` + worker `dur-proc-01-*` | Not power-loss; not PVC topology |
| **DUR-PROC-02-A** | Abrupt child termination after committed close + ready flag | **REPO-QUALIFIED** — `test_dur_proc_02_committed_state_survives_abrupt_child_termination` | Process kill ≠ storage power-loss |
| **DUR-PROC-02-B** | Incomplete / ambiguous persistence | **REPO-QUALIFIED (controlled inject)** — worker `dur-proc-02-incomplete-*` mirrors Q2-n02 at process boundary | **NOT** mid-transaction kill proof — **TRACKED FREEZE DEBT** for live crash window |
| **DUR-TEN-01** | Cross-tenant recovery deny | **REPO-QUALIFIED** — `test_dur_ten_01_*` via orchestrator GER `tenant_id` + sanctioned port | SQLite files not tenant-partitioned; semantic authorization on GER |
| **DUR-CONS-01** | PIS/GCA non-atomic ambiguity | **REPO-QUALIFIED** — Q2 suite + `test_dur_cons_01_q2_regression_*` | No distributed transaction claimed |

## Operator admission matrix (TARGET-ENVIRONMENT-VERIFIED / PRODUCTION-ACCEPTED)

| Requirement | Operator artifact | Default disposition |
|-------------|-------------------|---------------------|
| Deployment / runtime identifier | Runbook entry + revision id | **OPERATOR ARTIFACT REQUIRED** |
| Storage backend class | Declared GCA+PIS backend (not lab SQLite path) | **OPERATOR ARTIFACT REQUIRED** |
| Persistent volume type + retention | PVC/volume survives pod/process restart | **OPERATOR ARTIFACT REQUIRED** |
| Stable GCA/PIS DB paths | Config map / env contract | **OPERATOR ARTIFACT REQUIRED** |
| Filesystem + SQLite locking | Single-writer topology documented | **OPERATOR ARTIFACT REQUIRED** |
| `journal_mode=WAL`, `synchronous=FULL` | Runtime PRAGMA verification log | **REPO-QUALIFIED** (code) — **TARGET replay REQUIRED** |
| Writer concurrency | Exactly-one process writer per DB pair | **OPERATOR ARTIFACT REQUIRED** |
| Tenant isolation | OS/DB ACL + app tenant binding | **OPERATOR ARTIFACT REQUIRED** |
| Graceful restart evidence | DUR-PROC-01 replay on target | **OPERATOR ARTIFACT REQUIRED** |
| Abrupt termination evidence | DUR-PROC-02-A replay + incident runbook | **OPERATOR ARTIFACT REQUIRED** |
| Schema / migration compatibility | Migration audit for GCA+PIS schema v1 | **OPERATOR ARTIFACT REQUIRED** |
| Backup / restore | Per FRZ-REL-03 operator policy | **OPERATOR ARTIFACT REQUIRED** |
| Incident / recovery ownership | Named on-call + reconciliation authority | **OPERATOR ARTIFACT REQUIRED** |
| Config provenance | Signed deployment manifest | **OPERATOR ARTIFACT REQUIRED** |
| Test timestamp, SHA, result, sign-off | B6 acceptance record | **B6 DEFERRED** |

**IN-SCOPE BLOCKER (unchanged):** **PRODUCTION-QUALIFIED RESTART-DURABLE GCA BACKEND** for integrated strict production beyond laboratory SQLite.

## Applicable FRZ (no global PASS)

| FRZ | Disposition | Evidence |
|-----|-------------|----------|
| FRZ-REL-03 | partial REPO | DUR-PROC-01/02, Q2 |
| FRZ-PRD-03/04/05/08/12 | partial REPO | strict composition gates + subprocess proofs |
| FRZ-SEC-04 | partial REPO | fail-closed negatives |
| FRZ-TEN-* | partial REPO | DUR-TEN-01 |
| FRZ-OWN-* / FRZ-CTR-* / FRZ-BND-* / FRZ-TYP-* | partial REPO | architecture gates @ sibling dossiers |

## ECAP

| Item | Result |
|------|--------|
| A1–A8 | No new authority; contracts preserved |
| B1–B5 | Boundaries + fail-closed durability negatives |
| B6 | **Deferred** — operator-only |

## IMPL-01 GCA restart durability — invariant matrix (batched qualification)

| Invariant | Contract | Evidence SHA | Process boundary | Outcome | Limitation | FRZ |
|-----------|----------|--------------|------------------|---------|------------|-----|
| Committed GER + F02 + PIS outcome survive graceful cross-process reopen | `GovernedExecutionStore` | `15a8fd14…` | **CROSS-PROCESS** (DUR-PROC-01) | **PROVEN — CROSS-PROCESS** | Not power-loss; lab SQLite | FRZ-REL-03 partial |
| Committed state after abrupt child termination | same + close before flag | `15a8fd14…` | **CROSS-PROCESS** (DUR-PROC-02-A) | **PROVEN — CROSS-PROCESS** | Kill ≠ storage power-loss | FRZ-REL-03 partial |
| Incomplete finalization / PIS intent without outcome | orchestrator + dispatch gate | `15a8fd14…` | **CROSS-PROCESS** (DUR-PROC-02-B) | **PROVEN — CROSS-PROCESS** (controlled inject) | Not live mid-txn crash | FRZ-REL-05 partial |
| Missing GER/F02 denies follow-on; no ordinary CREATE redispatch | F02 Option B | `15a8fd14…` | in-process + DUR-CONS-01 | **PROVEN — LAB ONLY** | Q2 full suite not re-run | FRZ-PRD-08 partial |
| Corrupt F02 / GER-without-F02 fail-closed | `GovernedExecutionStore` | `15a8fd14…` | in-process reopen | **PROVEN — LAB ONLY** | Subset @ batched `-k` | FRZ-SEC-04 partial |
| F03 uncertain commit / marker persistence & reopen deny | SQLite host txn + marker | `15a8fd14…` | in-process reopen | **PROVEN — LAB ONLY** | F03 fix in `sqlite_governed_external_work_host_store_bundle.py` @ HEAD | FRZ-REL-03 partial |
| Policy bundle artifact round-trip | `PolicyBundleArtifactStore` | `15a8fd14…` | in-process reopen | **PROVEN — LAB ONLY** | Not subprocess-isolated | FRZ-CTR partial |
| Proof receipt durable round-trip | `ProofReceiptStore` | — | — | **NOT PROVEN** (reopen gate) | No dedicated close/reopen pytest; CREATE path writes receipt but DUR-PROC verify does not assert | FRZ-REL-03 OPEN |
| Continuation durable round-trip | `ContinuationStateStore` | — | — | **NOT PROVEN** (reopen gate) | No dedicated close/reopen pytest @ gca_sqlite module | FRZ-REL-03 OPEN |
| PIS vs GCA non-atomic ordering | separate SQLite files | `15a8fd14…` | cross-process + Q2 regression | **PROVEN — CROSS-PROCESS** (ambiguity) | No distributed txn | FRZ-PRD-08 partial |
| Cross-tenant recovery deny | GER `tenant_id` + sanctioned port | `15a8fd14…` | in-process reopen | **PROVEN — LAB ONLY** | File not tenant-partitioned | FRZ-TEN-* partial |
| Production-qualified restart-durable GCA backend | OD-R2-03 | — | target deployment | **NOT PROVEN — INFRASTRUCTURE REQUIRED** | Lab SQLite only | **IN-SCOPE BLOCKER** |
| Operator B6 / target replay | ECAP B6 | — | production | **NOT PROVEN — INFRASTRUCTURE REQUIRED** | Deferred | FRZ-PRD-* OPEN |

**Cursor disposition:** **GCA RESTART DURABILITY — READY FOR INDEPENDENT AUDIT** (laboratory + cross-process subprocess scope). **BLOCKED ON EXACT MANDATORY EVIDENCE** for production-global and receipt/continuation reopen gates.

## Focused pytest (authoritative command)

```text
uv run pytest -p no:xdist \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_impl_01_r2_od_r2_03_restart_safety_operator_admission.py \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_gca_sqlite_durable_recovery.py \
  -k "dur_proc or test_f03 or test_corrupt_f02 or test_ambiguous_commit or test_ger_without_f02 or test_policy_bundle_round_trip or test_sqlite_ger_round_trip" \
  -q
```

Log: `.tmp/session/impl-01-gca-restart-durability-qual/pytest-batched.log` — **10 passed**, 18 deselected (2026-10-21).

---

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
