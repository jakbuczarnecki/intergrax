# PROD-Q-R1-R7-Q-IMPL-01-R2-OD-R2-03 — RESTART SAFETY & OPERATOR ADMISSION

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** |
| **Parent** | PROD-Q-R1-R7-Q — IN PROGRESS / NOT QUALIFIED |
| **START_HEAD** | `fc176db7fa2290c47849ff3d159299f936a05b2d` |
| **Evidence SHA** | *(set at commit — see Final report)* |
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

## Focused pytest (authoritative command)

```text
uv run pytest -p no:xdist \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_impl_01_r2_od_r2_03_restart_safety_operator_admission.py \
  -q
```

Log: `.tmp/session/impl-01-r2-od-r2-03/pytest-od-r2-03.log`

---

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
