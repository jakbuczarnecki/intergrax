# STATE-X-P0 — State Truth Ownership & Recovery Baseline Lock

**Parent:** STATE-X — Persistence, State & Recovery Certification

**Stage:** STATE-X-P0 (baseline lock — no production changes)

**AUDITED_HEAD:** `658cf95864970d7c8bde1c7005cf178747c68b55`

**Mechanical SSOT:** `tests/qualification/state_x/inventory.py`

**Status:** STATE-X-P0 = **READY FOR AUDIT** (pending independent GitHub SHA audit)

---

## 1. Repository / AUDITED_HEAD

| Field | Value |
|---|---|
| Branch | `development` |
| AUDITED_HEAD | `658cf95864970d7c8bde1c7005cf178747c68b55` |
| `origin/development` (pre-task) | `658cf95864970d7c8bde1c7005cf178747c68b55` |
| Production delta | **0** |

---

## 2. Canonical roadmap state

Per `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` @ AUDITED_HEAD:

- **CTRL-X** = CLOSED
- **STATE-X** = CURRENT / MANDATORY — discovery/implementation/qualification entered via **P0 baseline only**
- **STATE-X-P0** = current task
- **TRACE-X** = PLANNED / MANDATORY — **NOT ENTERED**
- **CONFIG-X / TENANT-X** = not entered by this task

---

## 3. Scope / exclusions

**In scope:** canonical inventory SX-F01..SX-F15, FRZ mapping (no PASS promotion), known blockers SX-B01..B03, mechanical pytest gates.

**Out of scope:** production/runtime fixes, TRACE-X causal reconstruction, CONFIG-X provider activation, global TENANT-X closure, FRZ criterion PASS.

**Forbidden:** omnibus “single database owns all semantics” conclusion; Nexus as second checkpoint truth owner.

---

## 4. FRZ mapping

P0 maps evidence disposition only — **no FRZ-STA / FRZ-REC / FRZ-TEN promotion**.

| Criterion | P0 disposition |
|---|---|
| FRZ-STA-01..08 | Inventoried per family; certification → R1..R4 |
| FRZ-REC-01..10 | Inventoried; certification → R1..R4 |
| FRZ-TEN-04, FRZ-TEN-08 | Local tenant audit disposition per family (mostly BLOCKED pending children) |

---

## 5. State family inventory

Full typed records: `STATE_X_FAMILY_INVENTORY` in `tests/qualification/state_x/inventory.py`.

Mandatory families: **SX-F01 .. SX-F15** (exactly once).

---

## 6. Semantic ownership matrix

| ID | Semantic owner | Canonical contract |
|---|---|---|
| SX-F01 | Long-running checkpoint subsystem | TaskCheckpointPersistence |
| SX-F02 | Long-running checkpoint (tree component) | RuntimeCheckpoint.execution_tree |
| SX-F03 | Decision orchestration / recovery | DecisionCheckpointPersistence |
| SX-F04 | AttemptLifecycleService | AttemptLifecycleStore |
| SX-F05 | Execution terminal plane | ExecutionTerminalStore |
| SX-F06 | Execution lineage subsystem | ExecutionLineagePersistence |
| SX-F07 | Runtime events / evidence | RuntimeEventPersistence |
| SX-F08 | Nexus tracing read plane | RunTraceStore / RunTraceReader |
| SX-F09 | Execution budget plane | RunBudgetPersistence |
| SX-F10 | Side-effect deduplication | IdempotencyStore |
| SX-F11 | Compensation queue | CompensationQueueStore |
| SX-F12 | Human decision persistence | HumanDecisionPersistence |
| SX-F13 | Long-running scheduler | ScheduledResumePersistence / SchedulerLedger |
| SX-F14 | ACP agent checkpoint | AgentCheckpointStore |
| SX-F15 | **Truth ownership: NONE** — cross-family recovery coordination | **None** (coordination only; see `underlying_family_ids`) |

**Invariant:** storage backend ≠ semantic owner (see §23 task spec).

---

## 7. Contract / implementation matrix

See `production_paths`, `implementation_symbols`, and `contract_path` per inventory entry. **Paths** are mechanically existence-checked at current HEAD; **symbols** are statically resolved in contract or production path sources (no dynamic imports). Physical SQLite/KV/Document backends may host multiple families; contracts remain distinct.

---

## 8. Writer / reader matrix

See `writers`, `readers`, `restore_consumers` per inventory entry.

---

## 9. Recovery / restore matrix

| Concern | P0 lock |
|---|---|
| Checkpoint | Historical state; resume validates tenant/lineage/terminal/authority narrowing |
| Authority source | **Current admitted authority** narrowed by valid historical checkpoint bounds |
| Identity | Preserved per family; checkpoint stores facts, does not mint IDs |
| Restore command source | Not runtime events / run trace read models |

---

## 10. Authority preservation matrix

**Persisted families that mint new Execution/Governance authority:** **0** (mechanical gate `test_sx_p0_no_family_mints_authority`).

Checkpoint (SX-F01): `authority_role = HISTORICAL_EVIDENCE_ONLY`; resume uses `narrow_resume_execution_authority` / `validate_checkpoint_resume_authority` (`checkpoint_resume_validation.py`).

Human decisions (SX-F12): evidence only — resume still requires governance admission.

---

## 11. Identity preservation matrix

| Identity | Primary owners (recovery-relevant) |
|---|---|
| tenant_id | Checkpoints, terminal, lineage, decision keys, agent checkpoints |
| task_id | TaskCheckpoint, terminal |
| run_id / attempt_id | AttemptLifecycleStore, lineage, agent checkpoints |
| execution tree root | SX-F02 component; validated vs lineage on resume |

**Mint identity from durable store alone:** none classified as `MINT_IDENTITY`.

---

## 12. Tenant Isolation Audit

Per-family `tenant_audit_disposition` in inventory (PASS / BLOCKED / N/A-WITH-EVIDENCE). Global FRZ-TEN closure remains **TENANT-X**.

---

## 13. Atomicity / CAS / fencing matrix

| Family | Mechanism (current-HEAD existence) |
|---|---|
| SX-F01 | `expected_revision` on TaskCheckpointPersistence.save |
| SX-F03 | `snapshot_revision` + `expected_revision` CAS (SQLiteDecisionCheckpointPersistence) |
| SX-F04 | generation / durable provider CAS |
| SX-F13 | SchedulerLedger claim_action lease |
| SX-F14 | Agent checkpoint expected_revision CAS |

---

## 14. Stale-state matrix

Documented per `stale_state_rule` in inventory (revision CAS, terminal denial, lineage mismatch, StaleDecisionCheckpointWriteError).

---

## 15. Projection vs truth matrix

| Classification | Families |
|---|---|
| CANONICAL_TRUTH | F01, F03–F07, F09–F14 |
| DURABLE_COMPONENT | F02 |
| READ_MODEL | F08 |
| COORDINATION_ONLY | F15 |
| CONFIGURATION_INPUT | (none at durable truth — RunBudget config vs ledger distinguished in F09) |

---

## 16. Backup / restore responsibility matrix

Every durable family has `backup_restore_responsibility` ∈ allowed enum (no TBD). Predominantly **BACKEND/OPERATOR … WITH EXPLICIT PLATFORM CONSISTENCY REQUIREMENTS**; F08 **NOT DURABLE / REBUILDABLE PROJECTION**.

Full certification → **STATE-X-R4**.

---

## 17. Corrupt / partial state handling matrix

Summarized in `corrupt_partial_handling` per inventory entry — inventory only; R1–R4 certify/fix.

---

## 18. Fork / retry / resume semantics

`fork_retry_resume_notes` per family. P0 lock: **resume ≠ retry ≠ fork** — unified behavior not assumed; undefined forks → tracked blockers in R2/R3.

---

## 19. External-effect uncertainty matrix

SX-F10, F11, F15 reference FRZ-REC-07; recovery must not treat unknown external outcome as safe retry without idempotency/terminal proof.

---

## 20. Known current-head blockers

| ID | Classification | Owner |
|---|---|---|
| SX-B01 | TRACKED FREEZE DEBT (Q2-D1) | STATE-X-R1 |
| SX-B02 | IN-SCOPE BLOCKER (task_snapshot / historical authority) | STATE-X-R1 |
| SX-B03 | ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED (R1-SQLITE-ENV-01) | STATE-X-R4 / QUAL-X / PROD-Q |

**SX-B02 P0 lock:** **CURRENT CANON REQUIRES SNAPSHOT** — durable SQLite `task_snapshot_json NOT NULL`; authority parse uses `Task.model_validate(checkpoint.task_snapshot)` without empty-snapshot shortcut (CTRL-X revert preserved).

**SX-B01 direction:** composition must depend on `ExecutionTerminalPersistenceCapability`, not accidental `TaskCheckpointPersistence` widening.

---

## 21. Historical evidence disposition

| Item | Disposition |
|---|---|
| W3-A SQLite decision UPSERT without CAS | **SUPERSEDED WITH CURRENT-HEAD EVIDENCE** — `SQLiteDecisionCheckpointPersistence.save` uses `expected_revision` + `snapshot_revision` CAS |
| NPSC-5E / SCHED-01 / HARNESS-FINAL / GOV-X2 / EBH-4 tenant recovery | **Evidence candidates** — not STATE-X PASS |
| GOV-X2 Q2-D1 | **TRACKED FREEZE DEBT** → SX-B01 |

---

## 22. Child roadmap P0→R1→R2→R3→R4→FINAL

```text
STATE-X-P0  (CURRENT — baseline lock)
STATE-X-R1  Task checkpoint / resume / terminal capability & authority closure
STATE-X-R2  Decision / attempt / lineage concurrency & replay
STATE-X-R3  Auxiliary durable state consistency
STATE-X-R4  Backup / restore / corruption / DR certification
STATE-X-FINAL  Current-HEAD parent replay & freeze evidence
```

---

## 23. Enterprise audit matrix

| Check | Result |
|---|---|
| Production changes | 0 |
| Contract changes | 0 |
| Exactly-one semantic owner per SX-F* | Inventoried |
| FRZ promoted to PASS | 0 |

---

## 24. Exact test commands / results

```bash
uv run pytest -p no:xdist tests/qualification/state_x/test_state_x_p0_baseline.py -q --tb=short
```

Record result in commit message / audit trail after local run.

---

## 25. Findings

| Finding | Classification |
|---|---|
| SX-B01 Q2-D1 typing seam | TRACKED FREEZE DEBT |
| SX-B02 task_snapshot authority | IN-SCOPE BLOCKER |
| SX-B03 R1-SQLITE-ENV-01 | ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED |
| W3-A decision CAS gap | SUPERSEDED WITH CURRENT-HEAD EVIDENCE |

**unclassified = 0**

---

## 26. Recommendation

- **STATE-X-P0** = READY FOR AUDIT
- **STATE-X** = CURRENT / **BLOCKED PENDING MANDATORY CHILDREN**
- **STATE-X-R1** = NEXT / NOT ENTERED
- **TRACE-X** = NOT ENTERED

---

## Before / after graph

### BEFORE P0

```text
many previously qualified state mechanisms
+ historical recovery documents
+ future debts
+ no current STATE-X global truth map
```

### AFTER P0

```text
STATE-X canonical inventory
    ├── state family (SX-F01..15)
    ├── exactly-one semantic owner
    ├── canonical contract
    ├── physical provider(s)
    ├── writers/readers
    ├── concurrency semantics
    ├── tenant semantics
    ├── recovery semantics
    ├── authority/identity role
    ├── backup responsibility
    └── exact blocker owner
```

---

**Mandatory:** Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
