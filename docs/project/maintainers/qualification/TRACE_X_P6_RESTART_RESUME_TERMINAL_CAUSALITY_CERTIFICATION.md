# TRACE-X-P6 — Restart/Resume Continuity & Terminal Outcome Causality Certification

| Field | Value |
|---|---|
| **Stage** | `TRACE-X-P6` / **`TRACE-X-P6-R1`** / **`TRACE-X-P6-R1-R1`** / **`TRACE-X-P6-R1-R1-R1`** (resume/retry owner taxonomy) |
| **Parent** | `TRACE-X` |
| **START_HEAD (P6)** | `e2deaee3efd7ee4e7d43414be3e76fe024a85408` |
| **START_HEAD (R1-R1)** | `21f4aa38c68221bff56f893a8770d030e1c2ccf6` |
| **START_HEAD (R1-R1-R1)** | `f45cfe709b0f97eaf1afbb641bbc91002d1f4e34` |
| **Audited implementation (initial)** | `51b1f26a7a473658f6f83d0d8dbfd66ab62f4516` |
| **Audited R1 qualification tip** | `96e70c354a5805953c85031c707f115e8dcd1129` |
| **Audited R1 qualification (independent)** | `0ce1469f440f58938ceeeb0f7c9be347091206be` — **BLOCKED** (`P6-SEMANTIC-OWNER-DUPLICATE-DISCOVERY-03R`) |
| **FINAL_COMMIT (R1-R1)** | `cd1f6fbbb0991c02dba95e07433fa999e61260cf` |
| **Qualification mechanics (R1-R1, accepted)** | `cd1f6fbbb0991c02dba95e07433fa999e61260cf` — independent discovery `discovered == expected`; duplicate sensitivity **03R** |
| **FINAL_COMMIT (R1-R1-R1)** | `86db8f11d98637dda5829d04623a357ac37bc095` |
| **Independent verdict (initial P6)** | **REJECTED / BLOCKED** (permissive classification + non-mechanical owner matrix) |
| **R1 blockers closed** | `P6-CLOSED-WORLD-PERMISSIVE-RESTART-CLASSIFICATION-01`, `P6-CLOSED-WORLD-PERMISSIVE-TERMINAL-CLASSIFICATION-02`, `P6-SEMANTIC-OWNER-MATRIX-NON-MECHANICAL-03` (partially remediated in R1; residual **03R** closed in R1-R1) |
| **R1-R1 blockers closed** | `P6-SEMANTIC-OWNER-DUPLICATE-DISCOVERY-03R` |
| **R1-R1-R1 blockers closed** | `P6-RESUME-OWNER-TAXONOMY-04`, `P6-RETRY-OWNER-TAXONOMY-05` |
| **R1-R1-R1 residual blockers** | none (taxonomy decomposition only; no production semantic collision exposed) |
| **FRZ-TRC-09** | **OPEN / PASS CANDIDATE** (no self-PASS) |
| **FRZ-TRC-10** | **OPEN / PASS CANDIDATE** (no self-PASS) |
| **TRACE-X-P6-R1-R1-R1** | **READY FOR AUDIT** |
| **TRACE-X-P6-R1-R1** | **BLOCKED ON R1-R1-R1 AUDIT** |
| **TRACE-X-P6-R1** | **BLOCKED ON R1-R1-R1 AUDIT** |
| **TRACE-X-P6** | **BLOCKED** |
| **TRACE-X** | **CURRENT** |
| **CONFIG-X** | **NOT ENTERED** |
| **Production delta** | **0** |

## 0. Historical initial P6 evidence (preserved)

Initial qualification on `51b1f26a7a473658f6f83d0d8dbfd66ab62f4516` reported closed-world parity with **permissive defaults**:

- restart/resume: default `A_CANONICAL_RESUME` when no rule matched;
- terminal producers: default `CANONICAL_TERMINAL_DELEGATE` + filename substring heuristics;
- semantic owner matrix: prose owners with `assert count == 1` only.

Independent audit **REJECTED** that bundle; R1 corrects classification and owner evidence without production architecture changes.

## 1. Certification scope (FRZ-TRC-09 / FRZ-TRC-10)

Prove on R1 HEAD:

- **FRZ-TRC-09:** supported restart/resume/recovery preserves attributable causal continuity.
- **FRZ-TRC-10:** terminal outcomes have exactly-one semantic truth owner and non-authoritative projections are classified.

**Discovery model:** marker scan identifies **candidate** surfaces; classification must **positively** establish semantics. Unknown candidate ⇒ qualification failure (fail-closed).

## 2. Closed-world inventories (mechanical parity)

| Inventory | Count | G / UNCLEAR | F / forbidden bypass |
|---|---:|---:|---:|
| Restart/resume/recovery modules | **159** | **0** | **0** |
| Terminal outcome producer modules | **50** | **0** | **0** |

**Restart/resume classification (159):** A=144 · B=2 · C=2 · D=1 · E=10 · F=0 · G=0.

**Terminal producer roles (50):** `CANONICAL_TERMINAL_TRUTH`=1 · `CANONICAL_TERMINAL_DELEGATE`=4 · `COMPATIBILITY_ADAPTER`=43 · `DIAGNOSTIC_PROJECTION`=2 · `FORBIDDEN_BYPASS`=0 · `UNCLEAR`=0.

**SSOT:** `RESTART_RESUME_REGISTRY` / `TERMINAL_PRODUCER_REGISTRY` (rebuilt at import; fail-closed classifiers).  
**Gates:** `test_txp6_q02` … `test_txp6_q09`, `test_txp6_q04`.

## 3. Fail-closed classification (R1)

| Surface | Forbidden default | R1 default |
|---|---|---|
| Restart/resume | `A_CANONICAL_RESUME` | `G_UNCLEAR` |
| Terminal producer | `CANONICAL_TERMINAL_DELEGATE` | `UNCLEAR` |

Terminal roles use **module evidence** (`path` + AST/marker facts via `_trace_x_p6_module_evidence.py`), not filename substring inference.

**Negative sensitivity:** `test_txp6_q11`–`test_txp6_q14` (registry unknown + classifier fail-closed for synthetic paths).

## 4. Semantic owner matrix (mechanical duplicate discovery, R1-R1-R1)

**Model:** `discover_owner_candidates(concern)` scans production modules (AST / responsibility-bearing symbols).  
`P6_CANONICAL_OWNER_EXPECTATIONS` is the **expectation only** — gate `compare_semantic_owner_gate` asserts `discovered == expected` (`test_txp6_q10`).  
**Expected ≠ discovery implementation** (no direct return of expected paths).

| Concern | Expected / discovered (modules) |
|---|---|
| Execution identity owner | `intergrax/contracts/execution_identity.py` |
| Run identity owner | `intergrax/runtime/execution/identity_authority.py` |
| Attempt identity owner | `intergrax/runtime/execution/attempt_lifecycle/service.py` |
| checkpoint persistence semantic contract owner | `intergrax/runtime/long_running/persistence_contract.py` (`TaskCheckpointPersistence` — semantic port; SQLite adapters are implementations, not duplicate semantic owners) |
| resumability decision owner | `intergrax/runtime/cancellation/resume_admission.py` |
| scheduled resume trigger owner | `intergrax/runtime/long_running/scheduler.py` (`LongRunningScheduler` — poll/claim/trigger; not checkpoint restoration truth) |
| resume restoration / recovery coordination owner | `intergrax/runtime/long_running/coordinator.py` (`restore_if_resuming`, `recovery_admission_request_for_checkpoint`, `admit_task_resume_recovery_handoff`) |
| resume admission validation owner | `resume_admission.py` + `reentry_admission.py` (distinct admission boundaries) |
| execution retry eligibility policy owner | `intergrax/runtime/execution/retry/policy.py` (`evaluate_execution_retry_eligibility` — MAY retry only) |
| execution-attempt retry authority owner | `intergrax/runtime/execution/retry/service.py` (`ExecutionAttemptRetryService` — same Run → new Attempt) |
| run/graph retry scheduling facade owner | `intergrax/runtime/nexus/retry/coordinator.py` (`RetryCoordinator` — `should_retry_run`, `RETRY_SCHEDULED` evidence; not execution-attempt retry authority) |
| terminal state truth owner | `intergrax/runtime/execution/execution_terminal/service.py` |
| terminal RuntimeEvent/evidence owner | `intergrax/runtime/events/trace_bridge.py` |
| failure reconstruction owner | `intergrax/runtime/observability/reconstruction/execution_reconstruction.py` |
| parent-child causality owner | `intergrax/contracts/execution_lineage.py` |

**Negative sensitivity (owner gate):** `test_txp6_q15` terminal truth duplicate; `test_txp6_q16` scheduled resume + restoration coordination duplicates; `test_txp6_q17` execution-attempt retry authority duplicate (synthetic second candidate ⇒ gate FAIL).

**R1-R1-R1 taxonomy note:** prior `resume coordination owner` and generic `retry orchestration owner` were too broad; responsibilities are split per scheduler vs coordinator vs execution-attempt vs Nexus facade without re-opening accepted P6-A…H or fail-closed classification evidence.

## 5. Adversarial bundle P6-A … P6-H

Unchanged matrix; replayed in R1 Pass1 session (`.tmp/session/trace-x-p6/pass1_observed_nodeids.json`, `TRACE_X_P6_PASS1=1`).

## 6. Tenant isolation audit (P6-local)

**PASS** — P6-F, P6-G + supporting STATE-X / TRACE-X negatives (no global FRZ-TEN promotion).

## 7. Tests (R1-R1-R1 Cursor session)

| Batch | Result |
|---|---|
| P6 closed-world gates (`test_trace_x_p6_closed_world_gates.py`) | **17 passed** |
| P6 adversarial bundle registry | **3 passed** |
| P6-A … P6-H nodeids (targeted batch) | **8 passed** |
| **Combined qualification batch** | **31 passed** |

Command: `pytest -p no:xdist` on closed-world gates + adversarial bundle + P6-A…H nodeids (see `.tmp/session/trace-x-p6-r1-r1-r1/pytest.log`).

## 8. Pyright

**Production delta = 0** (qualification/support modules only).

## 9. Post-step enterprise discovery (R1-R1-R1)

| Item | Finding |
|---|---|
| New current blockers | none beyond **BLOCKED ON R1-R1-R1 AUDIT** |
| New future mandatory debt | none from qualification-only R1 |
| New candidate roadmap stages | none |
| FRZ coverage gaps | **FRZ-TRC-09** / **FRZ-TRC-10** await independent PASS |
| Ownership/boundary concerns | none exposed by fail-closed reclassification |
| Roadmap amendment required | no |
