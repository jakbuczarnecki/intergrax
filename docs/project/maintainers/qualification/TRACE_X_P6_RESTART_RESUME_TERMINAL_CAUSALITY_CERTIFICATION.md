# TRACE-X-P6 — Restart/Resume Continuity & Terminal Outcome Causality Certification

| Field | Value |
|---|---|
| **Stage** | `TRACE-X-P6` / **`TRACE-X-P6-R1`** (fail-closed correction) |
| **Parent** | `TRACE-X` |
| **START_HEAD** | `e2deaee3efd7ee4e7d43414be3e76fe024a85408` |
| **Audited implementation (initial)** | `51b1f26a7a473658f6f83d0d8dbfd66ab62f4516` |
| **FINAL_COMMIT** | `eee314330b926692291520af6c11f70faf878858` |
| **Independent verdict (initial P6)** | **REJECTED / BLOCKED** (permissive classification + non-mechanical owner matrix) |
| **R1 blockers closed** | `P6-CLOSED-WORLD-PERMISSIVE-RESTART-CLASSIFICATION-01`, `P6-CLOSED-WORLD-PERMISSIVE-TERMINAL-CLASSIFICATION-02`, `P6-SEMANTIC-OWNER-MATRIX-NON-MECHANICAL-03` |
| **FRZ-TRC-09** | **OPEN / PASS CANDIDATE** (Cursor: **READY FOR PASS AUDIT** — no self-PASS) |
| **FRZ-TRC-10** | **OPEN / PASS CANDIDATE** (Cursor: **READY FOR PASS AUDIT** — no self-PASS) |
| **TRACE-X-P6-R1** | **READY FOR AUDIT** |
| **TRACE-X-P6** | **BLOCKED ON R1 AUDIT** |
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

## 4. Semantic owner matrix (mechanical)

`discover_semantic_owners(concern)` in `_trace_x_p6_semantic_owner_discovery.py` — gated by `test_txp6_q10`:

| Concern | Expected owners |
|---|---:|
| Execution identity owner | 1 |
| Run identity owner | 1 |
| Attempt identity owner | 1 |
| checkpoint persistence owner | 1 |
| resumability decision owner | 1 |
| resume coordination owner | 1 |
| resume admission tenant/identity validation owner | 2 |
| retry policy owner | 1 |
| retry orchestration owner | 1 |
| terminal state truth owner | 1 |
| terminal RuntimeEvent/evidence owner | 1 |
| failure reconstruction owner | 1 |
| parent-child causality owner | 1 |

Resume checkpoint persistence vs coordination vs admission validation are **distinct** concerns (not collapsed into a single prose owner).

## 5. Adversarial bundle P6-A … P6-H

Unchanged matrix; replayed in R1 Pass1 session (`.tmp/session/trace-x-p6/pass1_observed_nodeids.json`, `TRACE_X_P6_PASS1=1`).

## 6. Tenant isolation audit (P6-local)

**PASS** — P6-F, P6-G + supporting STATE-X / TRACE-X negatives (no global FRZ-TEN promotion).

## 7. Tests (R1 Cursor session)

| Batch | Result |
|---|---|
| P6 closed-world gates (`test_trace_x_p6_closed_world_gates.py`) | **14 passed** |
| P6 adversarial bundle + PASS1 manifest | **3 passed** |
| P6-A … P6-H nodeids (targeted PASS1 batch) | **8 passed** |
| Tenant negatives (P6-F, P6-G) | **PASS** |

Command: `pytest -p no:xdist` on qualification gates + targeted P6-A…H nodeids.

## 8. Pyright

**Production delta = 0** (qualification/support modules only).

## 9. Post-step enterprise discovery (R1)

| Item | Finding |
|---|---|
| New current blockers | none beyond **BLOCKED ON R1 AUDIT** |
| New future mandatory debt | none from qualification-only R1 |
| New candidate roadmap stages | none |
| FRZ coverage gaps | **FRZ-TRC-09** / **FRZ-TRC-10** await independent PASS |
| Ownership/boundary concerns | none exposed by fail-closed reclassification |
| Roadmap amendment required | no |
