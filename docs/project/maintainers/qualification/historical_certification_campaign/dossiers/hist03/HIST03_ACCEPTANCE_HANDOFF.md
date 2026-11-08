# HIST-03 - acceptance preparation handoff (NOT an audit handoff for an implementation)

Baseline `ba07ebaa34d04633bd60510f68c18346b1ef160d`, branch `hist-cert/hist03-controls` (worktree clean; **no implementation, no commit, no push**). Implementation SHA: none (no repair authorized/needed so far). Frozen audited scope: n/a until a corrective batch exists.

## 1. Promotion statement
**`promotion_allowed_now` = false for ALL 40 rows** (machine-checked in `HIST03_FRZ_40_PER_ID_MATRIX.json` by assertion when generated; every row `checklist_status_at_baseline` = OPEN). No row is PASS, none is promotable by a worker narrative, a delegation (no delegation can promote an FRZ row; IIAC covers only EVOL-X/ENT-AUDIT-X; Amendment 01 not cited), or by scoped historical evidence. Acceptance rule unmet for every row: exact SHA + criterion mapping + static artifact + executed gate output + independent auditor sign-off + operator promotion.

## 2. Routes (from the matrix `evidence_route`)
**A. Pinned exact-SHA replay, then independent audit (evidence already exists; 16 rows):**
FRZ-OBS-01, 02, 03 (W5 matrix + CX-06/07 nodes); FRZ-OBS-04, 05, 06, 07 (same plus docs reconciliation to TRACE-X evidence first - P-02); FRZ-REL-04, 06, 07, 08, 09, 10, 11 (HARNESS-W4 nodes T1-T11; state strict-production-only scope; W2-C warnings + legacy invoker debt need explicit freeze treatment, F-010); FRZ-PRD-06 (hosting engine + LKW readiness tests), FRZ-PRD-07 (D4, P1, P2-N02 nodes).
**B. Independent audit only after a documentation record is authored (4 rows):** FRZ-SEC-07 (consolidated threat-model claim bound), FRZ-SEC-09, FRZ-SEC-10 (data-responsibility record, F-003), FRZ-PRD-12 (recovery responsibility/procedure). Replay of cited gates optional.
**C. Qualification work first, then replay + audit (16 rows):** FRZ-SEC-01, 02, 03, 04, 05, 06, 08; FRZ-REL-03, 05; FRZ-PRD-03, 04, 05, 08, 09, 10, 11.
**D. Mapping + replay + audit, blocked/conditional (2 rows):** FRZ-REL-01 (needs TOOL-LIFE-X C-1/C-2 correctives and re-audit; TOOL-LIFE-X is NOT accepted); FRZ-REL-02 (ADR-CTRL-X-001 `p0_safety_8` file passed on the pin; REL-02 mapping of retry-ownership gates still missing; the same ADR's `test_sec_ent.py` failures are SEC-02/F-006).
**E. Nothing now - Path B live (2 rows):** FRZ-PRD-01, FRZ-PRD-02 (PROD-Q live gate / SCENARIO-GATE).

## 2a. Blockers found by the run (must be cleared before route-A/C audit)
- F-015 (HIGH, IMPLEMENTATION_GAP): P2-C HTTP admission red in isolation; SEC-04, PRD-08, PRD-09 evidence downgraded; corrective is product code (not done; owner/ticket needed).
- F-006: `test_sec_ent.py` 7 red (test-side) -> SEC-02 qualification cannot start from green; F-016: W5 b2/c/d need process isolation until fixed (OBS route-A replay must run them per file).
- REL-02: `test_p0_safety_8` passed on the pin; mapping + audit still required.

## 3. Pinned-checkout requests (`needs_master`)
0. DONE on shared pin (mutable venv, overlap caveats, see audit S9). Still requested: a clean-room run for audit evidence.
1. One clean checkout at a pinned SHA to run, as single `uv run pytest` batches: W4 node set (`tests/unit/runtime/architecture/test_harness_w4_r1_*`, `tests/unit/runtime/nexus/tools/test_harness_w4_r1_*` plus W2/W4 suites per W4 doc T1-T11), W5 set (`tests/unit/runtime/{events,observability}/*enterprise_scale_resilience_w5*`), `tests/qualification/control_planes/ctrl_x/`, `tests/qualification/prod_q/test_prod_q_p0_fast.py`, D4/P1/P2 qualification modules, `tests/unit/hosting/engine/test_health_readiness.py`, `applications/local_workspace_application/tests/hosting/test_hosted_readiness.py`, and the two ADR-CTRL-X-001 files.
2. Operator-level rulings: P-03 (SEC-06 split note), P-04 (PRD-04 scope), SEC-05 scope meaning, Decision-03/D03-A infra-vs-architecture split, mandatory-vs-waived for AUTH-08 HTTP / G-08 / IG-F01 / DIM16 S2b (F-012).

## 4. Candidate dispositions prepared for independent acceptance
Only the 16 route-A rows are "ready for an auditor to attempt", and only after the replay. Preferred auditor input: this matrix + `HIST03_FINDINGS_AND_RESIDUALS.json` + the cited evidence files. The auditor must be a fresh session (not the implementer, not the original closer); MASTER starts it.

## 5. Not handed off / explicit non-claims
No production readiness; no Dimension B closure; no live activation; no PROD-Q parent change; no weakening of any gate; PARTIAL/CONDITIONAL records (P2-C partial log, AUTH-08 PARTIAL, TOOL-LIFE-X CONDITIONAL) are not rewritten.
