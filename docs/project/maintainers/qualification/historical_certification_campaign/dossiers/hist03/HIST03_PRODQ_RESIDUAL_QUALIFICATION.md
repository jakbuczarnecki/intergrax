# HIST-03 - PROD-Q residual qualification (review of scoped qualifications)

Baseline `ba07ebaa34d04633bd60510f68c18346b1ef160d`. Basis: document + static inventory; **nothing re-executed**. No PASS, no production readiness, no live activation, no FRZ promotion is claimed or implied.

## 1. Formal PROD-Q standing (cited, not altered)
- Operator closure 2026-11-02 (`PROD_Q_OPERATOR_SCOPE_CLOSURE_2026_11_02.md` L6-39): "administrative/scoped operator closure, not certification that all PROD-Q mandatory children passed, not a production readiness attestation". Formal disposition: **PROD-Q - NOT FULLY QUALIFIED / LIVE INTEGRATIONS BLOCKED**. Production use of affected live integrations remains BLOCKED.
- Explicitly OPEN: PROD-Q-P1 (live provider/sandbox), PROD-Q-R1 and PROD-Q-R1-R7-Q mandatory remainder, DIM6, DIM17.

## 2. Accepted scoped waves (evidence contribution only; each "not global PASS")
| Wave | Record | Scope actually proven | Limits recorded in its own document |
|---|---|---|---|
| PROD-Q-P0 | CLOSED OPERATOR ACCEPTED/SCOPED (2026-10-21) | closed-world inventory + mechanical FRZ reconciliation module (`tests/qualification/prod_q/_p0_frz_reconciliation.py`) | module hard-pins PRD-04 BLOCKER, PRD-01/08 IN_SCOPE_BLOCKER, PRD-02/SEC-06 provider-evidence-required |
| D1/D1b/D1-R6/D2/D3 | CLOSED OPERATOR ACCEPTED/SCOPED A1 @5dfefe41 | colocated adapter, admission bridge, lifecycle composition, restart reconciliation | A1 scope; global FRZ OPEN |
| D4 (+R1,R2) | CLOSED independently accepted (operator) @0dd51fe | strict GCA launcher, legacy entry fail-closed, startup-failure cleanup | does not close R7/PROD-Q |
| P1 (R7-Q) | CLOSED scoped static reachability @aee2e543 (23/23) | bounded AST reachability | not full dynamic/HTTP bypass proof |
| P2 | CLOSED exact-SHA acceptance @8e43b1604 | strict-vs-lab, integrated chain | **P2-C HTTP 6/6 user-reported, partial log**; P2-B not replayed at HEAD |
| P3 | CLOSED operator accepted @28c9bc40 / P3 B6 2026-10-22 | tenant/caller chain, DENY/CAS, stale READY, drain (static), failed prepare, repeat dispatch (composition) | AUTH-08 HTTP PARTIAL; G-08 async ContextVar NOT PROVEN; IG-F01 HTTP re-entry NOT PROVEN; no waiver recorded |
| E.1 | CLOSED scoped D0-D4 regression (21 modules / 209 tests) | batch replay across distinct SHAs | no monolithic pytest/CI artifact |
| Package P / §C / Package D | CLOSED scoped (2026-10-21 / 2026-10-17) | DIM2,3,7,8,9,10,11,13,14,15,18,19,21 ; DIM20 ; DIM16 | DIM16 F02-09 S2b NOT PROVEN; DIM6, DIM17 excluded |
| DISPATCH-Q1 | ACCEPTED scoped operator | PIS + dispatch gate | HTTP E2E, business exactly-once excluded |

(The statement in the P3 decision package that E.1 is "NOT EXECUTED" is superseded by the E.1 ledger row - see F-008.)

## 2a. Reproduction result (pin ba07ebaa3, 2026-11-08)
P0/P1/P2-B/D4/closure-regression modules pass; **P2-C HTTP admission is red in isolation** (6/7): SKILL-X Batch C (`d1040f485`) made the host runtime pass both kv_store and document_store into a wiring function that rejects both - the accepted P2-C 6/6 (user-reported, partial log) no longer holds at baseline (F-015). SEC-04, PRD-08, PRD-09 downgraded accordingly.

## 3. Unqualified paths (HTTP / provider / live environment / production graph)
1. **Providers (E2B, Modal, Daytona)** - EXTERNAL_PROVIDER_EVIDENCE_REQUIRED; Path B live; FRZ-PRD-02 and FRZ-PRD-01 (primary Path B); FRZ-SEC-06 live dependency only (secondary; architecture part stays HIST-03 gap).
2. **Operator production profiles** - DIM6 (OP-01 mutation policy), DIM17 (OP-02/03): live-scoped parts of FRZ-PRD-03/08/SEC-04.
3. **HTTP paths** - AUTH-08 stale-URL E2E (PARTIAL), IG-F01 same-`invocation_id` re-entry, async ASGI ContextVar isolation (G-08): architecture-scope relevance undecided (no operator waiver) -> F-012.
4. **Production graph / multi-instance** - Kubernetes target ratified (Decision-03) but "NO BACKEND OR GLOBAL PIS AUTHORIZED"; LEGACY-HISTORY-TRUST-ANCHOR-GATE "STOP - ARCHITECTURE DECISION REQUIRED"; D03-A Spanner+Azure witness NOT ratified. Infrastructure-dependent -> Path B candidates but not listed in checklist L66 (ruling) -> F-013.
5. **Startup via HTTP/GCA reproduction (R1-SQLITE-ENV-01)** - environment/test issue, opt-in reproduction never recorded as executed in-repo; blocker for FRZ-PRD-04 in the mechanical record -> F-004.
6. **Capacity under load** - DIST-SCALE-X WL-7 4 h soak NOT PROVEN (deferred non-blocking); relevant to FRZ-PRD-11.

## 4. Architecture blockers vs approved Path B deferrals
- **Path B deferral, canonical** (checklist L66): FRZ-PRD-01, FRZ-PRD-02, live parts of PRD-03/PRD-08/SEC-04. -> PRIMARY PATH_B_LIVE_DEFERRED only for PRD-01 and PRD-02; secondary cause for the live parts of PRD-03/PRD-08/SEC-04.
- **FRZ-SEC-06 (MASTER ruling):** not in L66, Path B is not widened: primary = architecture-side plugin-trust QUALIFICATION_GAP; provider live qualification (PROD-Q-P1) recorded as secondary cause; stays OPEN. Proposed note/split = Operator-level SSOT proposal (candidate patch P-03).
- **Live-dependent but not covered by L66 (rule gap, not decided here):** PRD-04, parts of SEC-01/REL-03/REL-05/PRD-08.
- **Architecture-scoped gaps with no live excuse:** SEC-02, SEC-03, SEC-08, SEC-09, SEC-10, REL-01, REL-02, PRD-06, PRD-12, SEC-07 (no or weak row-specific evidence) and PRD-05/09/10/11, SEC-04 (arch part), SEC-05.
- **Possible true implementation/test defects (unverified):** ADR-CTRL-X-001 listed failing tests (`test_sec_ent.py`, `test_p0_safety_8_retry_redelivery_authorization.py`); TOOL-LIFE-X C-1/C-2 red tests (HIST-01/02 owned, cross-referenced).
- No live-provider failure has been converted into architecture PASS.

## 5. Dimension B ownership
Remains with PROD-Q live gate and SCENARIO-GATE production admission. HIST-03 closes nothing in Dimension B.
