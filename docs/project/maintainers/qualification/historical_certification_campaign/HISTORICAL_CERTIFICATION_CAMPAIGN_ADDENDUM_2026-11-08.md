# Historical certification campaign - addendum of 2026-11-08 (after PR 408)

Scope: what happened after the final package (PR 408) was merged, up to PR 416 (`origin/development` = `51cb5f336`). The final report, the master closure matrix, the 85-row disposition matrix and the other files of this directory remain as published at PR 408; where this addendum differs from them, this addendum is the later statement. Language: English only.

**Campaign result: still `PARTIAL / EXACT BLOCKERS`.** Independent closures: 0. Promotions: 0. The FRZ checklist is unchanged (all 85 historical rows OPEN). Nothing below promotes a row.

## 1. Campaign pull requests after PR 408 (merged by the Operator, as reported in the orchestrator chat; the ENT-AUDIT-X PRs 409, 411 and 412 were merged in between and are not part of this campaign)

| PR | Merge commit | Content |
|---|---|---|
| 410 | `c169852bf` | Operator decisions DR-1, 2, 3, 4, 6, 8 and DR-9 (partly) certified by the designated session `cert-w0`; independent review `rev-w0`; Q-11 report. Merged at head `de47bd2fe`, before the later pushed tip was reflected (the confirmation record was left out; see PR 413) |
| 413 | `70cceae21` | The explicit Operator confirmation record and section 12 that PR 410 missed |
| 414 | `6734971cf` | Rulings W1-R1, W1-R2, W1-R3 and the sequence W1-PLAN certified by `cert-w1`; independent review `rev-w1` |
| 415 | `af323c005` | Ticket T-IGX-031-A and its plan with the delegated plan approval (`cert-031a`) |
| 416 | `51cb5f336` | T-IGX-031-A implementation (tests only), the amended plan (S12), delegated approval of the amended plan (`cert-031a2`), two independent audits |

## 2. Operator decisions: all seven formerly open decisions now have an answer on record (scope rulings; limits below)

The answers are certified by designated sessions in the Operator's name (IIAC Amendment 01 R-1..R-6; assurance `process_enforced`; the delegation records are transcriptions by the orchestrator, not signed by the Operator). An explicit but general Operator confirmation of the W0 decisions is on record (the reply "I confirm and I consent", transcribed by the orchestrator; the reading of its scope is the orchestrator's; `delegations/DELEG-HIST-W0-CONFIRMATION-2026-11-08.md`). The W1 rulings rest on the Operator's acceptance recorded in `delegations/DELEG-HIST-W1-2026-11-08.md`.

| Decision | Answer on record | Where |
|---|---|---|
| DR-1 (OD-HIST04-A) EVOL-X delegated signatures | ratified from the original timestamps; no earlier record rewritten | `decisions/EVOL_X_DELEGATED_SIGNATURES_RATIFICATION.md`, `decisions/OPERATOR_DECISION_RECORDS_W0.md` |
| DR-2 (OD-HIST04-B) Amendment 01 status | option (c): in force as process-enforced policy; T-IGX-021-B/C a separate stage (not started; the tickets do not exist) | W0 records, status note in the amendment |
| DR-3 (OD-HIST04-C) TOOL-LIFE-X label | option A: annotation only, the status cell is unchanged | roadmap row, W0 records |
| DR-4 (OD-HIST01-B) FRZ-REG promotion scope | HIST-01 split; W1-R3 adds: REG-02/03/06/09 only after the ENT-AUDIT-X Population E rollup | W0 and W1 records |
| DR-6 (OD-HIST03-B) PROD-Q P2-C | option 1: root cause first. Q-11 CONFIRMED a code defect (SKILL-X Batch C `d1040f485`); a ticket plan exists (section 4) | W0 records, `audits/q11/Q11_REPORT.md` |
| DR-8 (OD-HIST02-A) Tier-0 -> Tier-3 import | option 2: a ticket to move the composition; a plan exists (section 4) | W0 records |
| DR-9 (OD-HIST03-A) PROD-Q / SEC rulings | P-03 split, P-04 architecture-scoped, no waivers (W0); SEC-05 = contract + reference substrate with the per-host limitation stated, Decision-03 / D03-A split into (a) contract/fail-closed semantics and (b) infrastructure/live (W1-R1, W1-R2). Rows stay OPEN. Checklist L66 is unchanged | W0 and W1 records |

Known limits of those answers (also in the records): W1-R2 does not exempt the (b) parts of SEC-01, REL-03, REL-05 from the checklist rule (`OPEN = 0` for architecture-scoped rows); SEC-05 keeps the tension "architecture-scoped" vs "PROD-Q still required"; "per-host isolation" is not defined in the repository; REL-03, REL-05 and PRD-08 do not name Decision-03 in their row text.

## 3. Stage T-IGX-031-A (tests only) - merged on `development`, PASS_SCOPED by two independent audits; ticket state `ready_for_audit`

- Problem: HIST03-F-006 (`test_sec_ent.py`, 7 red tests) and HIST03-F-016 (registry pollution in W5 b2/c/d).
- Result: `test_sec_ent.py` 14 passed; W5 20 passed; the four files in one process in both orders 34 passed. Negative probes behave as predicted. The 27-path run (Operator-authorized, one run) shows 0 "already registered"; the 6 remaining failures are the P2-C tests of F-015. The pass/fail counts come from pytest's own summary line: the exit code of that run was NOT captured, because the process did not exit after the summary (the authorization listed a captured exit code as a condition).
- Evidence: audits `aud-tests` (PASS_SCOPED at `b28398026`) and `aud-s12` (PASS_SCOPED at `47c29b4c3`, behaviour-neutral lint normalization S12, AST-equivalent, ruff clean); plan reviews `rev-plan`, `rev-plan2`; approvals `cert-031a`, `cert-031a2`.
- Statement allowed: F-016 is resolved **for the 27-path selection of the HIST-03 run at `b28398026`**, with the exit-code limitation above. Not claimed: the 27-path run was not repeated after S12 (R8), the transitive import order of the reordered modules is not traced (R9).
- Operator-accepted deviation: the two implementation commits were made with `--no-verify` because of lint defects already present at the base (`authorizations/AUTH-HIST-NOVERIFY-031A-2026-11-08.md`); S12 removed those defects.
- Ticket state on `development`: `ready_for_audit`.

## 4. Product batch: plans written, HELD

Both tickets are `plan_pending`, have no `Approved:` line and are NOT reviewed or approved. They touch `intergrax/` or `applications/`, so each trips gate ACT-N11 until one IIAC re-activation (DR-12 = B). Published here for visibility only.

- `T-IGX-032-A` (P2-C, HIST03-F-015), digest `2628d1a834d2fcf25f47c55b0d60355811898c9e60337306b557f152d7ba7446`. Recommendation A2: the resolver `resolve_skill_execution_pinning_wiring` passes the document store to the leaf only when `kv_store` is None (KV authoritative, document store fallback); the leaf keeps rejecting both. One product file (about 3 lines) plus a new test file with 8 tests including a red-first set. New finding: `is_durable` is hard-coded True on both pin stores, so the durability validator cannot detect an in-memory backing (residual, not fixed). Wider blast radius unmeasured.
- `T-IGX-033-A` (FRZ-BND-03), digest `ae167672d4075e3e4dc190ac93c78a937d7ad0c435dfc3971e615143e4273775`. Moving is feasible (no ACCEPTED_LIMITATION needed). Recommendation (b2): an injected command set; Tier-0 gets an extension point, Tier-3 registers the demo commands. Cost: 7 product-path files, 3 test/gate files, one regenerated EBH-7 matrix, a changed CLI invocation. Also found: a second Tier-0 -> Tier-3 dependency (`product_manifest_registry.py`) outside HIST-02.
- `T-IGX-030-B` and `T-IGX-030-C` (EBH-4 R1 enum move; `invoker.py` fix) stay HELD, unchanged.

## 5. Open items (exact)

Operator or later batch:
1. Answers on the two product plans (see the plan reports under `audits/plan-p2c/` and `audits/plan-bnd03/`): P2-C - confirm A2, owner SKILL-X, KV authoritative, the bounded wider measurement (C10/C15), whether the hard-coded `is_durable` becomes its own ticket (the P2-C plan says "five" extra files in Q3 but names four in step 6; the plan reviewer should settle the list); BND-03 - accept the changed CLI invocation, what to do with the old names, regenerate the EBH-7 matrix row, separate finding for `product_manifest_registry`, join the single re-activation, a decision id only if an ACCEPTED_LIMITATION is chosen.
2. W1-PLAN step 2: agreement with the ENT-AUDIT-X stream (a separate session) on whether ONE IIAC re-activation covers both streams. Not assumed; no owner has contacted that stream.
3. The single IIAC re-audit / re-activation batch, including `T-IGX-021-B/C` (they do not exist yet) and the validator alignment (Q-12).
4. QUAL-X Dimension A replay on the final integrated SHA (DR-10, approved), then certification of the 19 candidate rows (each needs its own independent audit; no promotion without it).
5. Residuals of T-IGX-031-A: R1 duplicate `ollama` slug (product), R7 raw `TypeError` for a pre-built non-contract secrets store (product), `no_ci` still excludes `test_sec_ent.py` from GitHub CI, R2 (about 72 other bare `register_default_integrations()` callers unmeasured), R8, R9; the 27-path run under `aud-tests` did not exit after its summary (AUD-1; the process tree was later terminated on the Operator's instruction); the HIST-03 pin run exited normally; a similar hang was seen once on an aborted HIST-03 run (HK-01, wrong flags). Cause unknown; not shown to be caused by the batch.
6. Published plan text of T-IGX-031-A says no Operator approval was requested or given for the planner's `.venv`; the Operator later said they answered a question in the implementer's panel (the question named both cases), so that sentence may be inaccurate. An erratum is needed if the Operator confirms.
7. Polish text in the repository (838 lines inventoried by HIST-04): a ruling on functional data is still needed; unchanged.
8. EVOL-X roadmap row L489 inconsistency (prose says CLOSED, stage cell PENDING_AUTHORITY): DR-1 ratified the lines but edited no roadmap cell.

Dashboard counter 7 counts only the three OPEN registry entries OD-NEW-P2C, OD-NEW-BND03 and OD-NEW-ENT; the other Operator items are listed in this section.

Housekeeping: the hung process trees HK-01 and HK-03 were terminated on the Operator's instruction and the heavy lock is free. Sessions and worktrees listed in `MASTER_RESUME_INSTRUCTIONS.md` are for the Operator to close and remove.

## 6. Authorization records

- `authorizations/AUTH-HIST-27PATH-2026-11-08.md`: ONE 27-path diagnostic run, used by `aud-tests`; not renewed.
- `authorizations/AUTH-HIST-NOVERIFY-031A-2026-11-08.md`: the `--no-verify` acceptance for the pre-existing lint defects only.
