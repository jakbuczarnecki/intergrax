# REV-MASTER delta review: plan of the remaining (write) steps

## Context

REV-MASTER (WP-12 phase check 7, delta review) must confirm that the HIGH findings RV-01..RV-06 of
`.tmp/session/eqo-a-remediation/REVIEW/REVIEW_REPORT.md` are fixed. It must also check that the
MEP-AF2 v2.1 section-0 amendments are correctly reflected in the regenerated `WP-12/DAG.md`, and
judge whether delegating phase check 3 to T-AF2-RW-PLAN is sufficient. MEP-AF2 makes this review
a hard prerequisite of SCOPE. Role: independent reviewer, read-only. Repo head:
`origin/development` = `7960ae7f6`, verified.

All the analysis is done, read-only and in memory: the DAG was parsed, master_rows was
cross-checked, `_before_af2_regen/` was diffed, and the dashboard merged DAG was read. Only the
write steps below remain.

## Findings (to be written into REV_MASTER_DELTA.md)

Verdict per RV:
- **RV-01: PARTIAL.** The AF2 regen moved PT-VBN to W7. `master_rows.jsonl` was not regenerated
  (11:04), so A7-005, A7-027 and EDER-D-0010 keep `closure_wave` 6, below their audit node
  PT-VBN at W7. `PHASE_CHECKS.md:7` check 1 is now FAIL. NR-1 (A2-037 W2) is also still present.
- **RV-02: FIXED.** No BLOCKS, worst-case-BLOCKS or HIGH row has a node outside the ENTRY
  ancestry. DS-ADM is the only exception, and it is excluded by design. `entry_gate.json`: 50
  required nodes, 0 missing.
- **RV-03: FIXED as a plan correction.** Strict criterion: 18 of 29 missing, 35 FRZ rows
  UNPLANNED, status FAIL. The gap itself is delegated.
- **RV-04: FIXED.** Effective impact is 96/25/62/56. 0 rows are milder than the dossier.
- **RV-05: FIXED.** DS-EXC is a hard ancestor of SCOPE. The head facts hold at `7960ae7f6`.
- **RV-06: FIXED.** 16 DEBT_ENTRY rows have nodes. PT-VBN, DS-Q and DS-ADM exist. The D-0010
  wave defect is counted under RM-01.

Amendments (a) to (f):
- (a) is OK.
- (b) is OK but approximated: M3 stands in for LOCK, and PT-Q13 stands in for ECAP-2C.
- (c) is OK.
- (d) has a wording defect.
- (e) is OK in DAG.md, but the A6-004 data is stale.
- (f) is OK but has no source in section 0.
- The DAG has 90 nodes and 193 hard / 20 soft edges. It is acyclic and has no wave-order
  violations.

Findings table:

| ID | Sev | Finding | Fix |
|---|---|---|---|
| RM-01 | HIGH | The regeneration was partial. Only build_plan outputs were regenerated (12:25). master_rows, MASTER_PLAN, AUTHORIZATION_REQUESTS, RECONCILIATION, BLOCKERS and OPEN_QUESTIONS are stale (11:04-11:05). Check 1 is FAIL (A7-005, A7-027), and EDER-D-0010 is W6. `MASTER_PLAN.md:77` still has PT-VBN in W6, `:88` still has "PT-Q13 <- PT-VBN", and `:67` still has "FCS = RB-2 audited SHA". The AR-12-05 preconditions are "FCS; DS-5". CORRECTIONS_LOG has no AF2 entry. The dashboard nevertheless marks T-AF2-DAG-REGEN DONE. | The WP-12 owner re-runs reconcile.py, gen_master.py, gen_auth.py and build_plan.py, adds an "AF2 regen" section to the log, and gets check 1 back to PASS. |
| RM-02 | HIGH | MEP-AF2 :90 and :515 make REV-MASTER and T-AF2-RW-PLAN hard prerequisites of SCOPE. Neither `WP-12/DAG.md` encodes this (SCOPE prerequisites; REV-MASTER is listed as a non-ancestor) nor the merged `dashboard_steps.json` (no node depends on either of them). As a result PT-WA, the RW train, can start without the check-3 plan and without this review. | Add hard edges REV-MASTER → SCOPE (plan_data.py) and T-AF2-RW-PLAN → SCOPE (build_steps.py), then re-run validation. |
| RM-03 | MEDIUM | The check-3 delegation is not independently verified. RW-PLAN is executed by the orchestrator (kind DOC) with no audit node. Nothing says that check 3 is recomputed with the strict criterion over RW-PLAN's output. There is no red-first evidence that the negative tests fail on the pre-fix code, the regression scope is per ticket rather than per finding, and the 6 audit-flag confirmations have no owner. | Add an independent review of RW-PLAN that re-runs the strict check 3, plus a red-first rule. |
| RM-04 | LOW | (b) is weaker than MEP: ECAP-2C sits between PT-Q13 and PT-VBN and is not named in WP-12. M3 is W7 in WP-12 but W6 in the dashboard and in MEP :523; MEP :152 says LOCK (W7) = M3. | Add a note on the node and settle the wave of M3. |
| RM-05 | LOW | (d): the FCS text keeps "(= RB-2 audited SHA)", and so do MASTER_PLAN:67 and OPEN_QUESTIONS:9. | Use the wording of the amendment. |
| RM-06 | LOW | (e): A6-004 keeps PT-C01 as a follow-on node, so its closure stays at W6. After the fix it would move earlier, and that change needs an explicit reason. | Regenerate the row and record the reason. |
| RM-07 | LOW | (f) does not appear in section 0 (its basis is MEP §G :375 and §J :516). PT-E05 is a member of the RW train but has no SCOPE ancestor. | Record (f) in section 0, and decide whether PT-E05 needs SCOPE as a prerequisite. |
| RM-08 | LOW | Items carried over from DELTA_REVIEW_2 are still present: NR-1 to NR-4, and PHASE_CHECKS row 7 still reads "delta review OPEN". | Housekeeping. |

Phase check 3: the delegation exists (MEP §0 row 2, §F FIND-AF2-23, §G :378-382, §J :515) and
its content is adequate. It is **insufficient as wired** because of RM-02 and RM-03.

Overall verdict: **REV_MASTER_FINDINGS**.

## Steps after approval

1. Create `D:/Projekty/intergrax/.tmp/session/arch-freeze/rev-master/scratch/`. Copy `WP-12/`
   (excluding `__pycache__`) to `scratch/WP-12/`, and `WP-4/rows.jsonl` to
   `scratch/WP-4/rows.jsonl`, because build_plan reads `HERE.parent/WP-4`. Run
   `python -I build_plan.py` inside `scratch/WP-12`. Diff DAG.md, entry_gate.json and
   PHASE_CHECKS.md against WP-12. Expected result: identical, except for the git-status lines of
   check 6. Confirm that the graph is acyclic, that the ENTRY gate has 50 required nodes and 0
   missing, and that check 1 is FAIL.
2. Re-verify the NR-3 text of AR-12-04 (grep).
3. Write `D:/Projekty/intergrax/.tmp/session/arch-freeze/rev-master/REV_MASTER_DELTA.md` in
   English. It will contain the scope and method, the RV table, the amendment table, the
   check-3 judgement, the findings table (ID | severity | evidence | fix), the verdict, and what
   could not be verified.
4. Leave a 2-line note in the shared notes folder (`REV_MASTER_NOTE.md`).
5. Confirm that nothing was written to WP-12 or to tracked files (`git status --porcelain`), and
   that the branch is still `development`. Then reply in Polish, in 10 lines or fewer.

No pytest or uv, no edits to WP-12 or to tracked files, no fetch.
