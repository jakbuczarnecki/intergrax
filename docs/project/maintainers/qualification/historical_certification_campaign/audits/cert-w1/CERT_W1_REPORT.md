# CERT-W1 report - session cert-w1

Date: 2026-11-08. Branch `hist-cert/cert-w1` (local only, ahead of `origin/development` 70cceae21 by 2 commits). Nothing pushed, no PR, no merge, no rebase, no reset, no amend. No test run.

## Verdicts

| Item | Verdict |
|---|---|
| W1-R1 (FRZ-SEC-05) | SIGNED |
| W1-R2 (Decision-03 / D03-A; SEC-01, REL-03, REL-05, PRD-08) | SIGNED |
| W1-R3 (FRZ-REG-02, 03, 06, 09) | SIGNED |
| W1-PLAN | SIGNED, sequence acceptance only |

All four are delegate lines (never Operator-written), assurance `process_enforced`.

## Commits

1. `cbb511ec7ded7b55f068444e326158fcf807960a` - byte-identical copy of `delegations/DELEG-HIST-W1-2026-11-08.md`.
2. `8ccbcef4450ea6b015bb3535a4f31950c29483b9` - `decisions/OPERATOR_DECISION_RECORDS_W1.md` (148 lines, LF, ASCII). The four `Approved` lines end with the sha of commit 1.

Delegation sha256 verified before work: 747dce743947fa2d88d00b85d66631b2d562fc5939d3c8e83e0a44e89a4c9a1c (file, `.sha256`, brief, and staged blob all equal).

## Evidence in short

- W1-R1: checklist L222 (SEC-05) is already a scope-limited statement ("scoped capability evidence only; global criterion remains OPEN - PROD-Q still required"); PHYSQ JSON shows one reference provider, one allow/deny endpoint pair, redirect not escaped; both PHYSQ shas are ancestors of the base; C10/C15 rule asks for exactly this labelling.
- W1-R2: ratified text found (Decision-03, OD-D03-A-01, conditional infrastructure direction). Split maps onto the text (contract/gate semantics = a; Kubernetes, Spanner + Azure witness, trust domains, cloud = b).
- W1-R3: Population E is defined at `QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md` L2361 ("Full freeze checklist rollup - ENT-AUDIT-X"). WBS T-B7-PROMOTE-REG (L60) depends on T-B7-L3-FREEZE (L38, `run_enterprise_audit.py --mode FREEZE`). W1-R3 fills the gap DR-4 recorded as residual (1) in the W0 records; no contradiction.
- W1-PLAN: ACT-N11 triggers only on `intergrax/ agents/ applications/` prefixes (`iiac_common.py` L22, L95-110) and the six material paths (`iiac_activation_gate.py` L40-49, L329-344); tests-only does neither. Consistent with DR-12 = B and DR-10. Tickets: 030-B/C exist, `plan_pending`, HELD; test-only F-006/F-016, P2-C and BND-03 tickets do not exist.

## Findings MASTER should know (things that differ from, or go beyond, the brief)

1. **T-IGX-021-B and T-IGX-021-C do not exist either** (only 021-A; it names them as follow-ups). W1-PLAN step 3 lists them. 021-B edits "material verifiers", so it must stay before the single re-activation (step 4); the plan does that.
2. **Only SEC-01's row text names Decision-03 / D03-A.** The REL-03, REL-05 and PRD-08 rows do not. The per-row (a)/(b) mapping for those three is the HIST-03 matrix thesis plus my reading of Decision-03; flagged in the record.
3. **W1-R2 does not exempt the (b) parts of SEC-01, REL-03, REL-05 from the checklist rule** (`OPEN = 0` for architecture-scoped rows), because they are not on L66 and L66 was deliberately not changed. A later checklist ruling is needed if those parts are to stay OPEN at ARCH-FREEZE.
4. **D03-B / D03-C**: the text does not say whether offline or simulated evidence counts as Dimension A. The record requires such evidence to be labelled offline/simulated with no Kubernetes or exactly-once claim.
5. **SEC-05 tension**: HIST-03 classes it architecture-scoped (A), while the row itself says PROD-Q is still required. Not resolved by W1-R1.
6. **"Per-host isolation" is not defined in the repository.** I disclosed my reading (one reference substrate, one policy, one allow/deny pair; no per-real-provider or per-production-host proof). The Operator may correct it.
7. **REG-10** is among the 19 candidate rows (the only REG row there) and is governed by DR-4 (b), not by W1-R3. None of REG-02/03/06/09 is among the 19.
8. **F-006 / F-016 "tests-only" is a thesis.** F-006 is "run pending" with a static note about a name binding in product code; if a product edit is needed it moves from step 1 to step 3.
9. Step 6 wording ("certification of the 19 candidate rows") is ambiguous; recorded as a later act needing its own exact-SHA independent audit, not a promotion.

## Open items for MASTER

- Order the independent non-author review of `OPERATOR_DECISION_RECORDS_W1.md` (same pattern as rev-w0).
- Delegation record is a transcription; if the Operator wants the reading in "MASTER's reading" narrowed or corrected, that needs a new delegation record, not an edit.
- PR must be merged with a merge commit (not squash or rebase) or a follow-up record must point to the post-merge sha, since the `Delegation:` references point to commit `cbb511ec7`.
- Tickets still needed before step 3 can start: test-only F-006/F-016, P2-C corrective, BND-03 move, T-IGX-021-B/C; each needs its own plan and `Approved:` act. 030-B/C stay HELD.
- Step 2 (agreement with the ENT-AUDIT-X stream on one re-activation) has no owner yet.
- Not verified: other IIAC verifiers beyond the two gate files; whether any test file is bound by the ACT-N13 test matrix; the scope of the ENT-AUDIT-X Population E rollup.

## Process notes

- A first compound Bash command was denied by the permission layer; I then ran one command per call, as the brief asks.
- The commit hook's `uv` bootstrap created a `.venv` in the worktree on the first commit (untracked, ignored; no tracked file affected).
- The records file was checked for CR bytes (0), backslashes (0) and non-ASCII bytes (0) before commit, and the staged blob has no CR.
