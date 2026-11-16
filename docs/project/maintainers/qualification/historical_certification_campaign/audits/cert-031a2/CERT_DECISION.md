# CERT_DECISION - cert-031a2, amended plan of T-IGX-031-A (Amendment A1, S12 lint normalization)

## VERDICT: APPROVE

Decision taken by session cert-031a2 on its own judgment, in the Operator's name, under DELEG-HIST-031A2-2026-11-08. Scope of the approval: the amended plan `governance/tickets/plans/PLAN_T-IGX-031-A.md`, commit 1c9240802b661dcdd3ee1c50344515f475603788, Plan digest dc0fcf736456534e923000eaf4cce1d569edee9db28ef73386a5bd8284b3cc6b. Nothing else is approved: no implementation, no audit, no PASS, no FRZ promotion, no IIAC activation, no T-IGX-030-B/C.

## R-2 (IIAC Amendment 01) - the four points
(a) a new session; (b) dedicated solely to this certification task; (c) not a continuation of any other session; (d) not an implementation session. Session statement: this session was started fresh for this brief, read the brief first, authored and implemented nothing in the T-IGX-031-A chain, and is none of plan-tests, rev-plan, rev-plan2, cert-031a, impl-tests, aud-tests, MASTER. Assurance: process_enforced (no cryptographic proof, as the Amendment states).

## Authority
- `delegations/DELEG-HIST-031A2-2026-11-08.md`: `sha256sum` = 6ab01c23da23899576fc47de0fa8d9de8b4416bee4c40bb45f126c5eb5f657a5, equal to the brief and to the `.sha256` file (before any other action).
- The published copy at 5279ffbee (`docs/project/.../delegations/DELEG-HIST-031A2-2026-11-08.md`) has the same sha256. `git cat-file -e 5279ffbee0cc37b0145cea9371b58b899ec7837d` and `-e 1c9240802b661dcdd3ee1c50344515f475603788` succeed; worktree plan-tests is on hist-cert/plan-amend at HEAD 5279ffbee, `git status` clean.
- The delegation is a MASTER transcription (not signed by the Operator); S-1 is struck, so it is not a condition of validity. Scope narrow: approval of this amended plan only. Used within scope.

## Evidence per check
1. Digest. `python -I scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-031-A.md` (cwd plan-tests) = dc0fcf73...3cc6b = ticket line 20 `Plan digest:`. `git diff 1c9240802 HEAD -- <plan>` is empty; `git diff --name-only 1c9240802 HEAD` = only the delegation record. PASS.
2. rev-plan2 findings (diff 00df54795..1c9240802, plan and ticket only, no test or product file):
   - F1 (noqa fallback unusable): `ALLOWED_NOQA_DELTA` per file, default 0, auditor sets w5_c to 1 for the fallback; the check is delta-based. ADDRESSED.
   - F2 (blank line after deleted F401 import): stated in the F401 row; stays, whitespace only, implementer must not delete it by hand. ADDRESSED.
   - F3 (over-claim of the script): "proves" replaced by "What the script checks" and "What it does NOT prove"; the script now keys imports by position (non-import statements before it in its body), checks non-noqa comments, and the auditor must read every moved import. ADDRESSED. In addition I confirmed below that the position check closes gap G1.
   - F4 (PARENT by hand; ruff pinning): the script computes `git rev-parse <S12>^` and fails unless the parent carries b28398026 content of the five files; the locked ruff via `uv run --extra dev-ci ruff` (0.15.20 in uv.lock) is mandatory, a global ruff is not evidence. ADDRESSED.
   - F5 (`af323c005` undefined): defined as the merge of PR #415, parent of 232d03175 (implementation base). ADDRESSED.
   - R9 now cites the reviewer's static trace and keeps the untraceable list open. Ticket: new digest line, second superseded digest 980394... recorded as never approved, test-contract line updated. No unrelated hunk.
   - rev-plan findings 1-9 still hold in the amended plan (grep and reading): the false venv-approval sentence is gone and the corrected wording is in section 3 (1); F-016 wording is "sufficient polluter", step 11 required for audit and R3 open (2); S6/S7 strengthened with `__cause__` text checks (3); probe B predicts 12 failed / 22 passed (4); R7 recorded and S5 note (5); S1b clears the prebuilt store (6); `reset_entry_point_spec_cache_for_tests` in the helper (7); ASCII only, 0 CR, 0 tab, 0 backslash, ends LF in plan and ticket blobs at 1c9240802 (8); "about 72" (9). PASS.
3. Scope. Edit scope is the four existing files; S12 "adds no file" and leaves the helper untouched; forbidden list unchanged (intergrax/ agents/ applications/ scripts/, six MATERIAL_GOVERNANCE_PATHS, conftest, markers, `no_ci`, pyproject, manifests). S12 touches only imports, line breaks and one deleted unused function-local import (reading of the S12 detail table; I did not re-run ruff, by brief). PASS.
   AST check, not vacuous. I extracted the script verbatim from the plan and ran it ONLY against scratch git repos I created in my session scratchpad (small stand-in files under the five plan paths, a tag `b28398026` on the "audited" commit, one S12 commit each):
   - equivalent reorder of imports in two files + the declared F401 deletion: `S12 EQUIVALENT`, exit 0;
   - same plus one changed assertion (`== 2` -> `== 3`): `FAIL tree without imports differs`, `S12 NOT EQUIVALENT`, exit 1;
   - module-level import moved to the end of the file: `FAIL import position changed`, exit 1 (this class, rev-plan2 gap G1, is now caught).
   A first run without the declared F401 deletion failed with `FAIL imports`, showing the declared removal is required and not merely permitted. A fourth case I tried (assertion dropped) failed with a git error in my scratch setup and is not evidence. Verdict: the check discriminates; it is the plan's script, not the implementer's real run.
4. Audited baseline. 232d03175 and b28398026 are ancestors of HEAD; `git diff --name-only 232d03175^ b28398026` = the five test files only. aud-tests PASS_SCOPED report read: steps 1-5 (14/20/34/34/1), probes A (4 red) and B (12 red) as predicted, step 8 green, 27-path run 6 failed / 347 passed with 0 `already registered` and the 6 P2-C failures unchanged; gate-weakening checklist: none. The amended plan changes nothing of that content: S1-S11, steps 1-8, 10, 11 and the original gate-weakening checklist are unchanged in the diff. S12 cannot weaken an assertion by construction (check 3), and the delta audit must cover ruff clean (locked ruff), AST equivalence and steps 1-5 rerun (step 12 a-c). PASS.
   R8/R9 sufficiency. R8: the 27-path run is not repeated; the Operator's authorization for one run was used before S12. I judge that acceptable: the only differences are import order/grouping, line breaks and one dead local import; test bodies, fixtures, markers are AST-identical; the reordered groups are, by rev-plan2's static trace, side-effect-free, disjoint and loaded after the 2965-module closure of the file's own imports; steps 3 and 4 rerun the four files in both orders in one process. R9 stays honestly open for dynamic imports, importlib/entry points, third-party side effects and conftest. A further 27-path run would cost about 8 minutes plus a new Operator authorization for little added assurance. Condition (not blocking): any later statement that F-016 is resolved at the FINAL S12 SHA must say it is inferred (audit of b28398026 plus AST equivalence), not measured at that SHA.
5. Roles. Plan section 6: implementer, delta auditor and approver are three different fresh sessions; the approver is this session. No `Approved:` line existed before my edit (grep `^Approved:` = 0); the earlier approval (cert-031a, digest a83842...) is labelled SUPERSEDED. No PASS or FRZ promotion is claimed in the plan or ticket. T-IGX-030-B/C are out of scope and listed as such. PASS.

## Residuals (accepted, not closed by this approval)
R1, R2, R3 (until a clean 27-path run on the final tree), R7 (product, plan step 3), R8, R9 as written in the plan. `test_sec_ent.py` stays `no_ci`. Step 10's `git diff --name-only 6734971cf..HEAD` guard is stale text (base moved by PR #415); the implementer/auditor must use `af323c005` or the S12 parent, and the script's own changed-files check is the real guard for S12 (pre-existing text, flagged by rev-plan2; not a blocker).
Cosmetic: the plan states rev-plan2 script length differently from the reviewer's "76 lines" (the plan's script is longer after F3/F4 hardening); not a defect.

## What I could not verify
- Any run result after S12 (by brief: no tests, no ruff run); the predicted counts 14/20/34/34/1 and ruff-clean are predictions until the implementer's and delta auditor's runs.
- That the real S12 commit will contain exactly the simulated edits (planner and reviewer simulated with the locked ruff; I did not run ruff).
- Dynamic import behaviour (R9 list) and pre-push gate execution.
- That the Operator gave the quoted chat commission, and this session's novelty/dedication beyond its own statement (process_enforced).
- The scratch repos I created: the cleanup of two scratch directories (`scr`, `scr2` with tiny stand-in git repos) under the session scratchpad `C:/Users/XPS/AppData/Local/Temp/claude/4424d8d373fa87a5/bf432555-58a1-478a-a2d4-6746a8271e4b/scratchpad/` was refused by the permission classifier (recursive delete) and I did not work around it; the script copy and driver were deleted. They are outside the repository and contain no project content. MASTER/Operator may delete them.

## Action taken
One line added under `## Approved` of `D:/Projekty/_wt/intergrax/plan-tests/governance/tickets/T-IGX-031-A.md` (nothing else changed, plan file untouched, no commit; MASTER commits).
