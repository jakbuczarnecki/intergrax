# HIST-03 - documentation patch scope note (P-01, P-02, P-07)

Patch: `PATCH_H03_P01_P02_P07_492e361.diff` (sha256 `f3eaa1566f4cd4e0e0c5d524ba6cd3fa2734d0262d04e722c260e34de60d7725`), base `origin/development` `492e361c0706e1aedeff44900a8a228345bf1d75`. Authority: MASTER relay, Operator DR-11=A (publish documentation patches after independent non-author review). DR-9 is unanswered.

**Not reviewed, not applied, not committed.** HIST-03 is the author; the patch needs an independent non-author review before publication. No worktree was modified (hist03 HEAD `ba07ebaa3`, `git status` empty). No FRZ status is changed.

## Verification
- Baseline blobs were taken with `git show 492e361c0:<path>`; `git hash-object` equals the `492e361c0` blob id for all 4 files.
- `git apply --check` on a temp tree outside every worktree: exit 0.
- `git apply --cached --check` against a temporary index (`GIT_INDEX_FILE` in scratchpad, `read-tree 492e361c0`): exit 0. The real index was not touched.
- Machine check of the diff: only 4 `FRZ-OBS-04..07` rows change, with cells 0-5 (ID..STATUS) identical and the old evidence text preserved as a prefix; status stays OPEN. No other FRZ row is touched. LF only, no CRLF.
- Diff stat: 4 files, 5 removed / 8 added lines.
- Not verified: Markdown rendering, link resolution of the new relative link in P-01 (target file is in the same directory), or any repo doc-lint/CI gate (not run; no pytest run for this task).

## Per proposal

| ID | Decision | What the diff does | Notes |
|---|---|---|---|
| P-01 | INCLUDED | `ROADMAP_REPLAY_X_FRZ_225_RECONCILIATION.md`: blockquote after `AUDITED_HEAD`: HISTORICAL @ `a2ed11f0` (95 PASS / 130 OPEN); PATH_B column not authoritative; the live-scoped set is defined only by the checklist Path B table (referenced by section name, not line numbers, which drift). | Per-row tags deliberately not edited. |
| P-02 | INCLUDED | Checklist evidence cells `FRZ-OBS-04..07`: appended non-status note naming bearing TRACE-X rows (OBS-04: TRC-12, TRC-01; OBS-05: TRC-10 (TRACE-X-P6); OBS-06: TRC-06, TRC-03; OBS-07: TRC-01, TRC-02, TRC-11) and stating OPEN is unchanged, pending independent audit, "not a mapping acceptance". | Original proposal level was "MASTER/SSOT lane after audit"; included at MASTER's instruction, so the independent review should confirm that timing. The cited TRC rows exist and are PASS at `492e361c0` (L139-150). The 492 checklist differs from baseline by one unrelated line (L~29, internal-certification sentence); the hunk is far from it. |
| P-07 | INCLUDED | (a) P3 decision package, section J item 2: annotation that the three E.1 "NOT EXECUTED" statements (L40, L58, L222) are superseded by ledger row `PROD-Q-R1-R7-Q-EXEC-E1` (E.1 CLOSED - SCOPED D0-D4 REGRESSION / OPERATOR ACCEPTED, 21 modules / 209 tests, no monolithic artifact, no parent closure). (b) `PROD_Q_PLATFORM_PRODUCTION_QUALIFICATION.md` "Degraded operation" row (L187): LKW paths prefixed with `applications/local_workspace_application/` (`hosting/readiness.py`, `serving/readiness_routes.py`, `tests/hosting/test_hosted_readiness.py`, `tests/host/test_lkw_fastapi_lifecycle.py`); all four exist at `492e361c0`. | Original text of the P3 package is not rewritten (annotation only; P3 closure scope unchanged). Bare `readiness_routes.py` / `lifecycle.py` in the neighbouring Health/Readiness rows (L185-186) were NOT in the proposal and are left as is; a follow-up if the reviewer wants them. |
| P-03 | EXCLUDED | - | Operator-level ruling (DR-9, unanswered): SEC-06 split note. |
| P-04 | EXCLUDED | - | Operator-level ruling (DR-9, unanswered): PRD-04 scope; would also touch `_p0_frz_reconciliation.py`. |
| P-05 | EXCLUDED | - | Needs a new data-responsibility record authored by an implementer; not a pure annotation. |
| P-06 | EXCLUDED | - | Test path (`tests/qualification/prod_q/_p0_frz_reconciliation.py`); needs a ticket and implementer batch. |
| P-08 | EXCLUDED | - | Ledger label vs DS-3 = DR-3, unanswered; HIST-01/HIST-04 owned. |

## Apply
`git apply PATCH_H03_P01_P02_P07_492e361.diff` from the repository root on a checkout of `492e361c0` (or a descendant where these 4 files are unchanged). If the base moves, regenerate from the Python edit script (not kept in the repo; scratchpad only).

## Non-claims
No FRZ row promoted, no stage closed, no Path B change, no production readiness. The P-02 note is evidence pointing, not acceptance of the OBS rows.
