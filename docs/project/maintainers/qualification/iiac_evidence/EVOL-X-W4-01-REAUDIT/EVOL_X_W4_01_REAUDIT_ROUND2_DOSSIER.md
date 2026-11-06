# EVOL-X W4-01 — round-2 delta re-audit dossier

**Auditor session:** `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` (`independent-quality-auditor`, same session as round 1; implementer of round 2 is the session that requested this audit, not me)
**Audited code SHA:** `ca78b1c1f73f1e23c3a9bfed118988c12a0c71fa` (detached in the auditor worktree) · previous pin `9f8b04d2b…` · `origin/development` = `af7b98019`; `git diff ca78b1c1f..origin/development` outside `docs/` and `governance/tickets/` is empty.
This is audit evidence, not a certification. B-4 is not accepted; nothing was fixed, signed, promoted or closed by me. Machine-readable result: `EVOL_X_W4_01_REAUDIT_ROUND2_FINDINGS.json`. Raw output: `raw2/`.

## 1. Verdict

**PARTIAL — one exact blocker: `W402-01` (MEDIUM).** Four of the five targeted findings are closed under independent reproduction and variants; RA-03 is closed for every path I had listed but the new check is evaded with git index flags.

| Finding | Verdict |
|---|---|
| W401-RA-01 replace refs | **PASS** |
| W401-RA-02 lifecycle values / certification file set / duplicate keys | **PARTIAL** (residual free-text channels, W402-03, LOW) |
| W401-RA-03 HEAD-bound inputs | **PARTIAL** (W402-01 MEDIUM blocker; W402-02 LOW) |
| W401-RA-04 `--full-history` | **PASS** |
| W401-RA-06 format document | **PASS** |
| W401-RA-05 / 07 / 08 (left open on purpose) | unchanged, **PARTIAL / INFO**; parser code and exit-code contract not touched by the diff |
| W402-04 untested git redirections (commit-graph, fsmonitor, alternates) | **UNVERIFIED** |

## 2. Delta reviewed

`git diff 9f8b04d2b ca78b1c1f`: the preflight (+206/−28), the B4 test module (+219, **0 removed lines**, so no existing assertion weakened), the new format document, report/handoff/ticket text and the W4-F08 record/template (docs). No change under `intergrax/`, `agents/`, `applications/`, `scripts/governance/`, the activation record, `iiac_evidence/` or `.gitattributes` (0 files). Workflow unchanged.

## 3. Original repros re-run on `ca78b1c1f` (fresh fixtures built with the round-2 builders)

| Repro | Round 1 | Round 2 |
|---|---|---|
| S1 widening text in `certification_publication_sha_binding`, `frz_checklist_rows_changed` | Gate A PASS | **FAIL** (`A-19 … neither the text published at P2…`, `must be a boolean`) |
| S2 extra indexed file in `certification/` | PASS | **FAIL** (`A-19 SCOPE_ADDENDUM.md was added…`) |
| S5 duplicate `certification_verdict` key | PASS | **FAIL** (`A-17 malformed JSON … duplicate JSON object key`) |
| attack2 dirty FRZ checklist / register / governance JSON (HEAD failing, worktree corrected) | Gate B PASS | **FAIL** (`B-12`/`A-21 uncommitted changes in verified inputs`) |
| attack3 S4 side-branch blocker dropped by `merge -s ours` | Gate A PASS | **FAIL** (`A-18 … disappeared without closure evidence`) |
| attack3b S6b `refs/replace/<P2>` with widened LIM-01 | Gate A PASS | **FAIL** (`A-22 repository carries replace refs` plus the A-19 widening errors) |
| attack4 replace-ref blocker baseline, attack5 V1–V3, controls | FAIL / PASS as before | unchanged; controls still PASS (honest base: Gate A PASS, closed fixture: both PASS) |
| positive controls: limitation reorder, cert record re-indent | PASS | PASS |

## 4. Variants and new attack paths

* **V-a (new, W402-01):** skip-worktree **and** assume-unchanged on a tracked input: `git status --porcelain` is empty; Gate A **and** Gate B PASS although HEAD holds the widened row. FAIL.
* V-b same-size edit with restored mtime: detected (`B-12`).
* **V-c (W402-02):** one comment line appended to `scripts/governance/iiac_common.py` or `scripts/qualification/evol_x_frz_evol_canonical_matrix.py` in a real clone: not flagged (control: the preflight script itself is flagged `A-21`).
* **V-d/e/f/g (W402-03):** free text appended to the open B-5 blocker, free keys in resolved blocker entries, a long status token, and a committed addendum at `iiac_evidence/EVOL-X/` are still accepted; the same addendum in `audit/` or `certification/` is rejected.
* Real chain (clone of `ca78b1c1f`): a replace ref → `A-22`; a `.git/info/grafts` file → `A-22`; after cleanup back to exactly two `A-18` (`raw2/real_chain_variants.txt`).
* Shallow clone: refused (fail-closed; CI uses `fetch-depth: 0`).

## 5. Legitimate operations

* Fresh clones with `core.autocrlf` **true / false / input** (bundle clones of `ca78b1c1f`): `git status` clean, activation `IN_FORCE`, preflight repository class `canonical`, Gate A exactly the two `A-18` (B-4, B6-P3), **no `A-21`/`A-22`**, no `B-12`/`B-13` (`raw2/real_clones_autocrlf.txt`).
* Untracked unrelated files under the qualification tree and `scripts/` do not trip the check (V-i). Untracked junk inside an evidence directory does (by design).
* The strict-equality positive controls from round 1 (limitation reorder, re-indented cert record) still pass; manifest re-indent is still rejected by the byte contract (as before).
* Cost of the new rule: evidence must be committed before the preflight is run (as the report states); two B4 canonical tests fail on an uncommitted tree by design (I ran on a clean committed tree: pass).

## 6. Test and gate results (clean committed tree, Windows, `core.autocrlf=true`)

| Command | Result |
|---|---|
| `pytest …preflight_b4.py …ruff_evidence_exemption.py -q -n 6 --basetemp=D:/w4a2/bt` | **448 passed, 0 failed, 11 min 08 s, EXIT=0** (423 + 25 new) |
| `pytest tests/unit/governance -q -n 4 --basetemp=D:/w4a2/bg` | **145 passed, 1 skipped (symlinks), 0 failed, 14 min 29 s, EXIT=0** |
| `verify_iiac_policy_activation --require-in-force` | `IN_FORCE @ IIAC-2026-11-05.1`, exit 0 |
| canonical `--gate b6-readiness` | FAIL, exit 1, exactly two `A-18`, no A-19/A-21/A-22 |
| canonical `--gate formal-closure` | FAIL, exit 1, no A-21/A-22/B-12/B-13 |
| GitHub Actions run `42224727428` on `af7b98019` (docs delta from the pin) | success: governance-policy-pin, evol-x-b4-preflight, ci-smoke, governance-fast (see `raw2/gh_run_af7b98019.txt`) |

## 7. Not verified

Linux container run by me (Docker daemon unusable here; CI run above is the substitute); commit-graph / fsmonitor / alternates variants (W402-04); committer-date ordering in the first-publication derivation (can only cause a fail-closed rejection by reasoning, not tested).
Operator-owned items untouched: W4-F08 (I did not read, fill or sign the record/template), OD-1, OD-EVOL-12, OD-EVOLX-08, P3, B6, FRZ promotion, EVOL-X closure.
