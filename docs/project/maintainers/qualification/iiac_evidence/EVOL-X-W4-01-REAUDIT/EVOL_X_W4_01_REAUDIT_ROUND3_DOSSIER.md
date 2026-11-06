# EVOL-X W4-01 — round-3 delta re-audit dossier

**Auditor session:** `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` (`independent-quality-auditor`)
**Audited code SHA:** `2e0ed9c6b07fd3f4218d769b35a70cb7230db8fb` (detached in the auditor worktree) · previous pin `ca78b1c1f` · `git diff 2e0ed9c6b origin/development` outside `docs/` and `governance/tickets/` is empty.
Audit evidence only: nothing was fixed, signed, certified, promoted or closed; B-4 is not accepted; no operator record or W4-F08 file was touched. Findings: `EVOL_X_W4_01_REAUDIT_ROUND3_FINDINGS.json`; raw output: `raw3/`.

## 1. Verdict

**PARTIAL — one exact blocker: `W403-01` (MEDIUM).** The round-3 fixes close every concrete W402 repro I had, but the "verifier code is bound to HEAD" claim is defeated by a forged bytecode cache, which I used to make a canonical clone print PASS with exit 0.

| Finding | Verdict |
|---|---|
| W402-01 HEAD binding without `git status` (skip-worktree, assume-unchanged, same-size edit) | **PASS** |
| W402-02 verifier code + `.gitattributes` bound to HEAD | **PARTIAL** (source bound; bytecode not → W403-01) |
| W402-03 open-blocker text, resolved-entry keys, enums, evidence-root layout | **PARTIAL** (residue W403-02, LOW) |
| W402-04 fsmonitor / commit-graph / env redirection | **UNVERIFIED** (forged commit-graph and imported modules' git calls not exercised) |
| W401-RA-01 / RA-02 / RA-04 / RA-06 | **PASS** (original repros re-run) |
| W401-RA-03 | **PARTIAL** (W403-01) |
| W401-RA-05 / 07 / 08 | open on purpose, **PARTIAL / INFO**, code untouched |

## 2. Delta reviewed

`git diff ca78b1c1f 2e0ed9c6b`: preflight (+/- 486/58 together with its tests), the B4 test module (one removed line: a fixture literal replaced by the *published* B-5 text, i.e. stricter), OD-1 record, W4-F08 record and report/handoff text. No change under `intergrax/`, `agents/`, `applications/`, `scripts/governance/`, `.gitattributes`, the activation record or `iiac_evidence/` (0 files).

## 3. Repros re-run on `2e0ed9c6b` (fresh fixtures)

All previous attacks are rejected for the intended reason: S1, S2, S5 (round 1), S3 dirty-tree family, S4 side-branch merge, S6 replace ref (`A-22`), V1–V3 repoint/widen (round 1/2); **V-a** skip-worktree and assume-unchanged (now `B-12 … hidden changes`), **V-b** same-size edit with restored mtime (`B-12`), **V-d** open blocker text extended (`A-18 no longer carries the text it was published with`), **V-e** extra key in resolved entries (`unexpected key 'note'`), **V-f** free `stage_closure_status` (`must be one of ['CLOSED','NOT_CLOSED']`), **V-g** addendum at the evidence root (`outside the known EVOL-X evidence directories`), V-g2 and V-j unchanged. T9: an activation record changed under skip-worktree is rejected (gate compares bytes with HEAD). Positive controls (honest base, closed fixture, limitation reorder, cert-record re-indent) still pass; manifest re-indent still fails by the byte contract.

## 4. New variants

* **W403-01 (FAIL, blocker).** `_dirty_input_errors` skips `__pycache__` and `*.pyc`. A forged unchecked-hash `.pyc` for `scripts/governance/iiac_common.py` in the ignored cache directory: git status empty, no A-21, marker written, and with a payload that prints PASS and exits 0, `--gate formal-closure` on a canonical clone returned `PASS`, exit 0 (control: after deleting the pyc, FAIL, exit 1). (`raw3/t2_pyc_plant.txt`)
* **W403-02 (LOW).** Addenda in `operator_decisions/`, `b6_acceptance/`, `b4_reaudit/`, a new directory with its own `EVIDENCE_INDEX.json`, free keys in `closure_evidence` items, a gitlink in `audit/` are accepted (the last two reach PASS in the closed fixture). `certification/` stays exact (gitlink and file rejected).
* Repo-root module shadowing: the stdlib modules and the governance modules' imports are all loaded before the repository root is put on `sys.path`; measured set of modules first imported afterwards is only `scripts`. No shadowing path found. The script directory is removed from `sys.path`.
* Index bits, fsmonitor: content comparison and `ls-files -v`; my fsmonitor hook did not actually hide the change from this git, so that sub-case is weak.
* Line-ending trick: a worktree equal to the CRLF expansion of a pure-LF blob is accepted only where `text` is not unset; `_read_bytes` then returns the blob, and Python source semantics do not change with CRLF, so I found no exploit.
* Symlinks: could not be created on this Windows host (the repo's own symlink test is skipped); not tested. Gitlinks: see W403-02/T6.

## 5. Legitimate operations

Clean clones with `core.autocrlf` true / false / input: status clean, activation `IN_FORCE`, preflight class `canonical`, Gate A exactly the two `A-18`, no `A-21/A-22`, no `B-12/B-13`. A clean linked worktree (of a clone and my audited worktree) behaves the same. Modified `README.md`, untracked files at the repo root, under `tests/` and in the qualification docs tree are accepted. A modified or untracked file anywhere under `scripts/` is rejected (W403-03, LOW usability). Runtime: `formal-closure` 76 s idle, `b6-readiness` 2 min 39 s while the suites ran.

## 6. Tests and gates (clean committed tree, Windows, `core.autocrlf=true`)

| Command | Result |
|---|---|
| `pytest …preflight_b4.py …ruff_evidence_exemption.py -q -n 6 --basetemp=D:/w4a3/bt` | **481 passed, 0 failed, 18 min 41 s, EXIT=0** (448 + 33 new) |
| `pytest tests/unit/governance -q -n 4 --basetemp=D:/w4a3/bg` | **145 passed, 1 skipped (symlinks), 0 failed, 16 min 59 s, EXIT=0** |
| `verify_iiac_policy_activation --require-in-force` | `IN_FORCE @ IIAC-2026-11-05.1`, exit 0 |
| canonical `--gate b6-readiness` | FAIL, exit 1, exactly the two `A-18` (B-4, B6-P3) |
| canonical `--gate formal-closure` | FAIL, exit 1, no A-21/A-22/B-12/B-13/A-99 |

## 7. Not verified

Forged commit-graph; `GIT_DIR` / `GIT_INDEX_FILE` and other redirecting variables beyond the six refused; git calls made by the imported IIAC modules (no commitGraph/fsmonitor flags); symlink cases (host cannot create them); CI on the round-3 SHA and a Linux container run (not reproduced here). Operator-owned items untouched: W4-F08, OD-1 (its delegated record read only), OD-EVOL-12, OD-EVOLX-08, P3, B6, FRZ promotion, EVOL-X closure.
