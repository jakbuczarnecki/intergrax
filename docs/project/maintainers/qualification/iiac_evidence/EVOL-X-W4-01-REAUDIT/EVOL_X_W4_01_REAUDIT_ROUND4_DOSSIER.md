# EVOL-X W4-01 — round-4 delta re-audit dossier

**Auditor session:** `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` (`independent-quality-auditor`)
**Audited code SHA:** `dc650b8df67349896bb813c3fd723fcd36717362` (detached in the auditor worktree) · previous pin `2e0ed9c6b` · `git diff dc650b8df origin/development` outside `docs/` and `governance/tickets/` is empty.
Audit evidence only: nothing fixed, signed, certified, promoted or closed; B-4 not accepted; no operator record touched. Findings: `EVOL_X_W4_01_REAUDIT_ROUND4_FINDINGS.json`; raw output `raw4/`.

## 1. Verdict

**PARTIAL — one exact blocker: `W404-01` (HIGH), a regression introduced by the round-4 change itself.**

| Finding | Verdict |
|---|---|
| W403-01 forged bytecode | **PARTIAL** — closed for the preflight process; companion subprocesses still load forged bytecode (W404-03, LOW); and the fix opened W404-01 |
| W403-02 residual text channels | **PARTIAL** — closure_evidence keys, audit freeze, indexed-directory rules work; residue W404-04 (LOW, acknowledged for operator_decisions/) |
| W403-03 binding limited to the import closure | **PASS** |
| W403-04 commit-graph / GIT_DIR / imported modules' git calls | **UNVERIFIED** (unchanged) |
| W402-01, W402-03, W401-RA-01..04, RA-06 | **PASS** (all earlier repros re-run on `dc650b8df`: rejected) |
| W401-RA-05 / 07 / 08 | open on purpose, PARTIAL / INFO |
| **W404-01** `import tempfile` before the sys.path strip | **FAIL** (blocker) |
| **W404-02** repo-root module shadowing under `python -m` / `PYTHONPATH=<repo>` | **FAIL** (MEDIUM, non-documented invocation) |

## 2. Delta

`git diff 2e0ed9c6b dc650b8df`: preflight and its B4 tests, report/handoff/ticket text. Test diff removes no line. Nothing changed under `intergrax/`, `agents/`, `applications/`, `scripts/governance/`, `.gitattributes`, the activation record or `iiac_evidence/`.

## 3. The blocker (W404-01)

Round 4 added `import tempfile` (to compute a pycache prefix) above the line that strips the script directory from `sys.path`. In script mode `sys.path[0]` is `scripts/qualification/`, and `tempfile` pulls in `random`, `shutil`, `weakref`, `functools`... that are not loaded at interpreter start, so they are resolved from the repository first. A planted untracked `scripts/qualification/random.py` (or `shutil.py`) runs before any check. With the documented command and PYTHONPATH unset, `git status` shows only `?? scripts/qualification/random.py`, yet the canonical clone prints `EVOL-X closure preflight: PASS`, exit 0, with B-4/B-5/P3 open (control without the file: FAIL, exit 1). At `2e0ed9c6b` only `os` and `sys` preceded the strip, so this is new. The A-21 check that would report the untracked file runs only after the hijacked code.

## 4. Other results

* **W403-01 replay** (clone, forged unchecked-hash pycs for `iiac_common`, `iiac_activation_gate`, the canonical matrix and `scripts/__init__`): main process ran none of them; FAIL, exit 1. The two companion verifiers did load the forged `iiac_common` (marker lines). No reachable PASS from that after P2 because the companion inputs are sealed; still a gap against the report's "no repository bytecode is read".
* **W403-02:** T6a gitlink in `audit/`, T7b new indexed directory, T8 key in a closure_evidence item: rejected. Still accepted: a file in `operator_decisions/`, and an addendum listed in an edited index of the real `b4_reaudit/` directory (unlisted addendum there is rejected).
* **W403-03:** untracked or modified unrelated files under `scripts/` (including `scripts/dev`) no longer block; any untracked file in `scripts/qualification` still does. 45 s canonical run.
* **Legitimate operations:** clean clones with `core.autocrlf` true / false / input, a clone's linked worktree and the audited worktree: status clean, activation `IN_FORCE`, canonical class, Gate A exactly the two `A-18`, no `A-21/A-22`, no `B-12/B-13`. Modified `README.md` accepted.
* **Gates:** B4 + ruff-exemption suite **489 passed** (481 + 8 new), 0 failed, 21 min 24 s; `tests/unit/governance` **145 passed, 1 skipped (symlinks)**, 13 min 16 s; `verify_iiac_policy_activation --require-in-force` `IN_FORCE`; canonical `b6-readiness` and `formal-closure` FAIL, exit 1, only the two `A-18` in Gate A.

## 5. Threat-model boundary: which capability each finding assumes

| Tier | Capability | Findings | Status at `dc650b8df` |
|---|---|---|---|
| T1 | write to the working tree only (changes `git status` shows) | **W404-01** (new untracked `scripts/qualification/random.py`, documented command); W404-04 (committed addenda) | W404-01 open: **false PASS**; W404-04 harmless |
| T2 | write to ignored / hidden / unbound files (`__pycache__`, `.git/info/exclude`, repo-root files) | W404-02 (repo-root `json.py` under `-m`/PYTHONPATH), W404-03 (companion bytecode); W403-01 main-process bytecode is closed | W404-02 open: false PASS under a non-documented invocation; W404-03 no reachable PASS |
| T3 | write to the git index / git config / refs | skip-worktree, assume-unchanged, fsmonitor, replace refs, grafts | closed (earlier rounds, re-run); commit-graph forgery not exercised |
| T4 | control of the Python environment or process | `sitecustomize` on PYTHONPATH forces PASS (reproduced); also usercustomize, `.pth`, venv, interpreter, predictable temp prefix `evolx-preflight-no-bytecode-cache/<pid>` (not brute-forced) | cannot be closed by the script; only a clean environment on a fresh clone helps |

The honest boundary for the operator: T1 and T2 are in scope of what a fail-closed preflight can and should defend (the guard claims to bind the verdict to HEAD), and W404-01 shows T1 is still breached, by the fix itself. T3 is defended. T4 is not defensible from inside the script; the report's recommendation to run it on a fresh clone in a clean environment is the only mitigation, and it also removes T1/T2 for that run. Recording that boundary is the operator's decision (the report proposes OD-2); I do not take it.

## 6. Not verified

Forged commit-graph, `GIT_DIR`/`GIT_INDEX_FILE`, the imported modules' own git calls (W403-04 unchanged); symlink cases (host cannot create them); brute-forcing the predictable pycache prefix; CI and a Linux container run for the round-4 SHA (not reproduced here). Operator-owned items untouched: W4-F08, OD-1/OD-2, OD-EVOL-12, OD-EVOLX-08, P3, B6, FRZ promotion, EVOL-X closure.
