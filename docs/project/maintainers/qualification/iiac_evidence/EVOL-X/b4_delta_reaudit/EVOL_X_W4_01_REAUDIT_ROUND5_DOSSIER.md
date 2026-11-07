# EVOL-X W4-01 — round-5 delta re-audit dossier

**Auditor session:** `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` (`independent-quality-auditor`)
**Audited code SHA:** `fdb1e7a6215f8eb03bc3ca8d00cce9659dc5c3db` (detached in the auditor worktree) · previous pin `dc650b8df` · `git diff fdb1e7a62 origin/development` outside `docs/` and `governance/tickets/` is empty. (`2f02c92bd`, which I looked at last time, still contains the round-4 code, as the requester said.)
Audit evidence only: nothing fixed, signed, certified, promoted or closed; B-4 not accepted; no operator record touched. Findings: `EVOL_X_W4_01_REAUDIT_ROUND5_FINDINGS.json`; raw output `raw5/`.

## 1. Verdict

**PARTIAL — no blocking finding.** All three targeted findings are closed under independent reproduction and variants. The PARTIAL status is only the LOW residue and the undecided operator boundary.

| Finding | Capability | Verdict |
|---|---|---|
| W404-01 import before the sys.path strip (HIGH) | T1 untracked file in `scripts/qualification` | **PASS** |
| W404-02 repo-root module shadowing | T2 ignored/unbound root file | **PASS** (residue W405-01) |
| W404-03 companions load forged bytecode | T2 ignored `__pycache__` | **PASS** |
| W404-04 free files in `operator_decisions/`, index-edited `b4_reaudit/` | T1 committed text | **PARTIAL** (accepted by the implementer, unchanged) |
| W404-05 interpreter-level hooks | T4 | **PARTIAL** (boundary; reproduced again with a `.pth`; not closable in-script; OD-2 undecided) |
| W403-04 commit-graph, `GIT_DIR`/`GIT_INDEX_FILE`, imported modules' git calls | T3 | **UNVERIFIED** |
| earlier findings W403-01..03, W402-01..03, W401-RA-01..04, RA-06 | | **PASS** (repros re-run) |
| W401-RA-05 / 07 / 08 | | open on purpose, PARTIAL / INFO |
| **W405-01** children inherit `PYTHONPATH` | T2 + `PYTHONPATH=<repo>` | **PARTIAL** (LOW) |
| **W405-02** any untracked root `.py` / package blocks the gates | usability | **PARTIAL** (LOW) |
| **W405-03** `.pth`/sitecustomize force PASS; project venv is git-ignored inside the checkout | T4 / T2 | **UNVERIFIED** as a boundary item (INFO) |

## 2. Delta

`git diff dc650b8df fdb1e7a62`: the preflight, its B4 tests (no removed line), report/handoff/ticket text. Cumulatively since the baseline `5afaad9ea` nothing changed under `intergrax/`, `agents/`, `applications/`, `scripts/governance/`, the activation record, the certification protocol or `iiac_evidence/`; `.gitattributes` is the one file touched (the earlier W4-F01/F06 rules).

## 3. Targeted checks

* **W404-01 (r6 replay).** Documented command, `PYTHONPATH` unset, one planted untracked file per run in `scripts/qualification/`: `random`, `shutil`, `weakref`, `functools`, `json`, `argparse`, `hashlib`, `re`. None executed (no marker); every run FAIL; `git status` showed only the `??` file; control without a file: FAIL, status clean. Statically, only `from __future__`, `os` and `sys` precede the bootstrap block; the bytecode prefix no longer needs `tempfile`. Invocation spelling cannot defeat the strip: a directory junction path, a relative `./script` from inside `scripts/qualification`, and a case-different path all ran safely.
* **W404-02.** Untracked, `.git/info/exclude`-hidden root `json.py`, `random.py`, `runpy.py` under script mode, `python -m`, `PYTHONPATH=<repo>` + script mode and `PYTHONPATH=<repo>` + `-m`: never executed in the preflight process. Import closure of a run: `scripts.governance`, `iiac_activation_gate`, `iiac_common` and the canonical matrix, all from the checkout; no site-packages module; `sys.path` carries no repository or cwd entry. A non-executing run with an untracked root `json.py` reports `A-21 … untracked repository-root module`.
* **W404-03.** Forged unchecked-hash pycs for `iiac_common`, `iiac_activation_gate`, the canonical matrix and `scripts/__init__`, with and without `PYTHONPATH=<clone>`: no forged code ran anywhere, including both companion subprocesses (round 4: they did); FAIL, exit 1.
* **Residue W405-01.** With `PYTHONPATH=<repo root>` the planted root `json.py` does run inside the companions (`PYTHONSAFEPATH` does not remove `PYTHONPATH`); without it they are safe. No reachable PASS (companion inputs are sealed after P2).

## 4. Legitimate operations

Clean clones with `core.autocrlf` true / false / input, a clone's linked worktree and the audited worktree: status clean, activation `IN_FORCE`, class `canonical`, Gate A exactly the two `A-18`, no `A-21/A-22`, no `B-12/B-13`. Accepted despite being dirty: modified `README.md`, untracked or modified files under `scripts/dev` and `tests/`, root `notes.txt`/`scratch.md`. Rejected: any untracked `.py` file or package at the root (even `my_scratch.py`, `tmp_pkg/`) and any untracked file in `scripts/governance` / `scripts/qualification` (W405-02, usability only).

## 5. Gates and tests (clean committed tree, Windows, `core.autocrlf=true`)

| Command | Result |
|---|---|
| B4 + ruff-exemption suite, `-n 6`, `--basetemp=D:/w4a5/bt` | **501 passed**, 0 failed, 18 min 46 s, EXIT=0 (489 + 12 new) |
| `tests/unit/governance -n 4` | **145 passed, 1 skipped (symlinks)**, 9 min 07 s, EXIT=0 |
| `verify_iiac_policy_activation --require-in-force` | `IN_FORCE @ IIAC-2026-11-05.1`, exit 0 |
| canonical `--gate b6-readiness` | FAIL, exit 1, exactly the two `A-18` (B-4, B6-P3); 20 s idle |
| canonical `--gate formal-closure` | FAIL, exit 1; no `A-21/A-22/B-12/B-13` |
| earlier attack families atk1–atk7 on `fdb1e7a62` | all rejected as before; controls pass |

## 6. Threat-model boundary (for OD-2)

Inside the script the guard now holds against: T1 (untracked/modified working-tree files, including the pre-import window), T2 for the preflight process and its sealed companions (bytecode, root and script-directory modules, ignored files), T3 (index flags, fsmonitor, replace refs, grafts, shallow). It cannot hold against T4: a `.pth` file, `sitecustomize`/`usercustomize`, a replaced interpreter, a foreign `PYTHONPATH` entry or the temp directory run before or around the script; I reproduced `.pth` and `sitecustomize` forcing PASS on a canonical clone. Because the project venv is a git-ignored directory inside the checkout, a `uv run` user is exposed to the T2/T4 overlap; the only mitigation is running the preflight from a fresh clone with a clean interpreter. Choosing that as the declared boundary is the operator's decision (OD-2); I do not take it.

## 7. Not verified

Forged commit-graph, `GIT_DIR`/`GIT_INDEX_FILE` and the imported modules' own git calls (W403-04); `usercustomize`; brute-forcing the predictable pycache prefix; symlinks (host cannot create them); CI and a Linux container run for the round-5 SHA. Operator-owned items untouched: W4-F08, OD-1/OD-2, OD-EVOL-12, OD-EVOLX-08, P3, B6, FRZ promotion, EVOL-X closure.
