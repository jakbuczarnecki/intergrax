> **Publication note (2026-11-07).** This is the unedited report of the second independent certifier session `local_e34301ce-67df-40bf-b274-b2e062581541` (started by the recovery session `local_7eeb7387-df63-460d-88e9-be7359430173`), published byte-for-byte below this note. Its verdict was `NOT CERTIFIED` for `fdb1e7a62` because (R1) no PASS/PASS_SCOPED manifest existed yet, (R2) finding CERT-01 (forged ignored bytecode runs under `python -m`) and (R3) operator records were not operator-signed. Status at publication: R1 was resolved by the manifest published at `add7187bc` and the certifier disposition at `7a5476cdc`; R2 is resolved by Amendment 2 of `EVOL_X_OD2_DECISION_RECORD.md` (the `-m` form is not a supported form; the finding cannot be fixed in the script); R3 is covered by the operator's chat ratification recorded in OD-1, OD-2, OD-3 and W4-F01 (delegated, `process_enforced`). The certifier of record for W4-F08 remains `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18`; this report is additional independent evidence, not a second certifier row. Paths below refer to the author's local worktree and are not repository paths. Cite by commit SHA.

# EVOL-X T-W4-01 / B-4 — certification-governance review (report)

**Verdict (single line):**

`NOT CERTIFIED (R1: no PASS/PASS_SCOPED independent-audit manifest is bound to fdb1e7a62 — the only manifest is the round-1 draft and it fails verify_iiac_audit_manifest / verify_iiac_role_separation, so IIAC G2 step 5 and the preflight's own B-4 resolution contract cannot be satisfied; R2: new reproducible in-scope (T2) defect CERT-01 — forged git-ignored bytecode gives exit 0 under `python -m` on the pinned code, which triggers the OD-2 review trigger; R3: operator decisions OD-2 / W4-F01 / OD-1 / W4-F08 are still delegated-by-implementer, not operator-signed)` — for exactly the pinned code `fdb1e7a62` (published head `bec0a0948`, no non-docs change since).

This report is **not** a certification record, not a PASS, not an acceptance of B-4. No file in the audited tree, no record, no ladder, no ticket was edited; nothing was committed, merged or pushed. `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` was not touched by me (the main checkout was never used).

---------------------------------------------------------------------------------------------------

## 0. Session and independence

| Item | Value |
|---|---|
| My session id | `local_e34301ce-67df-40bf-b274-b2e062581541` (desktop session record `claude-code-sessions/.../local_e34301ce-….json`; CLI session `2e09ec3d-8b63-4b66-bf64-eb3d9fb37596`) |
| Role | certification-governance agent (IIAC protocol roles table) |
| Worktree (report only) | `D:\Projekty\intergrax\.claude\worktrees\objective-pasteur-903be7` (branch `claude/objective-pasteur-903be7`, never modified except this git-ignored `.tmp/session/w4-01-certification/`) |
| Verification clone | `D:\Projekty\_wt\intergrax\certw401` (fresh `git clone` from `https://github.com/jakbuczarnecki/intergrax.git`, HEAD `bec0a0948`) |

Independence: my id differs from every id in the task list — implementer `local_4b071fe5…`, auditor `local_e360e892…`, recovery `local_7eeb7387…`, duplicate round-3 `local_cc501aa4…`, certifier-track planner `local_ea0a6280…` — and from the historical ids in implementation report §H (`local_e804da4d…`, `local_4545af5d…`, `local_88e33ba5…`, `local_7cce176e…`, `local_db710787…`) and from the publication/auditor-draft ids (`9e3b9293…`, `cf31a7e9…`, session `epic-bhaskara-0ae5c1`/`local_f55aece0…`). I did not implement, audit or publish any part of this work. Assurance level `process_enforced` (no cryptographic session proof).

**Operator should know:** another session, `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18` ("Independent certifier run for EVOL-X W4-01 (B-4)", started by `local_6ff4e202…`, a sibling of the implementer's title), was started a few minutes before mine for the same task. I did not read its output or coordinate with it. The operator should name exactly **one** certifier session id for the W4-F08 certifier row; two parallel certification runs must not both write records.

## 1. Pins (verified, not trusted)

| Pin | Check | Result |
|---|---|---|
| `origin/development` | `git fetch` in my worktree, fresh clone from GitHub | `bec0a094858724cc3151a9bde7fffeaa2b49b30f` (= expected) |
| Baseline `5afaad9ea…` | ancestor of `fdb1e7a62`, commit exists | yes |
| Code pin `fdb1e7a62` (+ `e00967693`, `9f8b04d2b`, `ca78b1c1f`, `2e0ed9c6b`, `dc650b8df`) | all commits, ancestors of HEAD | yes |
| Non-docs change `fdb1e7a62..bec0a0948` | `git diff --name-only … \| grep -v '^(docs/\|governance/tickets/)'` | **empty** (only report/handoff/OD-2/evidence-package docs) |
| Cumulative non-doc change since baseline | `git diff --name-status 5afaad9ea HEAD -- . ':!docs' ':!governance/tickets'` | exactly: `.gitattributes`, `.github/workflows/unit-tests.yml`, the preflight script, `tests/unit/governance/test_evol_x_real_repo_fresh_clone.py` (A), `tests/unit/qualification/conftest.py` (A), `test_evol_x_closure_preflight_b4.py` |
| P2 anchor `ad0877715…` | `git log --full-history --diff-filter=A -- EVOL_X_IIAC_CERTIFICATION_RECORD.json` (replace objects off) | single result `ad0877715ed079ce920ff9d925fda03555b2c04e` = script constant (`:158`) |
| Replace refs / grafts / shallow in the clone | `git replace -l`, `.git/shallow`, `.git/info` | none |

## 2. Forbidden-area check (item 4)

`git diff --name-status 5afaad9ea bec0a0948` on `intergrax/ agents/ applications/ scripts/governance/`, `IIAC_POLICY_ACTIVATION_RECORD.json`, `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`, `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json`: **empty**. `iiac_evidence/`: 116 files **added** (all under the new sibling directory `iiac_evidence/EVOL-X-W4-01-REAUDIT/`, the byte-preserved auditor package, bec0a0948), **0 modified, 0 deleted** — no existing evidence changed. Original binding and delegation record are byte-identical to baseline (sha256 `584c504e…`, `2120852e…`).

## 3. Independent reproduction (item 1) — logs in `logs/`

Environment: Windows 11, Git Bash, `core.autocrlf=true`, `uv sync --extra dev-ci --frozen`, `-n 4`, no `-p no:cacheprovider`. Clean venv for the OD-2 runs: `uv venv --python 3.12` **outside** the clone (Python 3.12.11, jsonschema 4.26.0 + locked deps, no `.pth`/`sitecustomize`/`usercustomize` — checked), `PYTHONNOUSERSITE=1`.

| Check | Result | Log |
|---|---|---|
| B4 module + ruff-exemption (`--basetemp=D:/cw/b4`) | **501 passed**, 19 min 28 s | `pytest_b4_short.txt` |
| same with a **long** basetemp (`…/_scratch/certw401/tmp/b4`, 54 chars + `popen-gwN`) | 79 failed / 414–422 passed, **reproducible**, all fixture-repo tests; cause: Win32 MAX_PATH in the fixture chain (the longest evidence file name is the certification handoff — the failing message is “…HANDOFF_RECORD_EVOL_X_CERTIFICATION.md … absent”); passes with `D:/cw/…`. **Not a code defect**, but note: a ~54-char `--basetemp` is already too long on Windows (use ≤ ~15 chars) | `pytest_b4.txt`, `pytest_b4_lf.txt` |
| `tests/unit/governance` (`--basetemp=D:/cw/g`) | **145 passed, 1 skipped** (symlink creation unavailable on this host), 8 min 12 s | `pytest_governance.txt` |
| `verify_iiac_policy_activation --require-in-force` | exit 0, `IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8…` — **only with `PYTHONPATH=<clone root>`**; without it exit 1 (`ACT-N10 … No module named 'scripts'`, both companions) | `activation.txt`, `activation_no_pythonpath.txt` |
| activation in extra clones `core.autocrlf=false` and `=input` | `IN_FORCE`, handoff sha256 `96fcf995…` in both (and in the autocrlf=true clone) | `autocrlf_clones.txt` |
| `--gate b6-readiness` (canonical, clean venv) | FAIL, exit 1, **exactly two `A-18`** (B-4, B6-P3); no A-19/A-21/A-22/B-12/B-13 | `b6_readiness.txt` |
| `--gate formal-closure` | FAIL, exit 1 (Gate A same two A-18; Gate B: B-00/01/02/06×10/07×2/09×3; no B-12/B-13) | `formal_closure.txt` |
| `verify_iiac_audit_manifest` / `verify_iiac_role_separation` on the P2 manifest `iiac_evidence/EVOL-X/certification/iiac_audit_session_manifest.EVOL-X.json` | PASS / PASS | `manifest_verifiers.txt` |
| same two verifiers on the **only W4-01 manifest** `iiac_evidence/EVOL-X-W4-01-REAUDIT/iiac_audit_session_manifest.EVOL-X-W4-01.reaudit_draft.json` | **FAIL / FAIL** (missing `audit_scope_manifest_digest`, `test_matrix_digest`; `evidence_publication_sha`, `certification_publication_sha` = `PENDING_CERTIFIER`) | `manifest_verifiers.txt` |
| ruff | `ruff check` + `ruff format --check` on the 4 changed code files: clean. Repo-wide `ruff check .` reports 21 726 findings and `format --check` 6 541 files (legacy; I did not compare with the baseline, so I make no claim about it) | `ruff.txt` |
| CI (`gh run list`) | **No workflow run exists for `fdb1e7a62`, `e00967693`, `fb939835b` or `bec0a0948`** themselves. Nearest: run `42260796255` on `d75d905cc` (a docs-only descendant of `fdb1e7a62`, code identical) — Regression gate **success**: `governance-fast` 146 passed, `evol-x-b4-preflight` **501 passed**, `ci-smoke` 1882 passed. ENT-AUDIT-X workflow still fails on `development` (pre-existing, unrelated, per implementer report §L; not re-investigated by me) | `gh` output in this session |

## 4. Verification of the named controls (item 2)

| Control | Evidence | Verdict |
|---|---|---|
| DA-01 pinned P2, cross-check with first commit adding the record, no declared-field fallback, no canonical override | constant = first-add commit (§1); fixture run **without** `--p2-sha` fails (F13: A-18/A-19), attacker anchors `--p2-sha`=widened commit or previous commit fail (F1c/F1d); canonical `--p2-sha HEAD`, `--p2-sha HEAD~1`, `--evidence-root` → rejected with A-19 / A-01 “override is forbidden” (E2) | **PASS** |
| DA-02 / RA-B4-05 exact scope | canonical P2 vs HEAD: verdict `PASS_SCOPED` both; limitations equal (LIM-01..LIM-10 + F-C-01, F-C-02, implementation-session exception, evidence-only statement = 14 entries); FRZ dispositions equal; only `certification_publication_sha(_binding)` differ (the `ca2b14782` binding). Fixture attacks: verdict widening, LIM widening in manifest, extra key, dropped key, FRZ disposition edit, notes, binding text, padded SHA, NaN, duplicate key, extra file in certification dir — all rejected on both gates (F1a–F2n) | **PASS** |
| RA-B4-04 B6 not acceptable by flags / self-referential manifest | genuine closed fixture passes both gates (F7 control); flags + self-referential SHA without B6 record → Gate B fails (B-04/B-05); B6 record widened after publication → B-05; certification-record B6 SHA = HEAD → B-02; negated decision scope → B-07 (F7a–F7d). Gate A stays “readiness”, so it passes in these cases by design; Gate B holds | **PASS** |
| DA-09 UTF-8 republication, `-text` | original binding contains `0x97` (not UTF-8) and is byte-identical to baseline; `.utf8.json` decodes, all fields equal to the cp1252 decoding; `git check-attr text` = unset for the binding, its republication and the delegation record | **PASS** |
| W4-F01 option C (`text eol=crlf` on the activation handoff) | `git check-attr` = `text: set, eol: crlf`; working-tree sha256 `96fcf995…` (= `handoff_digest` in the activation record) in `autocrlf` true / false / input clones; blob stays LF (`3b7e901f…`); activation `IN_FORCE` in all three; gate and record untouched | **PASS (technical)**; the *decision* to ratify C is the operator's (see §8) |

## 5. Adversarial probes (item 3) — all run in throw-away clones `certprobe0..3` (no canonical checkout), harnesses in `probes/`

Probe counts: canonical clone 37 (C1) + 40 (C2/C3) + 36 (C4) + 13 (C5–C8) + extra E1–E4 (≈20); fixture repos 67 + 14. “Attack effect” = planted code ran or the gate did not fail closed. Flagged by the harness: C2/C3 = 11 (10 are planted root modules executing **inside the companion verifiers** = W405-01, one is CERT-01); C1, C4, C5–C8: 0; fixture set 1: 15 flagged, all explained (Gate-B-only inputs/flags: Gate B failed as designed; see table); fixture set 2: 0.

| Finding | Probes | Result |
|---|---|---|
| W401-RA-01 replace refs / grafts / shallow | replace P2→parent / →HEAD / →forged commit with widened manifest; `.git/info/grafts`; `.git/shallow`; `GIT_REPLACE_REF_BASE=refs/nothing/`; `GIT_OBJECT_DIRECTORY` | all rejected (A-22 / B-13) |
| W401-RA-02 lifecycle/free-text channels | 12 committed mutations of the certification record + extra file + duplicate key | all rejected on both gates except two observations: `stage_closure_status`=`CLOSED` or `b6_acceptance`=`ACCEPTED` (enum-valid values) keep **Gate A** PASS and are caught by **Gate B** (B-10 / B-02). INFO, by design (“Gate A PASS never implies closure”) |
| W401-RA-03 / W402-01 working-tree & index flags | append and same-size byte flip of 12 canonical inputs + 9 fixture inputs, each also hidden by `--skip-worktree` and `--assume-unchanged`; original W402-01 scenario (widened row committed, correct bytes only in worktree); `core.fsmonitor` stub; `GIT_INDEX_FILE` / `GIT_DIR` / `GIT_WORK_TREE` / `GIT_OBJECT_DIRECTORY` / `GIT_CEILING_DIRECTORIES`; combination skip-worktree + alt index | **all caught** (A-21 for Gate-A inputs, B-12 for register/FRZ checklist which only Gate B reads). Edits of the preflight script itself, `iiac_common.py`, `iiac_activation_gate.py`, the canonical matrix, `.gitattributes`, the activation record, binding, OD-1 record: all caught |
| W401-RA-04 `--full-history` | late blocker B-9 published on a side branch, hidden with `git merge -s ours` | rejected (A-18) |
| W402-02/03/04 | untracked file in each evidence directory (audit, certification, evidence root, operator_decisions, reaudit); corrupted `commit-graph` (preflight ignores it, same result) | rejected (A-21) / unchanged; **a forged (structurally valid) commit-graph was not built → W403-04 stays UNVERIFIED** |
| W403-01 / W404-03 forged `.pyc` (UNCHECKED_HASH) in ignored `__pycache__` | `iiac_common`, `iiac_activation_gate`, canonical matrix, `scripts/__init__`, `scripts/governance/__init__`, both companions, `scripts/qualification/__init__` | **script-path form: all rejected, no forged code ran.** **`python -m` form: see CERT-01 below — ATTACK EFFECT** |
| W404-01 planted stdlib shadows next to the script | 30 untracked `scripts/qualification/<stdlib>.py` (random, shutil, json, tempfile, argparse, hashlib, re, functools, weakref, subprocess, posixpath, unicodedata, collections, dataclasses, datetime, pathlib, typing, os, sys, importlib, types, enum, io, stat, ntpath, genericpath, encodings, codecs, abc, warnings), script mode; 3 via `-m`; `scripts/` and `scripts/governance/` json/random | none executed; all FAIL with A-21 |
| W404-02 ignored (`.git/info/exclude`) repo-root modules | json, random, runpy, hashlib, argparse, tempfile, re, scripts; script mode, `-m`, with/without `PYTHONPATH=<clone>` | **not executed in the preflight process** (E1 records the process: only `verify_iiac_audit_manifest.py` / `verify_iiac_role_separation.py` ran them). With `PYTHONPATH=<clone>` json/random/re/argparse/tempfile **do execute inside the companions** = W405-01 (confirmed). **New aggravation, E1 “selfdelete”:** a planted module that removes itself and its `exclude` entry from inside a companion makes the later untracked-root scan see nothing — **`A-21` is then not reported** (rc 1 only because of the real A-18). See CERT-02 |
| W405-02 | untracked root `my_scratch.py` → A-21 (confirmed); `notes.txt`, `scripts/dev/foo.py` accepted | confirmed, usability only |
| T4 boundary (OD-2) | foreign `PYTHONPATH` with `json.py`; `.pth` in site-packages; `sitecustomize` — all give exit 0 / forged pass | reproduced, **out of scope by OD-2**; relevant only because it shows why the fresh-clone/clean-interpreter procedure is mandatory |
| Canonical sanity | after every probe `git status --porcelain --ignored` was restored to clean (except pycache residue noted below) | ok |

### CERT-01 (new, MEDIUM, tier T2 — blocks certification): forged ignored bytecode is executed under `python -m`

Reproduction (`probes/probe_m_pyc.py`, `logs/probe_m_pyc.txt`, fresh clone at `bec0a0948`, clean venv, `PYTHONPATH=<clone>`):
1. Control `python -m scripts.qualification.evol_x_closure_publication_preflight --gate b6-readiness` → exit 1, two `A-18`.
2. Plant `scripts/__pycache__/__init__.cpython-312.pyc` (UNCHECKED_HASH, git-ignored; `git status --porcelain` empty) → same command → **exit 0, forged code ran in the preflight process** (marker written). The same planted file in the **script-path form** → exit 1 (bootstrap protects it).
3. Same with a forged `scripts/qualification/__pycache__/evol_x_closure_publication_preflight.cpython-312.pyc` → `-m` exit 0.
4. `python -m scripts.governance.verify_iiac_policy_activation --require-in-force` with the forged `scripts/__init__` pyc → exit 0 (same class, in an already-certified material path; not part of this correction).
Cause: with `-m` the interpreter imports package `scripts` (and loads the module's cached bytecode) **before** the bootstrap in the script can set `sys.dont_write_bytecode` / `pycache_prefix`. The clean `-m` control run also *writes* `.pyc` into the repository (`scripts/__pycache__/__init__…`, `scripts/qualification/__pycache__/…`), contradicting the stated “no bytecode is read or written in the repository”.
Why it counts: OD-2 puts T2 and “the documented command forms (script path **and** `python -m scripts.qualification…`)” in scope; round-5 W404-03 is recorded as PASS, but the auditor’s replay (`raw5/r3_pyc_replay.txt`) used only the script form. The OD-2 re-open trigger (“T1–T3 finding of MEDIUM or higher reproduced against the pinned SHA”) is met. Exploitability on the real tree is limited to a false *printed* PASS (the canonical chain cannot actually pass while B-4/B6-P3 are open) and is closed by the OD-2 fresh-clone procedure — but that is a procedure, not the code property the corrective claims.
Suggested correction (for the implementer, not done by me): in `-m` form refuse to run (check `__spec__`/`__package__` at the top and exit with an instruction to use the script form), or document `-m` as unsupported; never claim `-m` as a defended form.

### CERT-02 (new, LOW, T2 + `PYTHONPATH=<repo>`): A-21 can be evaded by a self-deleting planted root module
Extends W405-01. Only reachable when the companions run with the repository root on `PYTHONPATH` (the form the repo’s own tests and the working command use — see CERT-03). Cannot change the verdict of the canonical run (sealed inputs, open blockers) but makes the “untracked root module → A-21” claim non-absolute. Suggested: strip `PYTHONPATH` from the children and give the companions their own `sys.path` setup, or run the untracked-root scan after the companions *and* before them.

### CERT-03 (INFO, documentation): OD-2’s “no `PYTHONPATH`” procedure fails closed
Following OD-2 literally (clean venv, **no `PYTHONPATH`**) the activation gate exits 1 (`ACT-N10 … No module named 'scripts'`) and both preflight gates FAIL with `A-01` — because the companions need `<repo>` on `PYTHONPATH`. The working certification command therefore sets `PYTHONPATH=<fresh clone root>`. OD-2 should say so (the fresh clone makes W405-01 moot).

### Also observed
* Clean `-m` runs leave `scripts/__pycache__` residue (ignored) — fix with CERT-01.
* An in-repo `.venv` (`uv run`) is the T2/T4 overlap noted by the auditor (W405-03): not used for the OD-2 runs above.

## 6. Per-finding table (nothing inherited from the auditor’s numbers; “my evidence” = items above)

| Finding | Auditor r5 | My independent result | Blocks certification? |
|---|---|---|---|
| DA-01 | — | PASS | no |
| DA-02 / RA-B4-05 | — | PASS | no |
| RA-B4-04 | — | PASS | no |
| DA-09, W4-F01 (technical) | — | PASS | no |
| W401-RA-01 | PASS | PASS (6 variants) | no |
| W401-RA-02 | PASS | PASS (+INFO: Gate A tolerates enum-valid `CLOSED`/`ACCEPTED`) | no |
| W401-RA-03 | PASS | PASS (≈70 probes) | no |
| W401-RA-04 | PASS | PASS (`-s ours`) | no |
| W401-RA-06 | PASS | not re-tested beyond the drift test inside the 501 | no |
| W402-01..04 | PASS | PASS; W402-04 forged graph not built | no |
| W403-01 | PASS | PASS in script form; **FAILS in `-m` form → CERT-01** | **yes** |
| W403-02 | PASS (residue W404-04) | PASS; residue W404-04 confirmed | no |
| W403-03 | PASS | PASS (legit untracked files accepted) | no |
| W403-04 | UNVERIFIED | still UNVERIFIED (corrupt graph only) | no (INFO), see §7 |
| W404-01 | PASS | PASS (30 modules, script/`-m`) | no |
| W404-02 | PASS | PASS for the main process; CERT-02 aggravation | no (LOW) |
| W404-03 | PASS | PASS in script form; **FAIL in `-m` form (CERT-01)** | **yes** |
| W404-04 | PARTIAL | confirmed: free file in `operator_decisions/` keeps Gate A PASS | no |
| W404-05 / W405-03 | PARTIAL / INFO | reproduced (.pth, sitecustomize, foreign PYTHONPATH) | no — procedural by OD-2 |
| W405-01 | PARTIAL LOW | confirmed (+CERT-02) | no (LOW) |
| W405-02 | PARTIAL LOW | confirmed (usability) | no |
| W401-RA-05 / 07 / 08 | open | RA-07 confirmed (fixture PASS exits 0, F0); RA-05, RA-08 not re-tested (design/decision items) | no |

## 7. Residual-risk judgement (item 5)

* **W404-04** (free text in `operator_decisions/`, edited `b4_reaudit/` index): descriptive channels no gate reads; cannot widen scope; LOW; accepted — does not block.
* **W404-05 / W405-03 (T4, OD-2)**: not closable inside a script. Does not block *iff* the operator confirms OD-2 and the certification run uses a fresh clone + clean interpreter (done by me; see §3). Note that my clean run needed `PYTHONPATH=<clone>` (CERT-03).
* **W403-04**: forged-but-valid commit-graph, `GIT_DIR`/`GIT_INDEX_FILE` against the *imported* material modules’ own git calls: I exercised env redirection (no effect on the preflight’s verdict) and a corrupted graph, but **not** a structurally valid forged graph. Unverified; INFO; no blocking evidence either way. Imported modules are material IIAC paths and unchanged.
* **W401-RA-05/07/08**: RA-05 (negation deny-list) fail-closed; RA-07 (exit 0 for a fixture PASS) harmless as long as callers use `--repo`-less canonical runs; RA-08 (pin in an ungoverned file) is the OD-1 matter. None blocks on its own.
* **W405-01/02**: LOW; W405-01 grows to CERT-02 but stays LOW; W405-02 usability.
* The **blocking** items are R1 (procedure), CERT-01 (code), R3 (operator-owned records).

## 8. Why “NOT CERTIFIED” (reasons in order of weight)

R1 — **No eligible independent-audit verdict.** IIAC protocol G2 step 5: the certifier “validates manifest + `verify_iiac_*` + role separation before any PASS publication”; the certifier role is forbidden to replace the audit with a checklist or override fail-closed. The auditor’s round-5 overall verdict is **PARTIAL**; the only manifest (`…reaudit_draft.json`, round 1, audited `9f8b04d2b`, `CONDITIONAL`, `manifest_status: AUDITOR_DRAFT_NOT_A_PASS`) fails both verifiers, binds the wrong SHA and has `PENDING_CERTIFIER` fields. The preflight (`_reaudit_manifest_errors`) accepts a B-4 resolution only with a re-audit manifest verdict in `{PASS, PASS_SCOPED}`, no `manifest_status`, three distinct sessions, `closure_publication_sha` a strict ancestor of HEAD, etc. Writing `RESOLVED_WITH_EVIDENCE` now would either fail Gate A (A-18) or require me to convert PARTIAL into PASS — which the rules forbid. A round-6 audit (or the auditor’s explicit PASS_SCOPED manifest bound to the final code) is required.
R2 — **CERT-01** (MEDIUM, in-scope T2 tier, reproduced on the pinned code, hits the OD-2 review trigger).
R3 — **Operator-owned records are not operator-signed:** OD-1, OD-2 and the W4-F08 signature line were written by the implementer “by delegation” from a chat sentence (“Udzielam ci autoryzacji na wszystkie decyzje”); the ticket `Approved:` lines cite a mission text. The IIAC contract treats such entries as `process_enforced`; I take them as *recorded* but not as operator confirmation of OD-2/OD-1/W4-F01/W4-F08.
Not a reason: the technical side otherwise holds — B4 501 passed, CI green on the docs-only descendant, canonical gates fail exactly as designed, DA-01/02/09, RA-B4-04 and W4-F01 verified, every round-1..5 repro rejected in the script-path form.

## 9. Open operator decisions (I decide none of them)

1. **OD-2** (threat-model boundary): confirm/overrule `EVOL_X_OD2_DECISION_RECORD.md`; its own re-open trigger is now met by CERT-01; amend the procedure for CERT-03 (`PYTHONPATH=<fresh clone>`).
2. **W4-F01**: ratify option C (draft record untracked in the main checkout; quotes stale pins) — technically verified by me, decision is the operator’s; IIAC OD-01 is not W4-F01 approval.
3. **OD-1 / W4-F08**: confirm or overrule the delegated entries; enter **one** certifier session id (see the parallel session `local_dbad2215…` above).
4. **Residual LOW acceptance**: W404-04, W405-01/02, CERT-02, RA-05/07/08, W403-04 (unverified).
5. Untouched and undecided by me: OD-EVOL-12, OD-EVOLX-08, P3, B6 acceptance, FRZ promotion, EVOL-X closure.

## 10. Proposed B-4 status transition — **DO NOT APPLY** (preparatory skeleton only)

Applicable only after (a) a PASS/PASS_SCOPED re-audit manifest for the final code SHA exists and passes `verify_iiac_audit_manifest` + `verify_iiac_role_separation`, (b) CERT-01 is fixed and re-audited. Shape required by `resolution_errors` / `_reaudit_resolution_errors` in the preflight (every field is gate-checked; placeholders in `<>`):

```diff
--- a/docs/project/maintainers/qualification/iiac_evidence/EVOL-X/certification/EVOL_X_IIAC_CERTIFICATION_RECORD.json
+++ b/…/EVOL_X_IIAC_CERTIFICATION_RECORD.json
   "open_blockers_for_closure": [
-    "B-4 legacy closure preflight requires independent corrective implementation and re-audit",
     "B-5 OD-EVOL-12 / OD-EVOLX-08 authority confirmation or signed rows",
     "B6 acceptance (P3), FRZ promotion (P4) and formal closure (P5) not authorized in this publication"
   ],
+  "resolved_blockers_for_closure": [
+    {
+      "blocker_id": "B-4",
+      "status": "RESOLVED_WITH_EVIDENCE",
+      "baseline_text_sha256": "<sha256 of the B-4 text above>",
+      "opened_at_sha": "ad0877715ed079ce920ff9d925fda03555b2c04e",
+      "resolution_kind": "INDEPENDENT_REAUDIT",
+      "closure_publication_sha": "<evidence publication commit, strict ancestor of HEAD>",
+      "resolved_implementation_sha": "<final corrected code SHA>",
+      "closure_evidence_index_path": "<own dir, not audit/ or certification/>/EVIDENCE_INDEX.json",
+      "closure_evidence_index_digest": "<sha256>",
+      "reaudit_manifest_path": "<…>/iiac_audit_session_manifest.EVOL-X.<…>.json",
+      "reaudit_manifest_digest": "<sha256>",
+      "certifier_disposition": {
+        "disposition": "ACCEPTED",
+        "certifier_agent_id": "certification-governance-agent",
+        "certifier_session_id": "<the one named certifier session>",
+        "policy_version": "IIAC-2026-11-05.1",
+        "audit_report_digest": "<= manifest>", "findings_digest": "<= manifest>"
+      }
+    }
+  ],
```
Plus a dedicated publication commit (closure rule EBH-7: not in the commit that first creates a closed claim). Not written, not committed.

## 11. Not run / could not verify

* A Linux run (no container); CI evidence is the docs-only descendant `d75d905cc`, not the pinned SHAs themselves.
* A structurally valid forged commit-graph (W403-04); `usercustomize`; brute-forcing the pycache prefix; symlink cases (host cannot create them; the governance suite skips 1 test).
* Baseline ruff comparison; the ENT-AUDIT-X workflow failure.
* I did not re-read the whole 2 755-line preflight line by line; my assurance comes from the test suites, the CI evidence and the probes above.

## 12. Leftovers for the operator to delete (I ran no recursive delete)

`D:\Projekty\_wt\intergrax\certw401`, `certprobe0..3`, `certac_false`, `certac_input`, `_scratch\certw401` (venvs `cleanvenv`, `t4venv`, fixtures `fx`, markers, tmp), `D:\cw` (short basetemp: `b4`, `g`), and this worktree `objective-pasteur-903be7` (branch `claude/objective-pasteur-903be7`, only this git-ignored `.tmp` folder differs).
