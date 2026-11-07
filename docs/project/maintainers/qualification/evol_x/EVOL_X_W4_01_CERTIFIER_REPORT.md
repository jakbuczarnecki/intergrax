# EVOL-X W4-01 — independent certifier report (B-4)

**Ticket:** [`T-IGX-019-A`](../../../../../governance/tickets/T-IGX-019-A.md) · **Policy:** `IIAC-2026-11-05.1` · **Date:** 2026-11-07
**Certifier session:** `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18` (record this id in the W4-F08 attestation record). A third session: not the implementer `local_4b071fe5-82d0-48bc-b524-146a3daf5c45`, not the auditor `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9`. It started from the task text only and used none of their context.
**Code under certification:** `fdb1e7a6215f8eb03bc3ca8d00cce9659dc5c3db` (final pin, report §R). Also exercised at `origin/development` = `bec0a094858724cc3151a9bde7fffeaa2b49b30f`.
**This report is a recommendation.** It does not accept B-4, edit the certification record or the blocker list, sign anything for the operator, promote a FRZ status, execute B6 or close EVOL-X. Nothing was pushed.

## 1. Verdict

**ACCEPT_WITH_CONDITIONS** — I recommend that the operator accept the corrective at `fdb1e7a62` as sufficient to treat blocker **B-4** as *technically resolved*, on the conditions in §6. I found **no blocker** in the declared boundary (attacker tiers T1–T3, OD-2). Every run in a fresh clone reproduced the auditor's numbers and the expected two-blocker output; my own attacks were all rejected.

Why not a plain `ACCEPT_B4`: (1) the procedural conditions of §6 are not yet met (OD-1/OD-2 were authored by the implementer, not by the operator; OD-2's literal environment wording cannot be satisfied; the real B-4 resolution publication has never been exercised on the real chain); (2) the evidence is one machine class (Windows, one host) plus CI. Why not `REJECT`: nothing I ran contradicts the corrective, and the open items are LOW/INFO with a stated owner.

Accepting B-4 does **not** make the canonical gates pass: `--gate b6-readiness` stays FAIL until the operator publishes the B-4 resolution *and* P3, which this report neither does nor authorises.

## 2. Environment (OD-2 procedure) and commands

| Item | Value |
|---|---|
| Host | Windows 11 Home 10.0.26200, git 2.46.0.windows.1, system `core.autocrlf=true` (Git for Windows default) |
| Clone | `git clone --no-local https://github.com/jakbuczarnecki/intergrax.git D:/w4c/clone`, then `git checkout fdb1e7a62` (detached; `git rev-parse HEAD` = `fdb1e7a6215f8eb03bc3ca8d00cce9659dc5c3db`; `refs/replace` empty; status clean before and after every run) |
| Venv A ("full") | `UV_PROJECT_ENVIRONMENT=D:/w4c/venv uv sync --extra dev-ci --frozen --python 3.12` run in the clone → venv **outside** the clone, CPython 3.12.11 (uv-managed), `pytest 8.4.2`. `site-packages` holds exactly one `.pth`: `_editable_impl_intergrax_ai.pth`, content `D:\w4c\clone` (the clean clone itself); no `sitecustomize`/`usercustomize`. |
| Venv B ("min") | `uv venv --python 3.12 D:/w4c/venv_bare` + `jsonschema==4.26.0 attrs referencing rpds-py jsonschema-specifications typing-extensions` only; no `.pth`, no editable install |
| Process env | `env -u PYTHONPATH -u PYTHONSAFEPATH -u PYTHONDONTWRITEBYTECODE` (the host shell exports `PYTHONDONTWRITEBYTECODE=1`; unset for the gate and test runs). No other `PYTHON*`, `PIP*`, `UV_*`, `GIT_*` variable except `GIT_EDITOR=true`. |
| Scratch clone | `D:/w4c/atk` (`git clone --no-local D:/w4c/clone`) for attacks, never the test clone |

Commands (all run from the clone root, results verbatim in §3; raw logs kept in the shared notes folder `W4-01_certifier_logs/`):

```text
PY=D:/w4c/venv/Scripts/python.exe
$PY scripts/qualification/evol_x_closure_publication_preflight.py --gate b6-readiness
$PY scripts/governance/verify_iiac_policy_activation.py --require-in-force
$PY scripts/qualification/evol_x_closure_publication_preflight.py --gate formal-closure       # extra, to look for A-21/A-22/B-12/B-13
$PY -m pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py -q -n 6 -o cache_dir=D:/w4c/pcache --basetemp=D:/w4c/bt
$PY -m pytest tests/unit/governance -q -n 4 -o cache_dir=D:/w4c/pcache --basetemp=D:/w4c/bg
```

(`-o cache_dir=…` only keeps `.pytest_cache` out of the clone; a first attempt with `-p no:cacheprovider` was rejected by pytest with `Unknown config option: cache_dir` and ran nothing — not a result.)

## 3. Results

| # | Check | Result |
|---|---|---|
| 1 | `--gate b6-readiness` at the pin (venv A) | **FAIL, exit 1, exactly two `A-18`**: `B-4 legacy closure preflight requires independent corrective implementation and re-audit` and `B6 acceptance (P3), FRZ promotion (P4) and formal closure (P5) not authorized in this publication`. No other Gate A code; policy mode `IIAC`; `repository: canonical`. Output is byte-identical to the auditor's `raw5/canonical_b6_readiness.txt`. |
| 2 | same at `origin/development` `bec0a0948` | identical output, exit 1 |
| 3 | `verify_iiac_policy_activation.py --require-in-force` at the pin and at `bec0a0948` | `IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`, exit 0 |
| 4 | `--gate formal-closure` | FAIL, exit 1; 0 occurrences of `A-21`/`A-22`/`B-12`/`B-13` (Gate B failures are the expected B6/FRZ/OD items, listed in the auditor's raw) |
| 5 | `test_evol_x_closure_preflight_b4.py` (pin, venv A, 20-core host) | **493 passed in 1638.93 s (27:18), exit 0.** The auditor's 501 = these 493 + the 8 tests of `test_evol_x_b4_ruff_evidence_exemption.py` (I collected 8): consistent. |
| 6 | `tests/unit/governance` (pin, venv A) | **145 passed, 1 skipped in 784 s (13:04), exit 0** (skip: symlink creation unavailable on the host, `test_iiac_activation_gate.py:439`; same as the auditor). Includes the real-repository fresh-clone tests (autocrlf true/false/input). |
| 7 | `git diff --stat fdb1e7a62 bec0a0948 -- scripts tests intergrax` | empty (documents only: 120 files, +3311, all added) |
| 8 | Minimal clean env (venv B, no editable install, no `PYTHONPATH`) | **fail-closed**: `A-01 … ACT-N10 companion … ModuleNotFoundError: No module named 'scripts'` (and `jsonschema` before it was installed); `python -m scripts.governance.verify_iiac_policy_activation` from the root fails the same way. With `PYTHONPATH=D:/w4c/clone` the venv-B run gives exactly the two `A-18`. See finding C-01. |
| 9 | Linux/CI | not reproduced by me (no Linux run). `gh` shows run `42260796255` ("Regression gate", push `development`, head `d75d905cc`, code-identical to the pin: `git diff fdb1e7a62 d75d905cc -- . ':!docs' ':!governance/tickets'` is empty): `governance-fast`, `evol-x-b4-preflight`, `governance-policy-pin`, `ci-smoke` all **success**. `Enterprise audit (ENT-AUDIT-X)` is red on `development` since 2026-11-05 (before this corrective: runs for `477017f15`…`2ae101411`); unrelated to this change but still a red gate. |

### 3.1 Independent checks requested in the task

* **No material governance path changed.** `MATERIAL_GOVERNANCE_PATHS` (read from `scripts/governance/iiac_activation_gate.py` at the pin) = `scripts/governance/{verify_iiac_policy_activation,iiac_activation_gate,iiac_common,verify_iiac_audit_manifest,verify_iiac_role_separation}.py` and `…/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`. `git diff --name-status 5afaad9ea fdb1e7a62 -- scripts/governance <protocol> IIAC_POLICY_ACTIVATION_RECORD.json IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json intergrax agents applications` is **empty**; the same holds up to `bec0a0948`. All non-docs changes baseline→pin: `.gitattributes`, `.github/workflows/unit-tests.yml`, the preflight script, its B4 test + a new `tests/unit/qualification/conftest.py`, a new governance fresh-clone test, and the ticket — none is material. No existing file under `iiac_evidence/` was modified (only additions).
* **P2 pin** `P2_CERTIFICATION_PUBLICATION_SHA = ad0877715ed079ce920ff9d925fda03555b2c04e` equals `git log --full-history --diff-filter=A` for `EVOL_X_IIAC_CERTIFICATION_RECORD.json` ("Publish EVOL-X IIAC certification evidence (P2, evidence-only)") and is an ancestor of HEAD.
* **Auditor package integrity.** All **115** files listed in `PUBLICATION_PROVENANCE_NOTE.md` hash to the listed SHA-256 (checked on my worktree at `bec0a0948`, a `-text` path, with a script of my own); no file missing and no extra file besides the note (116 tracked). This proves the copy equals what the note says, not that the auditor's runs were honest; I re-ran the key ones (rows 1–6) and got the same numbers.
* **Tests not weakened.** Baseline→pin the B4 test file has 1843 additions / 16 deletions; I read the deleted lines: fixture-helper signature changes and two assertions restated (`Accepted limitations` now computed by an independent restatement of the `<id>:<sha256>` contract). No assertion was dropped.
* **Residual-claim spot checks in the code:** RA-07 confirmed (`_print`/`main` return 0 for a passing gate even when `_repository_class` is `fixture`; the class is only "does `--repo` resolve to the checkout that holds the script").

### 3.2 My own attacks (scratch clone `D:/w4c/atk` at the pin; venv A; script `D:/w4c/probes2.sh`)

All use the documented command form. Result "rejected" = gate FAIL with the stated code and no marker file written.

| Probe | Capability | Result |
|---|---|---|
| P0 control | — | exactly the two `A-18` |
| P1 untracked `scripts/qualification/random.py` writing a marker | T1 | rejected `A-21 … random.py [untracked]`, marker **not** written |
| P3 uncommitted edit emptying `open_blockers_for_closure` of the certification record | T1 | rejected `A-13` digest mismatch, differs from committed bytes, `A-18 blocker lifecycle … B-4 disappeared` |
| P4 same edit + `skip-worktree` + `assume-unchanged` (status clean) | T3 | rejected, same codes |
| P5 same edit + `.git/info/attributes` clean filter that prints HEAD content | T3 | rejected, same codes |
| P6 extra file in `…/certification/extra/` hidden by `.git/info/exclude` | T2 | rejected `A-13 unindexed evidence … extra/note.txt` |
| P2 root `json.py` hidden by `.git/info/exclude`, `PYTHONPATH=<clone>` | T2 + env | **verdict FAIL (`A-01`, `AttributeError: module 'json' has no attribute 'loads'`), but the planted module executed inside both ACT-N10 companions** (marker written twice). Reproduces W405-01. It cannot make a pass: the preflight process itself never ran it and a crashing companion yields FAIL. |
| P7 `GIT_DIR=<other clean clone>` | T4 | **not rejected** (gate ran against the other, identical repository: two `A-18`). Shows `GIT_DIR` is outside the guarded variables; consistent with W403-04/T4. I did not build a doctored second repository. |
| P8 `PYTHONSTARTUP` | T4 | not executed (non-interactive); informational |

A first pass of P5–P9 was invalid: my own reset helper left a `skip-worktree` flag (git ignores combined `--no-skip-worktree --no-assume-unchanged`) so later probes inherited the P4 edit — the gate correctly kept flagging it (`A-13`), which is itself a small positive control. The results above are from the corrected run; clone state was verified clean afterwards.

## 4. Findings of this certification

| ID | Sev | Finding | Impact on B-4 |
|---|---|---|---|
| **C-01** | LOW (procedure) | **OD-2's literal clean-environment wording ("no `PYTHONPATH`, no `.pth`") cannot be satisfied.** The ACT-N10 companions (`verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, material, unchanged) do `from scripts.governance…` and need the repository root on their path. In a venv with no `.pth` and no `PYTHONPATH` (also with `python -m` from the root) the gate fails closed with `A-01 … No module named 'scripts'`. The working clean forms are (a) the `uv sync` editable install whose only `.pth` points at the clean clone, or (b) `PYTHONPATH=<clean clone>` (the form the repo's own tests use). Both reached the expected result. | None on the code (it fails closed, not open). The operator should amend the OD-2 procedure text so a later certifier is not left with an unsatisfiable instruction. |
| **C-02** | MEDIUM (process) | **OD-1 and OD-2 were decided by the implementer.** Both records state that the implementer agent chose the option and recorded it "on the operator's behalf" under a blanket chat authorisation ("udzielam ci autoryzacji na wszystkie decyzje"). That is the same session whose corrective, residuals and thresholds are being accepted. OD-1 accepts residual W401-RA-08 (the P2 pin sits in an ungoverned file); OD-2 sets the T4 boundary that excuses W404-05/W405-03. They are `process_enforced` and self-described as overrulable. For a certification that is meant to be independent, this is a conflict of interest the operator can remove at no cost by re-adopting (or changing) both in their own words/commit. | Condition 1 below. Does not change my technical view of the code. |
| **C-03** | MEDIUM (readiness) | **The B-4 *resolution* path has never run on the real chain.** The blocker is listed as open in the committed certification record; `A-18` clears only through the blocker-lifecycle rules (resolution with `closure_evidence` limited to `path`/`sha256`/`decision_id`, indexed evidence directory, everything committed so `HEAD` binding holds, certifier disposition in structured keys). The fixture suite exercises this (`_resolve_a_blockers`, "Certifier records B-4 and P3 resolutions"), and the CI/auditor tests are green, but no one has produced the real resolution commit. A format error there is a fail-closed surprise at the worst time (P3). | Condition 2. |
| **C-04** | LOW | Same-host evidence: implementer, auditor and I all ran on the operator's Windows machine; Linux is covered only by CI (verified by me through `gh`, not reproduced). | Acceptable; see "not verified". |
| **C-05** | INFO | `ENT-AUDIT-X` workflow red on `development` since 2026-11-05; unrelated to B-4 but a red repository gate that EVOL-X closure will meet later. | None for B-4. |

## 5. Residuals — acceptability under OD-1 / OD-2

| Residual | Tier / sev | My assessment | Acceptable? |
|---|---|---|---|
| **W405-01** companions inherit `PYTHONPATH`; a hidden root module runs inside them | T2 + env, LOW | **Reproduced (P2).** Verdict stays FAIL; forging companion success is moot only while the inputs they check are sealed and genuinely valid, and since they are called with sealed post-P2 data, a real failure in them is not expected. A cheap fix exists (`-I` or dropping `PYTHONPATH` in the child env — but note C-01: they need the root on their path, so the real fix is an explicit `sys.path` in the companions, which are material). | **Yes**, under OD-2 (accepted LOW). Re-check at P3. |
| **W405-02** any untracked root `.py` blocks both gates | usability, LOW | Fail-closed; in practice the operator needs a clean tree anyway. | **Yes** |
| **W405-03** `.pth`/`sitecustomize`/venv-in-checkout can force PASS | T4, INFO | Out of scope by OD-2; mitigation is exactly this fresh-clone run (and C-01 shows the clean env still needs one benign `.pth` or `PYTHONPATH`). Note: the safe form is to keep the venv outside the clone (done here). | **Yes**, conditional on C-02 (operator owns the T4 decision) |
| **W404-04** free text in `operator_decisions/`; `b4_reaudit/` index editable | T1, LOW | Not read by any gate, so editing it cannot change a verdict; but it means those directories are **not tamper-evident**: cite commit SHAs, not paths, whenever they are referenced as evidence. | **Yes** with that rule |
| **W401-RA-05** negation denylist / ASCII-only decision scope | usability, LOW | Fail-closed. Real risk: operator decision records are written in Polish (diacritics) and the W4-F08/OD records themselves contain Polish quotes; a P3/OD-EVOL-12 record with non-ASCII scope text or a word on the denylist will be **rejected**. Recommend a dry-run of the intended record text against a fixture clone before the operator signs. | **Yes** for B-4; operational risk for B6 |
| **W401-RA-07** fixture-class PASS exits 0 | INFO | Confirmed in code. Any consumer that treats exit 0 as "canonical PASS" without checking `repository: canonical` (or `repository_class` in `--json`) is misled. Nothing currently does. | **Yes**; add to the B6 acceptance checklist |
| **W403-04** forged commit-graph / `GIT_DIR`, `GIT_INDEX_FILE` / git calls of imported material modules | T3/T4, INFO | Still **UNVERIFIED** for the forged commit-graph and the imported modules. New datum: `GIT_DIR` redirect is *not* rejected (P7, benign target only). The class is T4 (environment) and excluded by OD-2; the fresh-clone/clean-env procedure removes it. | **Yes under OD-2**, not independently closed |

W404-05 (T4 interpreter hooks) is the same boundary as W405-03 and covered by the same reasoning.

## 6. Conditions for acceptance (recommendations to the operator)

1. **Operator ratification of OD-1 and OD-2** by the operator themself (own words or own commit), or an explicit change; until then my acceptance relies on decisions made by the implementer (C-02). Also amend OD-2's environment wording per C-01.
2. **Dry-run the real B-4 resolution** (and the P3 record text, W401-RA-05) on a throwaway clone of the pin before publishing it: the expected end state is `--gate b6-readiness` with only the `B6-P3` `A-18` left (C-03). The resolution must cite this report by commit SHA, not by path (W404-04).
3. **Keep the pin immutable**: any change to `scripts/qualification/evol_x_closure_publication_preflight.py`, the P2 constant or `scripts/governance/*` after this report voids it (OD-1 review trigger) and needs its own ticket and audit.
4. **Re-open OD-1 and OD-2 before P3**, as both records already require; add: re-run this fresh-clone procedure at the SHA that carries the resolution commit.
5. Keep `EVOL-X` `NOT_CLOSED`; B-5 (OD-EVOL-12 / OD-EVOLX-08), B6 acceptance, FRZ promotion and closure remain operator-only and untouched by this report.

## 7. Not verified / limits

* No Linux run of my own; CI evidence only (green on `d75d905cc`, a doc-only descendant of the pin). No run of `evol-x-b4-preflight` or `governance-fast` on `bec0a0948` itself (no run listed for it when I looked).
* Governance suite and B4 suite were run **at the pin only**; at `bec0a0948` I ran the gates (rows 2–4) and confirmed no code differs.
* Forged commit-graph, `usercustomize`, brute-forcing the predictable pycache prefix (`<tmp>/evolx-preflight-no-bytecode-cache/<pid>`), symlink-based attacks, `GIT_INDEX_FILE`/doctored `GIT_DIR` and the imported modules' own git calls were not exercised.
* My attacks are a sample, not an exhaustive hunt; the auditor's five rounds and the hunt D1–D9 remain the broader coverage, which I did not redo.
* I cannot verify that the operator authorised what OD-1/OD-2/W4-F08 say they authorised (chat is not in the repository); I only read the records.
* The notes folder logs are my own copies; they are not part of the committed evidence.
