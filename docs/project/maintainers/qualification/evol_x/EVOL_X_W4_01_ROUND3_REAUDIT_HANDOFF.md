# EVOL-X W4-01 round 3 - delta re-audit handoff

**For:** an independent-quality-auditor session (the operator decides whether the round-1/2 auditor `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` continues or a fresh one is used). Not the implementer of this round, not the round-1/2 implementer `local_4b071fe5-82d0-48bc-b524-146a3daf5c45`, not the certifier-track session `local_ea0a6280-bbd7-4851-9129-a82e3519990f`.
**Ticket:** `T-IGX-019-A` ("Round 3" section) · **Policy:** `IIAC-2026-11-05.1`
This handoff is evidence for an audit. It is not a PASS record, resolves nothing and authorises nothing. EVOL-X stays `NOT_CLOSED`; B-4 stays open. **Round 3 is committed on a local branch only; it is not pushed or published, and the operator has not approved publication of round 3.**

## 1. Pins

| Role | SHA |
|---|---|
| Audit baseline | `5afaad9eadfc1655f4ef1ab59eb022960a924648` |
| Round-1 code (PARTIAL) | `9f8b04d2b8b670077b1e78ce812bb34b680f3968` |
| Round-2 code (PARTIAL, blocker W402-01) | `ca78b1c1f73f1e23c3a9bfed118988c12a0c71fa` |
| Base of round 3 (`origin/development`) | `f4ce43dcb43764f2c5376b2c4e10fd964625788b` |
| **Round-3 code (audit this)** | `a1510bee0` (`scripts/` and the preflight code); `a991a316f` changes one test only |
| Requalification SHA / head | the commit that adds this document (`git log -1 -- docs/project/maintainers/qualification/evol_x/EVOL_X_W4_01_ROUND3_REAUDIT_HANDOFF.md`); it differs from `a991a316f` by documentation only - verify with `git diff a991a316f HEAD -- . ':!docs' ':!governance/tickets'` (must be empty) |

Branch: `claude/elegant-cannon-a61eed`.

## 2. What changed per finding

Only `scripts/qualification/evol_x_closure_publication_preflight.py`, `tests/unit/qualification/test_evol_x_closure_preflight_b4.py`, `governance/tickets/T-IGX-019-A.md` and this document. Nothing under `intergrax/`, `agents/`, `applications/`, `scripts/governance/`, the activation record, `iiac_evidence/`, existing evidence or `.gitattributes`.

* **W402-01 (MEDIUM blocker).** `git status` is no longer consulted. `_dirty_input_errors` now (1) rejects tracked verified paths whose `git ls-files -v` tag is not `H` (`S`/`s` skip-worktree, lowercase assume-unchanged, other index states), (2) compares every verified input byte-wise with its HEAD blob through `_worktree_matches_blob` (the RA-B4-01 CRLF contract; the LF form of a CRLF file is hash-checked first and the `text` attribute is answered by one batched `git check-attr`, same semantics), and (3) lists the files on disk under the evidence root and flags those absent from the HEAD tree. Verified inputs = every path the preflight read, every `REL_*` input (so Gate A and Gate B agree), every HEAD file under the EVOL-X evidence root, every HEAD file under `scripts/`, and the preflight itself. Codes `A-21` / `B-12` unchanged; labels per path: `[skip-worktree]`, `[assume-unchanged]`, `[modified]`, `[missing]`, `[not in HEAD]`, `[untracked]`. All git calls run with `core.fsmonitor=false` (and `core.commitGraph=false`); `_git_repo_config` was removed.
* **W402-02 (LOW).** Everything under `scripts/` is byte-bound (covers `scripts/governance/iiac_common.py`, `iiac_activation_gate.py`, `scripts/qualification/evol_x_frz_evol_canonical_matrix.py`). Real-clone test appends a line to each of the three and expects `A-21`.
* **W402-03 (LOW).** (a) An open blocker's text must equal the text published at P2 (or when first published); (b) resolved-blocker entries, `certifier_disposition` and `closure_evidence` items accept only allow-listed keys; (c) `stage_closure_status` is an enum `{NOT_CLOSED, CLOSED}` (unhashable values no longer matter); (d) new `A-23`: every committed file under `iiac_evidence/EVOL-X/` must be covered by an evidence index in one of its ancestor directories below the root (`operator_decisions/` is exempt - its records are digest-bound by `closure_evidence`). A file at the root or in an unindexed directory fails.
* **W402-04 (INFO).** fsmonitor disabled and tested with a stub hook that makes `git status` blind. `core.commitGraph=false` added (no forged-commit-graph test; documented). Alternates / a tampered object database are documented residuals in the module docstring.

## 3. Tests (clean committed tree, Windows, `core.autocrlf=true`, `uv sync --extra dev-ci --frozen`)

| Command | Result |
|---|---|
| `pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py -q -n 4 --basetemp=D:/w4i3/b4` (tree `a991a316f`) | **497 passed, 0 failed, 28 min 49 s** (448 previous + 49 new) |
| `pytest tests/unit/governance -q -n 4 --basetemp=D:/w4i3/gov` (tree `a1510bee0`, code-identical) | **145 passed, 1 skipped (symlinks), 0 failed, 10 min 06 s** |
| `verify_iiac_policy_activation --require-in-force` | `IN_FORCE @ IIAC-2026-11-05.1` |
| canonical `--gate b6-readiness` | FAIL, exactly two `A-18` (B-4, B6-P3); no A-19/A-21/A-22/A-23 |
| canonical `--gate formal-closure` | FAIL; no A-21/A-22/A-23/B-12/B-13 |
| `ruff check` / `ruff format --check` (changed .py files) | clean |
| `igx_gov ticket` | OK (push-fast PASS) |
| GitHub Actions | not run (nothing pushed) |

New tests: 49 (the `w402` group in the B4 module, including the real-clone test); no existing assertion removed from the test module except one edit: **`test_open_blocker_holds_formal_closure` renamed the open B-5 blocker text and expected Gate A to pass - exactly the channel W402-03 closes. It now keeps the published `B5_TEXT`; its intent (an open B-5 blocker holds formal closure with `B-09`) is unchanged.** Please judge this edit.

Red-run evidence: the new tests were first run against the round-2 code (46 failed, 3 passed; for example the auditor's V-a case ended `gate_a=[] gate_b=[]` = both PASS). Log: `C:\Users\XPS\AppData\Roaming\Claude\side-session-notes\local_7eeb7387-df63-460d-88e9-be7359430173\w4-01-round3_red_run_against_round2_code.log`.

## 4. How to attack it

1. V-a: commit a widened FRZ row, `git update-index --skip-worktree` (or `--assume-unchanged`) the checklist, restore the correct row in the working tree (`git status --porcelain` is empty). Expect `A-21` and `B-12` naming `[skip-worktree]`.
2. Same for the register, governance JSON, ladder, certification record/handoff, pass record and the script (parametrized in `test_w402_01_a_hidden_difference_on_any_verified_input_fails_both_gates`).
3. fsmonitor stub hook (`core.fsmonitor=<script printing a token>`; `update-index --fsmonitor --refresh`): `git status` is blind, the gate still fails.
4. Real tree: append one line to `scripts/governance/iiac_common.py` in a clone; expect `A-21`.
5. W402-03 variants V-d..V-g. Controls that must stay green: fresh clones with `autocrlf` true/false/input, reordered limitations, untracked unrelated files outside evidence and `scripts/`, a skip-worktree flag on an unrelated tracked file.

## 5. Residuals (not bundled, name them again)

* W401-RA-05 (negation denylist, process-enforced), W401-RA-07 (fixture exit code), W401-RA-08 / OD-1 (operator decision; the pin and its cross-check live in the same ungoverned script). Ask for INFO.
* Alternates (`.git/objects/info/alternates`), a tampered object database or a forged `.git/HEAD`: repository is the trust anchor.
* Ignored files below the evidence root are flagged as untracked (stricter than `git status`); untracked files in `scripts/` that nothing imports are not flagged.
* Evidence must be committed before the preflight runs (unchanged from round 2).
* Linux container not run here; CI after publication is the substitute.

## 6. Not done / needs the operator

Publication (push/merge) of round 3 needs the user's own approval; then the delta audit, then a certifier. W4-F01 (option C), OD-1, W4-F08, OD-EVOL-12, OD-EVOLX-08, P3, B6, FRZ promotion and EVOL-X closure are untouched.
