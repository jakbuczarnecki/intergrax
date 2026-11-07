# EVOL-X T-W4-01 — post-implementation audit and next-stage plan

> **FINAL STATUS (2026-11-07, supersedes every verdict and plan below).** This is the post-implementation recovery report of session `local_7eeb7387-df63-460d-88e9-be7359430173` (not the implementer, not an auditor, not a certifier). Its earlier verdicts (including "QUALIFIED - READY FOR INDEPENDENT RE-AUDIT") applied to the code SHAs named in each block and are historical. Final chain for T-W4-01: five audit rounds by `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` (round 5 on `fdb1e7a62`: PARTIAL, no blocker); certifier `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18` ACCEPT_WITH_CONDITIONS; second certifier `local_e34301ce-67df-40bf-b274-b2e062581541` NOT CERTIFIED for CERT-01, resolved by OD-2 Amendment 2 (the `python -m` form is not supported; a script-level fix is impossible, verified by this session); B-4 recorded RESOLVED_WITH_EVIDENCE at `7a5476cdc`; ticket `T-IGX-019-A` closed. `--gate b6-readiness` fails only on `A-18` for `B6-P3/P4/P5`. EVOL-X stays `NOT_CLOSED`; B-5, P3, B6 acceptance, FRZ promotion, OD-EVOL-12 and OD-EVOLX-08 remain operator-only and open. This session made no code change; its test runs and probes cover earlier pins and are not an audit or certification result.


**Verdict:** `T-W4-01 — QUALIFIED — READY FOR INDEPENDENT RE-AUDIT`
**Role of this session:** recovery and qualification (not the independent auditor, not the certifier, not the implementer of T-W4-01). This report is **not** an audit PASS, resolves nothing and authorises nothing.
**Not done / not claimed:** no B6 acceptance, no FRZ promotion, no OD-EVOL-12 / OD-EVOLX-08 / P3 / OD-1 / W4-F01 / W4-F08 decision, no change to `open_blockers_for_closure`, no merge, no push, EVOL-X stays `NOT_CLOSED`.
**Policy:** `IIAC-2026-11-05.1` `IN_FORCE` (verified, §10).

> **ROUND 4 REQUALIFICATION (latest; supersedes the round-2 block below).** Independent audits returned PARTIAL on `9f8b04d2b` (3×MEDIUM), on `ca78b1c1f` (W402-01) and on the round-3 code `2e0ed9c6b` (W403-01, forged `.pyc`). The implementer's round 4 is `dc650b8df` (code); `origin/development` is `2f02c92bd8fc8df1d7f2b9e9e967ca54ab00cedf` (docs only after it; `git diff dc650b8df 2f02c92bd -- . ':!docs' ':!governance/tickets'` empty). A duplicate round-3 branch started by the recovery side (`claude/elegant-cannon-a61eed`, `6352aca13`) was **not published** and is superseded. Requalified at `2f02c92bd` (Windows, `core.autocrlf=true`, clean worktree before and after):
>
> | Control | Result |
> |---|---|
> | B-4 module + ruff-evidence exemption (`-n 4`) | **489 passed**, 0 failed, 27 min 03 s (equals Linux CI) |
> | `tests/unit/governance` (`-n 4`) | **145 passed, 1 skipped** (symlinks), 17 min 09 s |
> | activation; audit manifest; role separation | `IN_FORCE @ IIAC-2026-11-05.1`; PASS; PASS |
> | `--gate b6-readiness` | FAIL as expected: exactly two `A-18` (B-4, B6-P3); `formal-closure` FAIL as expected |
> | ruff check / format | clean |
> | CI run `42248677666` on `2f02c92bd` | green: evol-x-b4-preflight 489, governance-fast 146, ci-smoke 1882 |
> | Probes N1–N20 (`probes/probes_round4.py`, `probes_round4_n19_n20.py`) | all rejected: replace refs / grafts (`A-22`), modified / untracked inputs (`A-21`/`B-12`), `-s ours` hidden blocker (`A-18`), lifecycle values (`A-19`), **skip-worktree and assume-unchanged with a blind `git status`** (rejected), widened checklist row hidden by skip-worktree (`B-12`), **forged `UNCHECKED_HASH` .pyc: a plain import runs it, the preflight does not execute it and its output equals the baseline**, preflight script edited in the working tree (`A-21`) |
>
> Worktrees clean after the runs (main: only the pre-existing matrix change and the untracked recovery files). **Verdict for the current code: `T-W4-01 — QUALIFIED — READY FOR INDEPENDENT RE-AUDIT (round 4, pin dc650b8df)`** — a qualification, not an audit PASS; it cannot replace the auditor's verdict. Not executed: `igx_gov ticket`, Linux locally, `real_scope_probe2.py`, `auditor_probes_fixture.py`. Residuals named by the implementer: W401-RA-05, RA-07, RA-08 (OD-1), W403-04 (forged commit-graph, `GIT_DIR`/`GIT_INDEX_FILE`, git calls of the imported material modules — UNVERIFIED), free files in `operator_decisions/`. The implementer's threat-model note (a guard against careless divergence, not against an attacker who controls the machine; run on a fresh clone) is an operator decision (suggested OD-2). The OD-1 / W4-F08 records and the W4-F01 ratification draft keep the provenance caveats below; the W4-F01 draft is still unsigned with no decision id.
>
> **ROUND 2 REQUALIFICATION (superseded by the round-4 block above).** The independent re-audit `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` returned **PARTIAL (three MEDIUM blockers) on `9f8b04d2b`** — so the verdict in this report's title applies only to `9f8b04d2b` / `75b8ddc87` and was not an audit result. The implementer fixed W401-RA-01..04 and RA-06 in `ca78b1c1f` (code) and documented it in `af7b98019`; `origin/development` is now `f4ce43dcb43764f2c5376b2c4e10fd964625788b` (docs only after `ca78b1c1f`; `git diff af7b98019 f4ce43dcb -- . ':!docs' ':!governance/tickets'` empty). Requalified here at **`af7b9801933fdb554f50fa93d517cdbf4b9c0bbe`** (code-identical to `ca78b1c1f`, Windows, `core.autocrlf=true`, clean worktree before and after):
>
> | Control | Result |
> |---|---|
> | B-4 module + ruff-evidence exemption (`-n 4`) | **448 passed**, 0 failed/skipped, 20 min 50 s (equals Linux CI) |
> | `tests/unit/governance` (`-n 4`) | **145 passed, 1 skipped** (symlinks), 10 min 42 s |
> | activation `--require-in-force`; audit manifest; role separation | `IN_FORCE @ IIAC-2026-11-05.1`; PASS; PASS |
> | `--gate b6-readiness` | FAIL as expected: exactly two `A-18` (B-4, B6-P3), nothing else; `--gate formal-closure` FAIL as expected |
> | ruff check / format | clean |
> | CI run `42224727428` on `af7b98019` | green: evol-x-b4-preflight 448, governance-fast 146, ci-smoke 1882 |
> | Probes (`probes/probes_round2.py`, N1–N15) | all attacks rejected, **including the former R-1 / R-2** (`A-19` lifecycle shape / added file), `refs/replace` (`A-22`), grafts (`A-22`), modified/untracked inputs (`A-21`), late blocker hidden by `-s ours` (`A-18`); `stage_closure_status=CLOSED` is not an `A` error but fails Gate B (`B-10`); an unrelated dirty file is correctly not flagged |
>
> Worktrees clean after the runs (main: only the pre-existing matrix change). **Verdict for the current code: `T-W4-01 — QUALIFIED — READY FOR INDEPENDENT RE-AUDIT (round 2, pin ca78b1c1f)`** — a qualification, not an audit PASS. Not executed: `igx_gov ticket`, Linux locally, `real_scope_probe2.py` and `auditor_probes_fixture.py`. Residuals documented by the implementer and left open for the next audit: W401-RA-05 (negation denylist / ASCII-only scope), RA-07 (fixture PASS exit code), RA-08 (OD-1).
> **Provenance caveat:** the newest commits record OD-1 (`ACCEPT_RESIDUAL_NO_MATERIAL_CHANGE`, with a mandatory re-open before the P3 B6 authorisation) and the W4-F08 signature "by delegation"; both were written by the implementer agent on the operator's behalf. I did not verify the operator statements behind them; they are process-enforced text and the auditor/certifier should treat them as such. My own W4-F01 record (option C) is still unsigned and has no decision id.
>
> **Update (after the first tests, at `75b8ddc87`).** `origin/development` has moved once more to `6aa8c89e9b410f1e73f6e0ca0e109cfc260d4ae6`: the single new commit adds the unsigned `EVOL_X_W4_F08_OPERATOR_ATTESTATION_RECORD.md` (docs only; code diff vs `9f8b04d2b` still empty) and names an independent-auditor session `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` already started by the operator. Tests and probes in this report ran at `75b8ddc87`. The auditor package is in `.tmp/session/w4-01-reaudit-package/` (see its README).
>
> **Premise correction.** The brief assumed `origin/development` = `5afaad9ea` and that W4-F01 is a pending operator decision. Both are outdated: `origin/development` is `75b8ddc87` (14 commits ahead, fast-forward publication already done), and W4-F01 was repaired technically (§12). What is still open for W4-F01 is a *ratification*, not a repair.

## 1. Actual Git baseline

| Item | Value |
|---|---|
| Fetch | `git fetch origin` OK |
| Main checkout | branch `development`, HEAD `5afaad9eadfc1655f4ef1ab59eb022960a924648`, 0 ahead / **14 behind** `origin/development` |
| `origin/development` | `75b8ddc87c5047614e5b9c4a083ceee26a6891a4` (= local branch `evolx/w4-01-ci-evidence`) |
| Working tree (main) | only `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` modified (pre-existing, unrelated, **preserved**; unchanged after all test runs) |
| Qualification worktree | `D:/w4x` detached at `75b8ddc87`, `core.autocrlf=true`, clean before and after the runs |
| Not done | no pull, merge, reset, push, branch/worktree deletion of anyone else's |

## 2. Recovered session artifacts

| Artifact | Location | Status |
|---|---|---|
| Part 1 branch | local `claude/compassionate-khorana-cda630` @ `9a6b63abf` (worktree `.claude/worktrees/compassionate-khorana-cda630`) | `e0eeca8de`, `742b47cfc` are in `development`; **`9a6b63abf` (Part 1 re-audit handoff) is local-only** and superseded (still says W4-F01 "not done") |
| Qualification candidate | `qual/t-w4-01-candidate` @ `fa43976f8` (local + origin; worktrees `D:/w4q/fix`, `D:/w4q/cand` @ `742b47cfc`) | all ancestors of `development` |
| Consolidated corrective | `evolx/w4-01-consolidated` @ `f6d3b6aaf` (PR #388, merged 2026-11-06T11:58:46Z) | in `development` |
| CI-evidence branch | `evolx/w4-01-ci-evidence` @ `75b8ddc87` (`D:/w4d`) | = `origin/development` |
| Final report / handoff | `evol_x/EVOL_X_W4_01_CONSOLIDATED_CORRECTIVE_IMPLEMENTATION_REPORT.md` (+ addendum §L, binding §M), `…_REAUDIT_HANDOFF.md`, `…_W4_F08_OPERATOR_ATTESTATION_TEMPLATE.md` | present; the previous session **did** produce a complete final report |
| Auditor package (W4-01 audit) | `.tmp/session/w4-01-package/` (spec, findings JSON with W4-F01..F08, probes) | present, untracked |
| Delta re-audit dossier (`EVOL_X_B4_DELTA_REAUDIT_DOSSIER.md`, source of DA-01/02/09) | searched repo, all worktrees, `.tmp`, user folders | **NOT FOUND** (the implementer also reported it missing; DA-xx were taken from mission text) |
| Reflogs | khorana branch reflog consistent: reset to `5afaad9ea` → `e0eeca8de` → `742b47cfc` → `9a6b63abf` | no lost commits found |

Verified vs reported: the 14-commit chain, the code-identity of `9f8b04d2b` and `75b8ddc87`, the test counts and CI results below were **verified here**. The session-id attributions in report §H and the "Publication-Approved: operator" line were **reported only** (process-enforced text, cannot be verified from the repo).

## 3. Commit and SHA inventory

| Role | SHA |
|---|---|
| Audit baseline / START_HEAD | `5afaad9eadfc1655f4ef1ab59eb022960a924648` |
| `e0eeca8de` (UTF-8 republication, evidence-only) | `e0eeca8dee43369c1355c91437090beeb67f2a42` |
| `742b47cfc` (Part 1) | `742b47cfc096fc4878e49dce50313302169640ad` |
| P2 pin / qualification | `e641a134ae15eb5ab7933afcbbb85df653bdf02d`, `fa43976f8b3db90bf0b7a1cbe4127463419b7430` |
| W4-F01 repair | `4aa91dcc0e569c602baa782d2bb742b3bd55837d` |
| Consolidated hardening | `a4926c3d26ad7205ad733c9cd14ad683c2c73264`, `b82c066f7026352c1041b580691bd0173cc05db5` |
| **IMPLEMENTATION_SHA / TEST_EXECUTION_SHA (code)** | `9f8b04d2b8b670077b1e78ce812bb34b680f3968` |
| Docs-only after code | `445d5883c`, `5376d761a`, `f6d3b6aaf`, `49b50fd74` (publication head, CI run), `df502966c`, `75b8ddc87` |
| **Current `origin/development`** | `75b8ddc87c5047614e5b9c4a083ceee26a6891a4` |
| Original P2 anchor | `ad0877715ed079ce920ff9d925fda03555b2c04e` — **independently verified**: `git log --diff-filter=A` for `EVOL_X_IIAC_CERTIFICATION_RECORD.json` returns exactly this commit |

Later commits are code-identical: `git diff 9f8b04d2b HEAD -- . ':!docs' ':!governance/tickets'` is **empty**. `git diff 5afaad9ea HEAD -- intergrax agents applications scripts/governance <activation record> <protocol> iiac_evidence` is **empty**. All commits touching `scripts/` carry `Ref: governance/tickets/T-IGX-019-A.md`.

## 4. Changed-file assessment (`5afaad9ea..75b8ddc87`, 11 files, +2500/−41)

| File | Assessment |
|---|---|
| `scripts/qualification/evol_x_closure_publication_preflight.py` | the only code change; non-material by `MATERIAL_GOVERNANCE_PATHS` (not listed) |
| `tests/unit/qualification/test_evol_x_closure_preflight_b4.py`, `…/conftest.py`, `tests/unit/governance/test_evol_x_real_repo_fresh_clone.py` | tests; strict W4-F01 xfail markers removed (none remain) |
| `.gitattributes` | +3 `-text` paths (W4-F06) and `text eol=crlf` for the activation handoff (W4-F01); **not** a material path, not under CODEOWNERS (no CODEOWNERS file exists) |
| `.github/workflows/unit-tests.yml` | new `evol-x-b4-preflight` job, `governance-fast` timeout 8→15 min |
| `…BINDING.utf8.json` | evidence-only republication (§8) |
| ticket + report + handoff + W4-F08 template | documentation |

## 5. DA-01 — immutable certification baseline: **PASS (verified), with one governance residual**

* Pin `P2_CERTIFICATION_PUBLICATION_SHA = ad0877715…` is a module constant; the canonical repo rejects `--p2-sha` overrides and `--evidence-root`; fixtures without an explicit anchor also use the constant (declared-field fallback removed).
* Own probes (throw-away clone of `75b8ddc87`, not in the earlier auditor's set): **N4** repoint manifest+record `certification_publication_sha` to a later commit → rejected (2× `A-19 … baseline cannot be re-pointed`); **N5** `--p2-sha` override → `INVALID`; **N6** edit the constant itself and repoint everything → still rejected (`A-19 … not the commit that first published the certification record`), because the pin is cross-checked against history; N3 deleting blockers + flags → rejected (`A-18` lifecycle, 3 blockers).
* Residual (unchanged from implementer's OD-1): the script is not under CODEOWNERS/material protection; an actor who rewrites the derivation logic as well as the pin is caught only by review.

## 6. DA-02 / RA-B4-05 — certified scope integrity: **PASS (verified), two LOW hardening gaps**

* Earlier auditor's real-evidence attacks re-run against `75b8ddc87`: LIM-03 / LIM-05 widened, LIM-08..10 + F-C-* + evidence-only limitation dropped, FRZ-EVOL-07 widened, verdict widened, all blockers deleted, B-4 "resolved" with the historical PARTIAL evidence → **all rejected** (`A-19`, `A-07`, `A-18`, `A-13`).
* Own probes: manifest key-order/indent change (N7, M0 byte contract), P2 assessment edited with regenerated index (N8, F-C-01/02 withdrawn), auditor PASS record edited (N9), unindexed scope-amendment file (N2) → all rejected.
* **R-1 (LOW):** free text appended to the lifecycle key `certification_publication_sha_binding` is accepted (N1; no new gate error). No gate or verdict consumes it.
* **R-2 (LOW):** a new file added **and indexed** in the certification directory is accepted (N2b). Nothing consumes it; the file set is not allow-listed.
  Neither changes verdict, limitations, dispositions, blockers or any gate outcome; both fall under the implementer's "lifecycle key / no amendment channel" caveats. Recommended in §13 as optional hardening, not a blocker.

## 7. RA-B4-04 — B6 evidence binding: **PASS (code read + tests)**

`_gate_b_b6` / `_b6_record_errors`: B6 needs `b6_acceptance_publication_sha` (not self-referential, ancestor of HEAD, after P2 and after the P3 closure), a record that is **absent at P2 and present at the declared SHA**, byte-equal to the file, digest-bound, index-verified, manifest unchanged since, allow-listed fields only, canonical scope string, `Accepted limitations` = baseline `<id>:<sha256>` set, signature not older than P2 / not later than the B6 commit. Flags-only (N3) and limitation-ID-only attempts are rejected. **Provenance limit unchanged:** the operator signature line is process-enforced text; a format-valid agent-authored signature is not cryptographically distinguishable (documented since S04-05).

## 8. DA-09 — UTF-8 metadata: **PASS (verified)**

Original `…BINDING.json` is byte-identical to the baseline and still not UTF-8 (byte `0x97` at offset 954, by design); `…BINDING.utf8.json` is valid UTF-8, no BOM, no CRLF; `supersedes.sha256` = `584c504e…75be9` = SHA-256 of the original blob; delegation record identical to baseline, SHA-256 `2120852e…` = the bound `delegation_record_sha256`; `git check-attr` shows `text` unset (= `-text`) for all three. Only value difference vs the original: the new field `republication_ticket` (the report said "identical field values + supersedes" — small documentation inaccuracy, **R-4**).

## 9. Regression results

RA-B4-01 (checkout-independent bytes) and RA-B4-03 (strict operator decisions): their tests are inside the 423-test B4 module and passed; real-repo fresh clones with `autocrlf` true / false / input passed inside the governance run. HF-01..03: governance suite green, no canonical file written by tests (both worktrees clean after the runs). No regression observed.

## 10. Complete qualification matrix (tested tree `75b8ddc87`, Windows 11, `core.autocrlf=true`, Python 3.12.11, `uv --frozen`)

| Control | Command | Result |
|---|---|---|
| Complete B-4 module + ruff-evidence exemption (DA-01/02 mutation matrix, B6 binding, UTF-8/CRLF, blocker lifecycle) | `pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py -n 4 --basetemp=D:/w4x_bt/b4` | **423 passed**, 0 failed/skipped, 16 min 44 s |
| Governance suite (incl. real-repo fresh clones, W4-F01, IIAC activation tests) | `pytest tests/unit/governance -n 4` | **145 passed, 1 skipped** (symlink creation unavailable on this host), 0 failed, 9 min 40 s |
| IIAC activation | `verify_iiac_policy_activation --require-in-force` | exit 0, `IN_FORCE @ IIAC-2026-11-05.1` |
| Manifest verification | `verify_iiac_audit_manifest <EVOL-X certification manifest>` | PASS |
| Role separation | `verify_iiac_role_separation <same manifest>` | PASS |
| Canonical preflight | `--gate b6-readiness` | exit 1 as expected: exactly two `A-18` (B-4, B6-P3), **no `A-19`** |
| Canonical preflight | `--gate formal-closure` | exit 1 as expected (B6, FRZ-EVOL-01..10 OPEN, OD-EVOL-12/OD-EVOLX-08 PENDING, B-4/B-5/B6-P3) |
| Lint | `ruff check` / `ruff format --check` on the 4 changed Python files | clean / already formatted |
| Adversarial probes | earlier auditor's `real_scope_probe.py` (re-pointed to `75b8ddc87`) + own `own_probes.py` (N1–N9) | all attacks rejected except R-1 / R-2 (§6) |
| Files modified by tests | `git status` in both worktrees after runs | none (main: only the pre-existing matrix change) |
| **NOT RUN** | `igx_gov ticket` (ticket Ref check + regression fast); Linux locally; full-repo `ci-smoke` locally | replaced by CI evidence below; `igx_gov ticket` not executed here |

Run notes (disclosure): my first launch of the two suites failed immediately because I had not created the `--basetemp` parent and then passed a flag that conflicts with the repo's `cache_dir` config; both runs were discarded and rerun. The manifest/role-separation commands also first failed for lack of an argument and were rerun. Only the reruns are reported. All SHAs were checked: `00_sha.txt` shows `75b8ddc87…`.

## 11. CI verification (GitHub Actions)

| Run | Head | Result |
|---|---|---|
| `42214173868` "Regression gate", push to `development` | `49b50fd74` (code-identical to `75b8ddc87`) | success: `governance-fast` **146 passed**, `evol-x-b4-preflight` **423 passed**, `ci-smoke` **1882 passed**, `governance-policy-pin` success; remaining jobs skipped by design |
| `42210092666` (PR #388) | `445d5883c` | same two Linux jobs green (per implementer; consistent with the above) |
| `42175145655` | `81418b0dd` | failed (2× ACT-N08 = W4-F01) — the finding that the corrective fixes; green at `49b50fd74` |
| `df502966c`, `75b8ddc87` | docs-only | **no workflow run exists** |
| `ent-audit-fast` / ENT-AUDIT-X | — | red on `development` before this work (e.g. `2ae101411`, `eab65e010`, `116fb6a2a`), not triggered for `49b50fd74`; pre-existing, unrelated, **not fixed here** |

## 12. W4-F01 decision packet (unsigned; I decide nothing)

* **Normative source:** `.tmp/session/w4-01-package/EVOL_X_W4_01_INDEPENDENT_AUDIT_FINDINGS.json` (W4-F01, HIGH, `fix: … needs an Operator decision: material governance file`) and `EVOL_X_W4_01_CORRECTIVE_SPECIFICATION.md` §2 and §5.1 ("Operator decision (new, `OD-` id to be assigned): authorise the activation-record amendment of W4-F01. Not substitutable by the implementer … must not be done silently"). These two files are outside the repository (untracked, auditor-held).
* **Defect:** `handoff_digest` `96fcf995…` = SHA-256 of the CRLF-expanded handoff; blob (LF) digest `3b7e901f…`; LF checkouts failed `ACT-N08`.
* **What was done:** neither spec option. Option **C**: `.gitattributes` `text eol=crlf` for the handoff record (commit `4aa91dcc0`); blob, activation record and gate untouched.
* **Pending decision:** ratify C as a non-material repair of the spec's W4-F01 — or require A / B. No OD exists. Ticket `Approved:` cites the implementation mission, `Publication-Approved:` covers publication only, and **IIAC OD-01 (`IIAC-OD-01-2026-11-05`) is the policy activation, not W4-F01 approval**.

| Option | What | Security / authority |
|---|---|---|
| A (spec, preferred) | `-text` on the handoff, re-bind `handoff_digest` to blob `3b7e901f…` through the activation amendment | cleanest (digest of committed bytes); changes the activation record → amendment procedure, probably re-activation/re-audit |
| B (spec) | ACT-N08 hashes the committed blob | edits `iiac_activation_gate.py` (material path) → full IIAC governance re-audit |
| **C (implemented)** | `eol=crlf` attribute | no material file changed; **fail-closed** (removing the line or any byte change → `ACT-N08`; +1 byte, LF-collapsed, double-CR, deleted all fail); verified in `autocrlf` true/false/input clones and Linux CI. Weaker semantics: digest is over a *materialised* CRLF file, so a consumer that bypasses gitattributes (`git show`, raw download) sees LF; the attribute line is unprotected (no CODEOWNERS) |

* **Recommendation (for the operator to accept or reject):** ratify **C** for B-4, because it is tested on three Windows modes and Linux, fail-closed and needs no material change; record that A is the target at the next activation amendment, and decide OD-1 (protect `.gitattributes`, the preflight script and its tests via CODEOWNERS / material list) in the same act.
* **Required evidence:** commit `4aa91dcc0`, CI run `42214173868`, local run (§10), the independent auditor's confirmation that `eol=crlf` is an acceptable reading of "digest meaning unchanged" (handoff §1 item 4).
* **Boundary of approval:** `.gitattributes` line for this one path only; no edit to the activation record, gate, protocol or `MATERIAL_GOVERNANCE_PATHS`; no effect on B6, FRZ, OD-EVOL-12 / OD-EVOLX-08.
* **Draft decision text (UNSIGNED, id to be assigned by the operator, do not treat as a decision):** `Decision ID: <OD-… assigned by operator>` · `Decision scope: W4-F01 ratification of option C (.gitattributes eol=crlf for IIAC_SESSION_HANDOFF_RECORD_2026_11_05_CERTIFICATION.md) as a non-material repair` · `Decision status: <ACCEPTED | REJECTED — A | REJECTED — B>` · signature line left for the operator.
* **Update after this report:** the operator replied in chat "Ratyfikuj C" (option C accepted). A draft record with that instruction is in [`EVOL_X_W4_F01_OPERATOR_RATIFICATION_RECORD.md`](EVOL_X_W4_F01_OPERATOR_RATIFICATION_RECORD.md); decision id and signature line are left to the operator and the file is not committed. The independent auditor still gives its own opinion on C; this does not accept the corrective or close B-4. Section 13 item "W4-F01 ratification" is therefore: decided in chat, record pending signature/commit.
* **Dependencies:** SHA `4aa91dcc0` / `9f8b04d2b`; the independent audit report (should address C); must precede B-4 acceptance by the certifier, not the audit.

## 13. Remaining blockers

**Technical blockers: none found.** Non-blocking items:

| ID | Item | Gates |
|---|---|---|
| W4-F01 ratification | §12 | certification of B-4 (not the audit) |
| W4-F08 | session identities of historical implementers/auditors/certifiers — template `EVOL_X_W4_F08_OPERATOR_ATTESTATION_TEMPLATE.md` is unsigned | known limitation for the auditor; operator act |
| OD-1 | no CODEOWNERS / material protection for the preflight script, tests, `.gitattributes` | operator decision |
| Delta dossier | raw evidence of DA-01/02/09 absent from the workspace | operator should include it in the auditor package if it exists; otherwise the auditor works from the mission text and this report |
| R-1, R-2 (LOW) | unsealed lifecycle-key text; indexed extra files in the certification directory | optional hardening, would need a new corrective ticket |
| R-3 (INFO) | `development` ladder `EVOL_X_SHA_LADDER.md` does not record the W4 chain (0 hits) | update at the closure publication |
| R-4 (INFO) | republication has extra `republication_ticket`; report wording | auditor note |
| R-5 (INFO) | `Publication-Approved` line was written by the implementer session | provenance is process-enforced |
| Housekeeping | local `development` is 14 behind; `9a6b63abf` local-only and stale; worktrees `D:/w4c`, `D:/w4d`, `D:/w4q/*` from earlier sessions remain | operator choice; nothing deleted |
| Out of scope | `ent-audit-fast` red since before this work | independent owner |

## 14. Exact-SHA independent re-audit handoff

* **Auditor:** a **new** `independent-quality-auditor` session, different from `local_4b071fe5-82d0-48bc-b524-146a3daf5c45` (implementer), from every historical session in report §H, and from this recovery session; the certifier is a third, separate session.
* **Baseline** `5afaad9eadfc1655f4ef1ab59eb022960a924648` · **code under audit / TEST_EXECUTION_SHA** `9f8b04d2b8b670077b1e78ce812bb34b680f3968` · **audit head** `75b8ddc87c5047614e5b9c4a083ceee26a6891a4` (= `origin/development`, code-identical, §3). Re-verify with the two empty-diff commands in §3 and check `origin/development` has not moved.
* **Commands** (short `--basetemp`, e.g. `D:/w4a/...`, because of Windows MAX_PATH): `uv sync --extra dev-ci --frozen`; the two pytest commands and the three IIAC/preflight commands in the existing handoff §2; expected: 423 passed, 145 passed + 1 skipped (146 in Linux CI), `IN_FORCE`, exactly two `A-18`.
* **Auditor should additionally:** judge `eol=crlf` (W4-F01) and give an explicit opinion for the operator; try R-1 / R-2 and any further non-listed channel; confirm the P2 anchor via `git log --diff-filter=A`; run `real_scope_probe*.py` / `auditor_probes_fixture.py` (adapt the `SHA` constant); review the pin/CODEOWNERS residual.
* Must not: edit the audited tree, issue B6, promote FRZ, decide OD-EVOL-12 / OD-EVOLX-08.

## 15. Recommended integration sequence

0. **Integration already happened**: the chain is on `origin/development` (fast-forward, operator `Publication-Approved`). Nothing needs merging; a dedicated branch is not required (the audit target is a SHA on `development`; `origin/evolx/w4-01-consolidated` and `qual/t-w4-01-candidate` also exist). Optional: operator tags/branches `75b8ddc87` for the auditor.
1. Operator (optional) fast-forwards local `development`; keep the uncommitted matrix change.
2. Operator decides W4-F01 ratification and OD-1 (ideally before the audit so the auditor can cite them), supplies the delta dossier if it exists.
3. New auditor session audits §14 → findings + manifest.
4. Separate certifier session accepts or rejects; only then record B-4 `RESOLVED_WITH_EVIDENCE` through the evidence-bound `resolved_blockers_for_closure` entry (closure publication after the baseline, per the ledger rules).
5. Operator: W4-F08 attestation (or accepted `UNRECOVERABLE`), then OD-EVOL-12 / OD-EVOLX-08, P3, B6 acceptance, FRZ promotion, formal closure (two-phase SHA publication).
6. If an optional corrective for R-1 / R-2 is wanted, bundle it in **one** ticket after the audit findings so there is a single delta re-audit.

## 16. Roadmap and FRZ impact

None yet. EVOL-X remains `NOT_CLOSED` (roadmap: internally closed scoped; formal ECAP closure `PENDING_AUTHORITY`); `FRZ-EVOL-01..10` stay `OPEN` (confirmed by the canonical Gate B output); ENT-AUDIT-X and ARCH-FREEZE remain blocked on formal EVOL-X closure; `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` and the roadmap were not edited. Next roadmap action: independent re-audit of §14.
