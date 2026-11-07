# EVOL-X W4-01 — consolidated corrective implementation report

**Ticket:** [`T-IGX-019-A`](../../../../../governance/tickets/T-IGX-019-A.md) · **Policy:** `IIAC-2026-11-05.1` (`IN_FORCE`, unchanged) · **Role:** implementer
**Implementer session:** `local_4b071fe5-82d0-48bc-b524-146a3daf5c45` (verified from the desktop session API, not inferred) — a new auditor and a later certifier must be different sessions.
**Not done / not claimed:** no self-certification, no B6 acceptance, no FRZ promotion, no OD-EVOL-12 / OD-EVOLX-08 / P3 decision, no change to `open_blockers_for_closure`, EVOL-X stays `NOT_CLOSED`.

## A. Executive verdict

**PARTIAL — ready for independent re-audit; not certified, not published.** Gates not satisfied: G19 (not merged to `development`; branch + draft PR #388 only), and the baseline `ent-audit-fast` failure is outside this change. G11 is satisfied by the Ubuntu runs above.

* Implemented and tested: DA-01, DA-02 / RA-B4-05, W4-F01, W4-F03/F05, W4-F04, W4-F06 / DA-09, W4-F07, HF-01..03 regression coverage, RA-B4-01..04 regression (existing suites unchanged in meaning, see §C).
* Open (authority / external prerequisites, not technical): **W4-F08** (operator attestation of historical session identities), push/publication (see §K), GitHub Actions run on a pushed SHA (G11 — a genuine Linux container run was used instead, see §E).
* Certification readiness: the correction can be audited at the pinned SHA below; EVOL-X closure still needs B-4 independent acceptance, the operator decisions and B6.

## B. SHA ladder

| Role | SHA |
|---|---|
| START_HEAD (= ORIGIN_HEAD at start) | `5afaad9eadfc1655f4ef1ab59eb022960a924648` |
| Reused unpublished candidate (earlier sessions; reviewed here, see §H provenance) | `e0eeca8de` (evidence-only UTF-8 republication) → `742b47cfc` (Part 1) → `e641a134a` (P2 pin) → `fa43976f8` |
| W4-F01 repair | `4aa91dcc0` |
| Consolidated hardening | `a4926c3d2` |
| **IMPLEMENTATION_SHA / TEST_EXECUTION_SHA** | `9f8b04d2b8b670077b1e78ce812bb34b680f3968` |
| EVIDENCE_PUBLICATION_SHA | not created (nothing published) |
| PUBLICATION_HEAD | `origin/development` is still `5afaad9ea`; branch `evolx/w4-01-consolidated` and draft PR #388 are on `origin` (CI evidence only, not merged) |

Historical SHAs checked for existence and ancestry of `5afaad9ea`: `23feb2bbb`, `6cf8c5a22`, `1e28b9e2b`, `81418b0dd`, `cf264f3cf`, `ad0877715` (P2) — all present.
**P2 provenance (DA-01):** `ad0877715ed079ce920ff9d925fda03555b2c04e` is the commit that first added `EVOL_X_IIAC_CERTIFICATION_RECORD.json` (`git log --diff-filter=A`; asserted by a test against the canonical history), bound by `ca2b14782`. It is not taken from any manifest field.

## C. Finding closure matrix

Sources: Audit A (`w4-01-package`, 7 findings) and the operator mission register. The delta-audit dossier `EVOL_X_B4_DELTA_REAUDIT_DOSSIER.md` was **not found** in the local workspace or session store; DA-01 / DA-02 / DA-09 were taken from the mission text and each was reproduced by a test before closing (raw auditor evidence: **not locally verified**).

| ID | Baseline | Root cause → fix | Evidence (tests) | Remaining limitation | Disposition |
|---|---|---|---|---|---|
| DA-01 | CRITICAL, false Gate A PASS by repointing `certification_publication_sha` | Baseline taken from the mutable field. `P2_CERTIFICATION_PUBLICATION_SHA` constant; canonical run rejects overrides; **fixture/`--repo` runs without `--p2-sha` also use the constant (previous fallback to the declared field removed)**; anchor must equal the first publication of the record and be a strict ancestor of HEAD; manifest + record must carry it. | `test_da01_*` (incl. 11-attack matrix × 2 anchor modes, no-fallback, non-ancestor), real-clone `test_da01_real_*` | The constant lives in a script that is not a `MATERIAL_GOVERNANCE_PATHS` entry; changing the pin and the evidence together is detectable only by review (§J OD-1). A pinned 40-hex commit id is a content commitment (SHA-1). | **Fixed in code; independent re-audit required** |
| DA-02 / RA-B4-05 | CRITICAL, semantic scope widening | Substring checks. Exact multiset of `(id,text)` limitations, verdict, FRZ dispositions, promoted-row template, B6 allow-list (earlier commits) **plus**: every non-lifecycle key of the certification record sealed (`stage_id`, `policy_version`, `independent_audit_verdict`, `notes` = evidence-only statement, …), dropped keys fail, manifest byte contract (M0), P2 evidence files (assessment incl. F-C-*, handoff, validator runs) byte-identical with unchanged index entries. | `test_w4_ra_b4_05_*`, `test_da02_*`, real-clone `test_da02_real_*` | `certification_publication_sha_binding` (descriptive text) is a lifecycle key. No amendment channel exists by design. | **Fixed in code; re-audit required** |
| W4-F01 / LF-CRLF | HIGH | `handoff_digest` = SHA-256 of the **CRLF-expanded** handoff (blob is LF). Non-material repair: `text eol=crlf` for that path in `.gitattributes`; activation record, gate, blob unchanged. | Fresh clones `autocrlf=true/false/input` verify `--require-in-force`; tamper (+1 byte, LF-collapsed, double-CR, deleted) → `ACT-N08`; Linux container run (§E) | Working copies cloned earlier may need `git checkout -- <handoff>` once. Behaviour for a checkout that ignores `.gitattributes` is outside the model. | **Fixed, no material path touched** |
| W4-F03 / W4-F05 | MEDIUM | Ledger walks every published version of the record after P2; duplicate ids explicit | `test_w4_f03_*`, matrix `rename-blockers`, `delete-all-blockers-and-repoint` | History rewrite outside model | **Fixed** |
| W4-F04 | LOW | Signature older than P2, negation (NFKC), **plus** non-ASCII/non-printable scope text, revocation-type fields | `test_w4_f04_*` (35) | Signature line is process-enforced text (documented `process_enforced`), OD-record scope still token-contains | **Fixed within scope** |
| W4-F06 / DA-09 | LOW / MEDIUM | Binding blob contains byte `0x97`. Original untouched; append-only `…BINDING.utf8.json` (`e0eeca8de`); `-text` on binding, republication, delegation record | `test_w4_f06_*` | The original blob stays non-UTF-8 by design (immutable) | **Fixed (append-only)** |
| W4-F07 | LOW | Real-repo fresh-clone tests (true/false/input) | `test_evol_x_real_repo_fresh_clone.py` | — | **Fixed** |
| W4-F08 | MEDIUM / GOVERNANCE | Session identity of historical implementers is unrecorded | See §H | Needs operator attestation | **OPEN — `UNVERIFIED / OPERATOR_ATTESTATION_REQUIRED`** |
| RA-B4-01..04 | REGRESSION | Existing tests unchanged in meaning; fixtures adapted only as described in the Part 1 handoff §3 | full B4 suite | — | **No regression observed** |
| HF-01..03 | REGRESSION | Governance suite + CI depth | `tests/unit/governance`, workflow | — | **No regression observed** |
| CI | — | New `evol-x-b4-preflight` job (parallel, xdist-stable ids via `tests/unit/qualification/conftest.py`); `governance-fast` timeout 8→15 min (three more real-repo clones, measured below) | workflow YAML parses; Linux container run | GitHub Actions run not executed (needs a push) | Prepared |

## D. File-change manifest (`git diff 5afaad9ea..9f8b04d2b8b670077b1e78ce812bb34b680f3968`)

| Path | Reason |
|---|---|
| `scripts/qualification/evol_x_closure_publication_preflight.py` | DA-01/DA-02/W4-F03/W4-F04 logic |
| `tests/unit/qualification/test_evol_x_closure_preflight_b4.py` | adversarial matrix and regression tests |
| `tests/unit/qualification/conftest.py` | xdist-stable parametrize ids |
| `tests/unit/governance/test_evol_x_real_repo_fresh_clone.py` | W4-F07/W4-F01/real-evidence attacks |
| `.gitattributes` | `-text` on binding records (W4-F06), `eol=crlf` on the handoff (W4-F01) |
| `.github/workflows/unit-tests.yml` | CI job + timeout |
| `docs/.../IIAC_EVOLX_B4_EVIDENCE_PUBLICATION_BINDING.utf8.json` | evidence-only UTF-8 republication |
| `governance/tickets/T-IGX-019-A.md` | ticket |
| this report, `EVOL_X_W4_01_CONSOLIDATED_REAUDIT_HANDOFF.md` | documentation |

Untouched: `intergrax/`, `agents/`, `applications/`, `scripts/governance/*`, `IIAC_POLICY_ACTIVATION_RECORD.json`, the certification protocol, all existing evidence, `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` (your uncommitted change stayed in the main checkout).

## E. Test execution report

All runs: Windows 11, `core.autocrlf=true`, Python 3.12.11 via `uv` (`--frozen`), short temp paths under `D:\w4c`. Linux evidence is partial (see the last rows).

| Check | Tree | Result |
|---|---|---|
| `tests/unit/qualification/test_evol_x_closure_preflight_b4.py` + `test_evol_x_b4_ruff_evidence_exemption.py`, `-n 4` | `9f8b04d2b` (code identical to `b82c066f7`) | **423 passed**, 0 failed, 10 min 4 s (an earlier run of the same tree hit a full `D:` disk and produced 114 setup errors; invalid, discarded) |
| First B4 run on `a4926c3d2` | `a4926c3d2` | 413 passed, 10 failed: 6 old tests asserting the previous (fallback) behaviour, 1 over-strict `limitations` comparison, 2 own-test bugs, 1 own-test expectation. All fixed in `b82c066f7` |
| `tests/unit/governance`, `-n 4` | `b82c066f7` tree | 144 passed, 1 skipped (symlinks), 1 failed (own new test lacked `--evidence-root`) in 15 min 43 s |
| the six `da01_real*` / `da02_real*` clone tests | `9f8b04d2b` | 6 passed (fix of the failure above) |
| `test_evol_x_real_repo_fresh_clone.py` (W4-F01 part) | `a4926c3d2` ancestry | 9 passed: activation verifies in clones with `autocrlf` true / false / input; tamper cases fail `ACT-N08` |
| `verify_iiac_policy_activation --require-in-force` in a manual `autocrlf=false` and `input` clone | `4aa91dcc0` | `IN_FORCE @ IIAC-2026-11-05.1` |
| canonical `evol_x_closure_publication_preflight.py --gate b6-readiness` | `a4926c3d2` | FAIL with exactly two `A-18` open blockers (B-4, B6-P3); **no A-19** |
| Linux (`python:3.12-slim` container, clone of the branch bundle, `autocrlf` unset, kernel 6.18 WSL2) | `9f8b04d2b` | activation gate: **exit 0, `IN_FORCE`** (LF checkout, the former W4-F01 failure). Governance suite, B4 suite and preflight in the container: **NOT RUN** - the host `D:` disk filled, Docker Desktop's metadata store went read-only and the later retry failed on the same daemon error. Needs a Docker Desktop restart or a GitHub Actions run |
| GitHub Actions on PR #388 (draft, head `445d5883c`; code identical to `9f8b04d2b`), run `42210092666`, Ubuntu | `445d5883c` | `governance-fast`: **146 passed** (6 min 8 s, includes the real-repository clones and the former W4-F01 ACT-N08 tests); `evol-x-b4-preflight`: **423 passed** (1 min 28 s, `-n auto`); `ci-smoke`, `governance-policy-pin`, `ent-audit-pr-gate`: pass |
| GitHub Actions `ent-audit-fast` (run `42210092609`) | `445d5883c` | **FAIL, pre-existing and unrelated**: `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py` (2 tests, "disposition must declare READY, PLAN PARTIAL, or TECHNICAL PARTIAL constant"). Reproduced identically on a clean checkout of the baseline `5afaad9ea` (Windows); this change touches nothing in that area; not fixed here |

Environment notes: no path-length failure was observed with short paths (the deep-path fixture failures reported by the earlier audit were not investigated here). The mission item "tests touching canonical files" is covered by the unchanged HF-02 guard; no test in this change writes into the canonical repository (the working tree of the main checkout stayed clean apart from your pre-existing matrix change).

## F. Adversarial proof matrix (all must be rejected on both gates)

| Attack | Test |
|---|---|
| repoint P2 to HEAD / to an attacker commit / to an older unrelated SHA | `test_da01_adversarial_matrix_*[repoint-*]` |
| repoint and widen LIM-01 at once | `[repoint-and-widen-lim01]` |
| delete all blockers (+repoint) / rename blockers | `[delete-all-blockers-and-repoint]`, `[rename-blockers]` |
| delete and re-add a widened record, repoint to it (P2-like artifact) | `[delete-readd-widened-record-and-repoint]` |
| other stage's record | `[substitute-stage]` |
| status flags only, no evidence | `[status-flags-without-evidence]` |
| manifest changed, index unchanged | `[manifest-changed-index-unchanged]` |
| genuine SHA, mismatched digest | `[genuine-sha-mismatched-digest]` |
| changed trusted anchor (verifier pin = HEAD) | `…_a_changed_trusted_anchor_cannot_rescue_the_attack[*]` |
| fixture run with no explicit anchor | `test_da01_fixture_without_an_explicit_anchor_*` |
| anchor that is not an ancestor | `test_da01_anchor_that_is_not_an_ancestor_of_head_is_rejected` |
| LIM-01..10 append/prepend/replace/drop, `SCOPED`→`UNSCOPED`, FRZ disposition, verdict, scope keys, reorder (positive), duplicates, B6 record fields/limitation digests | `test_w4_ra_b4_05_*` |
| sealed record keys, notes (evidence-only), stage/policy/profile/assurance/verdict, dropped keys, edited/deleted P2 evidence with regenerated index | `test_da02_*` |
| the same on the **real** evidence in fresh clones | `test_da01_real_*`, `test_da02_real_*`, `test_real_preflight_rejects_a_widened_certified_scope_in_a_fresh_clone` |

## G. IIAC authority impact

**No material governance path was changed** (`MATERIAL_GOVERNANCE_PATHS`: `verify_iiac_policy_activation.py`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, the certification protocol). `.gitattributes` is not on that list. `verify_iiac_policy_activation --require-in-force` → `IN_FORCE @ IIAC-2026-11-05.1`; no re-activation or governance re-audit is triggered by this change. The MATERIAL authorization gate of the mission was therefore **not needed**; the independent auditor should still confirm that `eol=crlf` is an acceptable reading of "digest meaning unchanged".

## H. Evidence custody and session provenance

* Original binding `IIAC_EVOLX_B4_EVIDENCE_PUBLICATION_BINDING.json` and the delegation record are byte-identical to `5afaad9ea` (test `test_w4_f06_the_original_binding_and_delegation_record_are_never_edited`). The historical `PARTIAL / EXACT BLOCKERS` verdict is unchanged.
* The UTF-8 republication records the SHA-256 of the original blob and the single transcoding (`0x97` → `U+2014`); it confers no certification authority.
* **W4-F08.** This implementer session id comes from the desktop session API (`get_session self`). For the historical implementers only timing and session titles exist: `23feb2bbb` (2026-11-05 20:31Z) ↔ session "EVOL-X B-4 corrective implementation" `local_e804da4d-…` (last activity 20:40Z); `6cf8c5a22` (06:29Z) ↔ "EVOL-X RA-B4 corrective implementation" `local_4545af5d-…`; the unpublished candidate commits ↔ `local_88e33ba5-…`, `local_7cce176e-…`, `local_db710787-…`. These are **correlations, not proof** (commit author is the operator account for every commit) and are not written into any manifest. Status: `UNVERIFIED / OPERATOR_ATTESTATION_REQUIRED`.

## I. Independent audit handoff

See [`EVOL_X_W4_01_CONSOLIDATED_REAUDIT_HANDOFF.md`](EVOL_X_W4_01_CONSOLIDATED_REAUDIT_HANDOFF.md).

## J. Remaining operator decisions

1. **OD-1 (proposed id, to be assigned by the operator):** decide whether `scripts/qualification/evol_x_closure_publication_preflight.py` (and its test module) joins `MATERIAL_GOVERNANCE_PATHS` / a CODEOWNERS rule so that the P2 pin cannot be edited without independent review. Requires a material change of `iiac_activation_gate.py` and therefore the full IIAC governance re-audit path — **not done here**.
2. **W4-F08 attestation:** operator states the implementer/auditor/certifier session ids of `23feb2bbb`, `6cf8c5a22` and this corrective, or records them as unrecoverable. Text must be signed by the operator; nothing was signed on their behalf.
3. Publication: the push gate needs the ticket `Approved:` line to be the operator's. It currently cites the mission text. If the operator wants a distinct signed approval, replace the line and re-run `igx_gov ticket`.
4. Unchanged and still required before closure: OD-EVOL-12, OD-EVOLX-08, P3 authorization, B6 acceptance, FRZ promotion.

## K. Next roadmap action

**Independent re-audit of this consolidated corrective at `9f8b04d2b8b670077b1e78ce812bb34b680f3968`** (new `independent-quality-auditor` session), after the operator decides on publication/push. Then B-4 acceptance by a separate certifier, then the operator decisions and B6.

## L. Addendum — GitHub Actions evidence and gap closure (2026-11-06; documentation-only commit)

The branch was pushed as `origin/evolx/w4-01-consolidated` (PR `jakbuczarnecki/intergrax#388`, base `development`, not for merge before independent re-audit). `origin/development` is still `5afaad9ea`. The code under audit is unchanged: **IMPLEMENTATION_SHA remains `9f8b04d2b8b670077b1e78ce812bb34b680f3968`**; later commits are documentation only. This addendum supersedes the statements in §A, §B, §E that say "nothing pushed / Actions not run".

| Item | Evidence (run `42210092666` "Regression gate", head `445d5883c`, Ubuntu 24.04) |
|---|---|
| `governance-fast` | success - `146 passed in 338.38s` (Linux checkout, no `core.autocrlf`; includes the W4-F01 activation gate and the real-repository clone tests) |
| `evol-x-b4-preflight` | success - `423 passed in 87.98s` (xdist, isolated tmp repositories) |
| `governance-policy-pin`, `ci-smoke` | success |
| G11 | **satisfied** by the two Linux jobs above (replaces the unfinished container run) |

**`ent-audit-fast` (run `42210092609`) fails — pre-existing and unrelated.** Failing tests: `test_validate_post_implementation_master_plan.py::test_validator_passes_on_canonical_ssot` and `::test_validator_cli_passes` ("disposition must declare READY, PLAN PARTIAL, or TECHNICAL PARTIAL constant"). The same workflow fails on the last five `development` pushes (`2ae101411`, `eab65e010`, `116fb6a2a`, `4d54dfa9e`, `477017f15`), and this branch changes nothing under `scripts/enterprise_audit`, `tests/qualification/ent_audit_x` or the master plan. Not touched here; it is an independent defect for the owner of that validator and must not be read as a result of this corrective.

**Deep-path fixture failures (previously uninvestigated).** Reproduced on Windows with `--basetemp` of ~114 characters: fixtures error in setup (`WinError 3/206`; with ~200 characters `git init` additionally fails with `Filename too long` on `.git/hooks/*.sample`). Cause: Win32 `MAX_PATH` (260) — the fixture repositories nest `docs/project/maintainers/qualification/iiac_evidence/EVOL-X/...` (~140 characters) below the temp root, and `LongPathsEnabled` is off for the Python/git tool chain. It is neither state pollution nor a defect in the gates: `--basetemp` of 80 characters passes, 114 fails, identically with and without an empty git template (an `--template=` experiment was reverted as ineffective). Mitigation: keep `--basetemp` short (the handoff already prescribes `D:/w4a/...`) or enable long paths. Linux CI is unaffected.

**Still open (unchanged, authority items):** W4-F08 operator attestation, OD-1, publication of the branch to `development`, independent re-audit, OD-EVOL-12 / OD-EVOLX-08 / P3, B6, FRZ promotion. The scratch directory `D:/w4d_deep` created for the path experiment can be deleted by the operator.

## M. Publication binding (2026-11-06, after operator authorisation)

The operator authorised fast-forward publication to `development` in chat; the authorisation is recorded in `T-IGX-019-A` (`Publication-Approved:`, scoped to publication only). `igx_gov ticket` passed before the push.

| Role | SHA |
|---|---|
| START_HEAD | `5afaad9eadfc1655f4ef1ab59eb022960a924648` |
| IMPLEMENTATION_SHA / TEST_EXECUTION_SHA (code under audit) | `9f8b04d2b8b670077b1e78ce812bb34b680f3968` |
| Documentation commits after the implementation (no code change) | `445d5883c`, `5376d761a`, `f6d3b6aaf` |
| **PUBLICATION_HEAD** (`origin/development`, fast-forward from `5afaad9ea`, no force) | `49b50fd7442b1977cee53fdc80b7fbd0c33ec203` |
| This binding commit | the commit that adds this section (its own SHA is deliberately not recorded here) |

`git diff 5afaad9ea..49b50fd74 --stat` contains only the files listed in §D, this report, the handoff and the ticket. Publication is **not** acceptance: the auditor still audits `9f8b04d2b`, and `49b50fd74` differs from it by documentation and ticket text only (verify with `git diff 9f8b04d2b 49b50fd74 -- . ':!docs' ':!governance/tickets'`, expected empty).

### W4-F08 — exact operator attestation request (nothing below has been signed on the operator's behalf)

The operator is asked to record, in their own words and signature, one of the following per commit:

> `23feb2bbb`: implementer session id = `<id>` / auditor = `<id>` / certifier = `<id>`, **or** "unrecoverable".
> `6cf8c5a22`: implementer session id = `<id>`, **or** "unrecoverable".
> `9f8b04d2b` (this corrective): implementer = `local_4b071fe5-82d0-48bc-b524-146a3daf5c45` (verified), auditor and certifier must be two different new sessions.

Candidate correlations (timing and titles only, **not proof**) are in §H. Until the operator attests, W4-F08 stays `UNVERIFIED / OPERATOR_ATTESTATION_REQUIRED`.

### Remaining after publication

Independent re-audit (new auditor session) → B-4 acceptance by a separate certifier → OD-1, OD-EVOL-12, OD-EVOLX-08, P3 → B6 → FRZ. The agent that implemented this corrective must not perform any of them.

## N. Round 2 — independent re-audit W401 (verdict PARTIAL on `9f8b04d2b`) and its corrective

Auditor session `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` (new context, own worktree) audited `9f8b04d2b`: **PARTIAL, three MEDIUM blockers**; its dossier is held outside the repository (`.tmp/session/evolx-w4-01-reaudit/` of its worktree). It reproduced B4 423 passed and governance 145 passed / 1 skipped, and judged the `.gitattributes` repair (W4-F01) acceptable on its own reasoning. This implementer did not alter or answer its findings.

| Finding | Severity | Disposition | Fix and test |
|---|---|---|---|
| W401-RA-01 `refs/replace/<P2>` forges the scope baseline | MEDIUM | **Fixed in code** | every git call runs `--no-replace-objects`; replace refs, a grafts file and shallow repositories fail both gates (`A-22` / `B-13`); `test_w401_ra_01_*` (replace ref, grafts, shallow, clean control) |
| W401-RA-02 scope-bearing channels outside the sealed set | MEDIUM | **Fixed in code** | lifecycle keys accept only values of their shape; `certification_publication_sha_binding` only the P2 text or the one binding text `P2_BINDING_TEXT`; the certification directory must hold exactly the P2 file set; duplicate JSON keys are rejected for every evidence JSON; `test_w401_ra_02_*` (9 lifecycle cases, added file, duplicate key, legitimate binding text) |
| W401-RA-03 gates evaluate uncommitted working-tree bytes | MEDIUM | **Fixed in code** | every read input and anything below the EVOL-X evidence root must equal `HEAD` (modified, staged, untracked → `A-21` / `B-12`; repository configuration is used so a normal CRLF checkout stays clean; unrelated dirty files are not flagged); `test_w401_ra_03_*` |
| W401-RA-04 blocker dropped by an `-s ours` merge | LOW | **Fixed in code** | `--full-history` for the blocker history and the first-publication derivation; `test_w401_ra_04_*` |
| W401-RA-06 B6 record / promoted-row formats undocumented | LOW | **Fixed** | `EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md` + drift test |
| W401-RA-05 negation denylist, ASCII-only decision scope | LOW | **Open, not changed** | fail-closed today; the false rejects are a usability issue for Polish operator text; a grammar change would need its own audit |
| W401-RA-07 fixture-class PASS exits 0 | INFO | **Open, not changed** | changing the exit code would alter the contract of every caller |
| W401-RA-08 P2 pin in an ungoverned file | INFO | **Open: operator decision OD-1** | not an exploit of the shipped code |
| W401-RA-09 documentation at the old pin | INFO | **Closed by later documentation commits** | |
| W401-RA-10 Linux container run not reproduced by that auditor | INFO | **Evidence by CI** | run `42214173868` on `49b50fd74`; a new run is needed for the round-2 SHA |

**Pinned SHA for round 2: `ca78b1c1f`** (code and tests; the commits after it are documentation only). Evidence on that tree (Windows 11, `core.autocrlf=true`, short `--basetemp`):

* new tests: 24 `w401` tests passed, plus the format-drift test (1 passed);
* full B4 suite (`-n 6`) on the working tree before the commit: **445 passed**, 2 failed — both canonical-repository assertions failed only because the tree was uncommitted (`A-21` working as designed); the same assertions then **passed** on the committed tree (`-k canonical`: 20 passed). The full B4 suite was not re-run end to end after the commit; CI does that;
* `tests/unit/governance -n 4`: **145 passed, 1 skipped** (symlinks), 7 min 45 s;
* canonical `--gate b6-readiness`: FAIL with exactly the two real `A-18` blockers (B-4, B6-P3), no other code; `verify_iiac_policy_activation --require-in-force`: `IN_FORCE @ IIAC-2026-11-05.1`.

No `intergrax/`, `agents/`, `applications/`, `scripts/governance/*`, activation record, protocol or existing evidence changed. A canonical or `--repo` run now requires the verified inputs to be committed: operators must commit evidence before running the preflight.

## O. Round 3 — delta re-audit W402 (verdict PARTIAL on `ca78b1c1f`) and the defect hunt

The delta re-audit by `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` closed W401-RA-01, RA-04 and RA-06 (PASS) and left RA-02 and RA-03 PARTIAL with one MEDIUM blocker, **W402-01** (a tracked input hidden by `skip-worktree` / `assume-unchanged` passed the HEAD-binding because the check trusted `git status`). In parallel this implementer ran a read-only adversarial review by a subagent on `ca78b1c1f`; its findings D1-D9 are folded in below (the subagent's report is not an audit and certifies nothing).

| Finding | Source | Disposition | Fix and test |
|---|---|---|---|
| W402-01 index flags / fsmonitor hide an uncommitted input | auditor MEDIUM, hunt D2 | **Fixed in code** | `git status` is no longer the oracle: every verified input is compared byte for byte with its HEAD blob (RA-B4-01 byte contract), untracked files are found by walking the file system, and index flags other than `H` fail; `test_w402_01_*` |
| W402-02 verifier code outside the read inputs | auditor LOW, hunt D3 | **Fixed in code** | the whole `scripts/` tree, the EVOL-X evidence tree and `.gitattributes` are HEAD-bound; `test_w402_02_*`, `test_hunt_d3_*` |
| W402-03 free-text channels | auditor LOW, hunt D7 | **Fixed in code** | open blockers keep their published text; resolved entries and `certifier_disposition` accept only structured keys; `stage_closure_status` and `b6_acceptance` are exact enums; padded SHAs and `NaN`/`Infinity` rejected; the evidence root holds only known or index-bound directories; `test_w402_03_*`, `test_hunt_d7_*` |
| W402-04 / D1 forged commit-graph, fsmonitor | auditor INFO, hunt D1 | **Fixed in code** | `core.commitGraph=false` and `core.fsmonitor=false` on every preflight git call; `test_w402_04_*` (the forged-graph repro itself was not re-executed; the setting disables the graph) |
| D4 planted git-ignored module on `sys.path[0]` | hunt MEDIUM | **Fixed and checked by hand** | the script directory is removed from `sys.path` before any import; a planted ignored `json.py` in a fresh clone was not executed and the run exited 1 with an `A-21` untracked-file error; `test_hunt_d4_*` |
| D5 `GIT_REPLACE_REF_BASE`, imported modules' git calls | hunt MEDIUM | **Fixed in code** | `GIT_NO_REPLACE_OBJECTS=1` for the whole process (imported IIAC modules included); redirecting git variables (`GIT_REPLACE_REF_BASE`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_OBJECT_DIRECTORY`, `GIT_NAMESPACE`, `GIT_SHALLOW_FILE`, `GIT_GRAFT_FILE`) fail both gates; `test_hunt_d5_*` |
| D6 crash on unhashable lifecycle value | hunt LOW | **Fixed in code** | values are type-checked before they are hashed; any unexpected evidence shape is an `A-99` internal error with a structured, failing verdict; `test_hunt_d6_*`, `test_hunt_an_unexpected_*` |
| D8 gitlinks / non-ASCII names in the certification directory | hunt LOW | **Fixed in code** | `ls-tree -r -z` over every entry type; `test_hunt_d8_*` |
| D9 / W401-RA-07 fixture PASS exits 0 | INFO | **Open, not changed** | same reason as before |
| `GIT_DIR` / `GIT_INDEX_FILE` redirection, forged commit-graph in the *imported* activation modules | — | **UNVERIFIED / residual** | the imported material IIAC modules (not changed here) still call git without `core.commitGraph=false`; the process-wide `GIT_NO_REPLACE_OBJECTS` covers replace objects only |

One existing test changed meaning on purpose: `test_open_blocker_holds_formal_closure` used to reword blocker B-5; it now re-opens B-5 with its P2 text, because rewording an open blocker is the channel W402-03 closes.

**Pinned SHA for round 3: `2e0ed9c6b`** (code and tests; commits after it are documentation only). Evidence on that tree (Windows 11, `core.autocrlf=true`, short `--basetemp`):

* full B4 suite (`-n 6`) at `1b89c47b3`: 480 passed, 1 failed (the stale fixture above); that test passes after the fix in `2e0ed9c6b` and the 20 `canonical` tests pass on the committed tree. The whole suite was not repeated end to end after the one-test change; CI does that;
* `tests/unit/governance -n 4`: **145 passed, 1 skipped** (symlinks), 14 min 39 s;
* canonical `--gate b6-readiness`: FAIL with exactly the two real `A-18` blockers and no other code; `verify_iiac_policy_activation --require-in-force`: `IN_FORCE @ IIAC-2026-11-05.1`;
* no `intergrax/`, `agents/`, `applications/`, `scripts/governance/*`, activation record, protocol or existing evidence changed.

## P. Round 4 — round-3 delta re-audit W403 (verdict PARTIAL on `2e0ed9c6b`)

The auditor closed every W402 repro and found one MEDIUM blocker, **W403-01**: a forged `UNCHECKED_HASH` `.pyc` in a git-ignored `__pycache__` replaces a committed verifier module without touching the source, `git status` or the byte comparison, and can make the canonical clone print PASS.

| Finding | Severity | Disposition | Fix and test |
|---|---|---|---|
| W403-01 forged bytecode in `__pycache__` | MEDIUM | **Fixed in code** | the preflight sets `sys.dont_write_bytecode` and a `sys.pycache_prefix` outside the repository before any import, so no repository bytecode is read or written and every imported module is compiled from the HEAD-bound source; `test_w403_01_*` plants a forged `.pyc` in a copy of `scripts/` (control: a plain import does run it; the preflight does not) |
| W403-02 residual free-text channels | LOW | **Fixed in code** | `closure_evidence` items accept only `path`/`sha256`/`decision_id`; the audit directory is frozen like the certification directory (gitlinks included); a new evidence directory must be named by a blocker resolution (`closure_evidence_index_path`); every file of an indexed directory is listed in its index; `test_w403_02_*`. Not covered: free files inside `operator_decisions/` (no index exists for that directory) |
| W403-03 whole-`scripts/` binding rejects unrelated files, slow | LOW | **Fixed in code** | the binding covers `scripts/governance`, `scripts/qualification`, `scripts/__init__.py`, `.gitattributes` and the modules actually loaded from `scripts/` or `intergrax/`; unrelated scripts no longer block the run; `test_w403_03_*`. A canonical run takes about 30 s on this machine |
| W403-04 forged commit-graph, `GIT_DIR`/`GIT_INDEX_FILE`, git calls of the imported (material) IIAC modules | INFO | **UNVERIFIED / residual** | unchanged: the imported modules are material paths and were not edited |

**Pinned SHA for round 4: `dc650b8df`** (code and tests; later commits are documentation only). Evidence on that tree (Windows 11, `core.autocrlf=true`, short `--basetemp`, `-n 6`): B4 + ruff-exemption + governance suites together **634 passed, 1 skipped** (symlinks), 30 min 31 s; canonical `--gate b6-readiness` FAIL with exactly the two real `A-18` blockers; no `intergrax/`, `agents/`, `applications/`, `scripts/governance/*`, activation record, protocol or existing evidence changed.

**Threat-model note for the next audit and for the operator.** Rounds 2-4 each moved the same boundary: what a person with *write access to the checkout* (working tree, index, git config, ignored caches) can do. Every such attack found so far is now closed, but the class is open-ended by nature; the preflight is a fail-closed guard against accidental or careless divergence from the committed publication, not a defence against an attacker who controls the machine that runs it. A certifier should run it on a fresh clone of the pinned SHA, in a clean environment, which makes this class moot. Recording that boundary is an operator decision (suggested id: OD-2, not decided here).

## Q. Round 5 — round-4 re-audit W404 (verdict PARTIAL on `dc650b8df`, HIGH regression)

The auditor found that the round-4 change itself introduced **W404-01 (HIGH)**: `import tempfile` ran before the `sys.path` strip, so an untracked `random.py` / `shutil.py` next to the script executed first and could force PASS with the documented command. This was a regression of the round-3 protection (D4) and is reported here as such.

| Finding | Severity | Capability | Disposition | Fix and test |
|---|---|---|---|---|
| W404-01 `tempfile` import before the strip | HIGH | untracked file | **Fixed in code, reproduced by hand** | the bootstrap imports only `os`/`sys` before stripping the script directory (and, as a program, the cwd and the repository root); the bytecode prefix is computed without `tempfile`; `test_w404_01_*` (random, shutil, tempfile, json, argparse, hashlib, each with a control that the plain import does run the planted file); on a fresh clone the documented command with a planted `random.py` exits 1 with an `A-21` untracked-file error and runs nothing |
| W404-02 repository-root file shadows the stdlib (`python -m`, `PYTHONPATH`) | MEDIUM | ignored root file | **Fixed in code** | the repository `scripts` package is registered explicitly from this checkout (editable installs, `PYTHONPATH` and root files cannot supply another one) and the IIAC modules must come from it; the repository root is never put before the standard library; an untracked repository-root module (`.py`/`.pyc`/`.pyd` or package) fails the HEAD binding; `test_w404_02_*` |
| W404-03 companion subprocesses load forged bytecode | LOW | ignored `.pyc` | **Fixed in code** | `PYTHONDONTWRITEBYTECODE`, `PYTHONPYCACHEPREFIX` and `PYTHONSAFEPATH=1` are exported to children; `test_w404_03_*`. The companion scripts still need the repository root on `PYTHONPATH` (they import `scripts.governance.*` without their own `sys.path` handling), which is why a stray root module is flagged instead of being removed from their path |
| W404-04 free text in `operator_decisions/` and an edited `b4_reaudit/` index | LOW | committed file | **Open, accepted residual** | descriptive channels not read by any gate; freezing `b4_reaudit/` would need a pinned first-publication commit |
| W404-05 Python-environment hooks (`sitecustomize`, `usercustomize`, `.pth`, replaced interpreter) | INFO | environment | **Outside what the script can defend** | needs the operator's threat-model decision (suggested OD-2); mitigation: run the preflight on a fresh clone in a clean environment |

**Pinned SHA for round 5: `fdb1e7a62`** (code and tests; later commits are documentation only). Evidence on that tree (Windows 11, `core.autocrlf=true`, short `--basetemp`, `-n 6`): B4 + ruff-exemption + governance suites together **646 passed, 1 skipped** (symlinks), 28 min 15 s; canonical `--gate b6-readiness` FAIL with exactly the two real `A-18` blockers; no `intergrax/`, `agents/`, `applications/`, `scripts/governance/*`, activation record, protocol or existing evidence changed.

A note on process: three consecutive fixes (rounds 3, 4, 5) each opened a neighbouring hole in the same bootstrap/binding code, and round 4 introduced a regression. The remaining findings are in the environment class (W404-05) that no in-script check can close. Recommended: the operator decides OD-2 (the threat-model boundary) before another corrective round, and the certifier runs the preflight on a fresh clone of the pinned SHA.

## R. Round-5 re-audit result (`fdb1e7a62`): PARTIAL, no blocker

The independent auditor (`local_e360e892-825d-4e0f-81f6-b4dc46ed4df9`) re-audited `fdb1e7a62`: **all three targeted findings (W404-01, W404-02, W404-03) PASS; `exact_blockers_for_acceptance` is empty.** Its own run: B4 501 passed (489 + 12 new, no existing test weakened), governance 145 passed / 1 skipped, canonical `b6-readiness` FAIL with exactly the two `A-18` blockers, clean real-chain clones with `core.autocrlf` true/false/input. The overall verdict stays **PARTIAL** only because of LOW/INFO residuals and the undecided T4 boundary. The auditor's dossier and findings are held in its worktree (`.tmp/session/evolx-w4-01-reaudit/`, round-5 files) and have not been published to the repository; publishing them is the next step and belongs to a publication session, not to this implementer.

| Finding | Severity | Capability | Disposition |
|---|---|---|---|
| W405-01 companion subprocesses inherit `PYTHONPATH`, so a repository-root module shadows the standard library inside them | LOW | T2 + `PYTHONPATH=<repo>` | **Accepted residual** (OD-2): the companions need the repository root on their path; dropping it breaks them; the stray module is flagged by `A-21` and their exit codes cannot produce a PASS |
| W405-02 any untracked root-level Python file or package blocks both gates | LOW | none (usability) | **Accepted** (auditor: acceptable as is) |
| W405-03 interpreter-level hooks (`.pth`, `sitecustomize`, the git-ignored project venv inside the checkout) force PASS | INFO | T4 | **Outside the script; decided by OD-2** — mandatory procedure: certifier runs on a fresh clone of the pinned SHA in a clean environment |

Open and unchanged: W404-04, W401-RA-05, W401-RA-07, W403-04 (unverified), OD-1 (decided C). OD-2 is recorded in `EVOL_X_OD2_DECISION_RECORD.md` as a delegated entry, the same way as OD-1.

**Final pin for acceptance: `fdb1e7a62`** (code and tests). No further corrective round is proposed: a further round would be spent on LOW residuals and the T4 class that the script cannot close. Open before certification: publication of the auditor's evidence, the certifier session (a third, different session; its id goes into the W4-F08 record), the certifier's run on a fresh clone as defined in OD-2, and then B-4 acceptance.


## S. Certifier result and stage closure (2026-11-07)

The third session `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18` ran the OD-2 procedure on a fresh clone of `fdb1e7a62` (and at `bec0a0948`, which differs only in documents). Gates, activation verifier, B4 suite (493 passed) and governance suite (145 passed, 1 skipped) matched the auditor's numbers; its own attacks in tiers T1-T3 were rejected. Verdict **ACCEPT_WITH_CONDITIONS** (`EVOL_X_W4_01_CERTIFIER_REPORT.md`, published at `b59052993`). Its new findings: C-01 (OD-2 environment wording cannot be literal; amended in OD-2), C-02 (OD-1/OD-2 were decided by the implementer; ratified by the operator on 2026-11-07), C-03 (the real B-4 resolution has not been exercised; dry run assigned to the certifier), C-04, C-05 (informational).

The operator's acceptance is recorded in `EVOL_X_OD3_DECISION_RECORD.md`. The W4-F08 certifier row is filled. This closes the W4-01 corrective stage at the level of decisions and evidence. Still not done and not done by this report: the machine publication of the B-4 resolution (condition 2), P3 / B6 acceptance, FRZ promotion, OD-EVOL-12, OD-EVOLX-08 and EVOL-X closure.
