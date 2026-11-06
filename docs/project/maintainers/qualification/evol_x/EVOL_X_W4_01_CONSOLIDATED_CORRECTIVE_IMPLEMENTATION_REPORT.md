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
