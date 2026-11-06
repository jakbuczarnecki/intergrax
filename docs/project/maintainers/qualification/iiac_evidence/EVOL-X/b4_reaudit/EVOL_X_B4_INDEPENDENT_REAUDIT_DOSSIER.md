# EVOL-X B-4 — Independent re-audit dossier

**Stage:** `EVOL-X` · **Subject:** B-4 corrective implementation (`T-IGX-017-A`) · **Policy:** `IIAC-2026-11-05.1` (`IN_FORCE`, verified) · **Date:** 2026-11-05
**Auditor role:** `independent-quality-auditor` (`may_commit: false`) · **Session:** `IIAC-C-EVOL-X-B4-REAUDIT-2026-11-05__claude-code-cf31a7e9-08ee-47f4-84ca-5b420d9a377d` (id taken from this session's scratchpad path; the operator attests it in the handoff record)

## 0. Terminal verdict

**`EVOL-X B-4 — PARTIAL / EXACT BLOCKERS`**

The correction removes the original B-4 defect (circular B6 readiness ↔ closure, filename-based IIAC acceptance, superseded human gates) and keeps Session C evidence, P2 evidence and the IIAC activation intact. It does **not** satisfy the full B-4 acceptance contract: independent reproduction found one environment-dependent false rejection and three unsafe acceptances introduced by the new code (RA-B4-01..04). **The original B-4 blocker is not eligible for closure by the certifier.** One narrow corrective plus a delta re-audit of those four items is required; nothing else needs rework.

This audit did not edit the audited implementation, the certification record, any evidence, any ticket or any tracked file. It issued no B6 acceptance, promoted no FRZ row, authorised no operator decision and did not close EVOL-X.

## 1. SHA lineage (verified)

| Role | SHA | Check |
|---|---|---|
| Baseline before B-4 | `ca2b147825ce6117cc7219e63a93d2872669cb49` | ancestor of implementation SHA |
| `B4_CORRECTIVE_IMPLEMENTATION_SHA` | `23feb2bbb017b1beb63f9368b5978384093ff924` | the code audited; ancestor of HEAD |
| Binding commit = HEAD = `origin/development` | `9a8fa3634713f69b56ce87ceaf9ed319fad27dae` | `git rev-parse HEAD origin/development` identical at audit start (and re-checked at the end, after `git fetch`) |
| Session C / P1 / P2 / TES / FINAL | `4dee0050d` · `ad0877715` · `3c4a641b` · `bbf3e33b` | unchanged |

- `git log ca2b14782..HEAD` = exactly two commits (`23feb2bbb`, `9a8fa3634`).
- Binding commit `23feb2bbb..9a8fa3634` changes only `EVOL_X_B4_CORRECTIVE_REAUDIT_HANDOFF.md` (SHA rows + risk R-7) and `T-IGX-017-A.md` (checkpoint). No unaudited semantic change.
- `git diff ca2b14782 HEAD`: 9 files, +2311/−82. Zero paths under `intergrax/`, `agents/`, `applications/`, `iiac_evidence/`, the activation record, `iiac_activation_gate`, `iiac_common`, `verify_iiac_*`, the protocol. `iiac_evidence/**` is byte-identical to `ca2b14782`; audit evidence is byte-identical to P1 `4dee0050d`; the certification directory differs from P2 `ad0877715` only by the pre-B-4 SHA-binding commit (3 files, 6/6 lines, `certification_publication_sha` + binding note).
- Activation record blob `c505dd5b…` = HEAD blob before and after every run in the canonical repository. All destructive experiments ran in disposable clones (`C:\ia\h`, `C:\ia\b`, `C:\ia\x`) made from a bundle of `development`.

## 2. Reviewed diff

| File | Change |
|---|---|
| `scripts/qualification/evol_x_closure_publication_preflight.py` | rewritten (1030 lines): `policy_mode`, `gate_a` (A-01..A-18), `gate_b` (B-00..B-10), legacy diagnostic |
| `scripts/ci/push_gate_config.py`, `run_pre_commit_gate.py`, `run_ruff_changed_paths.py`, `pyproject.toml` | Ruff exemption for `iiac_evidence` |
| `tests/unit/qualification/test_evol_x_closure_preflight_b4.py` (51), `test_evol_x_b4_ruff_evidence_exemption.py` (8) | new tests |
| `governance/tickets/T-IGX-017-A.md`, handoff | documentation |

## 3. Method

Handoff assertions were treated as claims. Evidence came from: (a) code reading of the whole preflight; (b) the real repository, read-only; (c) clones of the real evidence with edits committed only inside the clone (to exercise the real bytes, not just the implementer's builders); (d) synthetic git fixtures built with the implementer's builder functions but with my own scenarios and assertions (`exp_fixture*.py`); (e) fresh clones with default Windows Git configuration.

Commands and results are in `EVOL_X_B4_INDEPENDENT_REAUDIT_TEST_MATRIX.txt`.

## 4. Gate A — B6 readiness (A-01..A-18)

| Control | Disposition | Independent evidence |
|---|---|---|
| A-01 genuine `IN_FORCE` | **PASS** | Delegates to `evaluate_activation(require_in_force=True)`. Forged/edited record committed with altered `operator_decision_id` → `INVALID` (phase-1 binding + OD id); uncommitted worktree edit → `INVALID` (`does not match HEAD`, digest mismatch); `--activation-record` outside the repo → `INVALID`; `--evidence-root` on canonical → rejected; `status` removed or `NOT_IN_FORCE` → `LEGACY`, which can never pass. A forged `IN_FORCE` value cannot bypass the gate. |
| A-02 dual activation | PASS (code; implementer test) | ECAP-P4 `IN_FORCE` → error. |
| A-03/A-04/A-05 manifest schema, companions, stage/policy/profile/assurance | PASS | canonical `validate_manifest_dict`, `verify_iiac_audit_manifest`, `verify_iiac_role_separation --strict-product-diff` re-run: PASS. |
| A-06/A-07/A-08 verdict, auditor-draft binding, scope restrictions | PASS for removal/rewording, **limited** for widening | `PASS_SCOPED→PASS` and dropping/rewording LIM-01..05 are rejected. Appended widening is not (RA-B4-05, accepted limitation). |
| A-09 role/agent separation | PASS | certifier session = auditor session → A-04 + A-09. |
| A-10/A-11 SHAs, ancestry, freshness | PASS | post-audit change under audited scope → A-11. |
| A-12/A-13 bytes vs SHA-256, indexes, blobs | PASS | dossier tampered (uncommitted and committed) → A-12 + A-13; injected blocking finding → A-12/A-13/A-15. |
| A-14 pass record | **FAIL on CRLF checkouts** | RA-B4-01. |
| A-15 findings | PASS | |
| A-16 handoff attested | PASS | |
| A-17 certification record | PASS | verdict widening to `…UNRESTRICTED…` not detected (RA-B4-05). |
| A-18 open blockers | **Insufficient** | RA-B4-02. |

### 4.1 Blocker-state transition (isolated; real certification record untouched)

| Case | Result |
|---|---|
| T1 B-4 open, P3 open | Gate A **FAIL** |
| T2 B-4 open, P3 authorised | Gate A **FAIL** |
| T3 B-4 resolved, P3 **not** authorised | Gate A **FAIL** |
| T4 B-4 resolved + P3 authorised (B-5 open) | Gate A **PASS**, Gate B **FAIL** (B-01, B-02, B-06, B-07, B-09) |
| T5 B-4 line rewritten to `B-4 RESOLVED …` | Gate A **FAIL** (status text alone is insufficient) |
| T5c unprefixed blocker / T5d non-list | Gate A **FAIL** |
| T5e field absent | Gate A **PASS** (RA-B4-02) |
| Real evidence, two blocker lines deleted | Gate A **PASS** with no evidence reference (RA-B4-02) |

Gate A PASS never implies Gate B PASS: `gate_b_pass` requires both, and every Gate A PASS case above left Gate B FAIL.

## 5. Gate B — formal closure (B-01..B-10)

| Control | Disposition | Evidence |
|---|---|---|
| Gate A prerequisite (B-00) | PASS | |
| B-01..B-04 B6 published, two-phase SHA | PASS on SHA mechanics, **weak on genuineness** | readiness without a B6 SHA → Gate B FAIL (G2); B6 SHA = certification publication, = pre-certification commit, = unrelated no-op commit → FAIL (G3, G4, G5). But the acceptance itself is three self-asserted flags (RA-B4-04, fixture G6). |
| B-05 | **no such control** — Gate B has nine controls (B-00 prerequisite, B-01..B-04, B-06..B-10); all requirement areas are covered, the handoff's "B-01..B-10" is inaccurate | RA-B4-10 (informational) |
| B-06 FRZ-EVOL-01..10 individual dispositions | PASS for missing/OPEN/promotion mismatch; **limited** for semantic reversal | G9-G11 (RA-B4-05). |
| B-07 operator decisions | **FAIL (false acceptance)** | RA-B4-03. |
| B-08 ladder, FORMAL_CLOSURE SHA | PASS | formal-closure readiness passes without its own SHA (G1); an older closure SHA fails (G7); a post-B6 ancestor passes (G8). |
| B-09 no open blocker | PASS | |
| B-10 no premature closure claim | PASS (code; implementer test) | |
| Complete positive path | PASS | G1: A PASS, B PASS on a synthetic repo; real repository state untouched. |

## 6. Scope review (semantic)

Session C restrictions are enforced against removal and rewording, not against contradiction. With the real evidence cloned and only the scope checks deciding, all of the following kept Gate A PASS: appended "platform-wide test ownership is established" (LIM-03/F-C-03), appended "production ActivationService.rollback enforcement is in force" (LIM-05), "M-09H was measured" (LIM-02), "M-03 generalises" (LIM-04), FRZ-EVOL-07 disposition "… enforcement IS claimed", FRZ-EVOL-01 "unrestricted ownership", certification verdict `PASS_SCOPED_UNRESTRICTED_PLATFORM_WIDE`, and dropping every certifier-authored limitation after P2. Gate B fixtures G9-G11 passed with FRZ-07/-08/-01 rows contradicting their scope but retaining the tokens.

Determination (RA-B4-05): not a bypass of the independent authority — the digest-bound auditor draft, pass record and dossier remain and the original restrictions stay visible as prefix — and it is the documented limitation R-1. Text presence is therefore **not** a semantic guarantee, and a gate PASS must not be read as one. The post-publication mutability is cheap to close (compare to the blob at `certification_publication_sha`) and is recommended in the same corrective.

`PASS_SCOPED → PASS`, a missing LIM id, a rewritten LIM-02..05 phrase, a FRZ row promoted without its LIM id / mandatory phrase, and an OPEN row are all rejected.

## 7. Ruff gate assessment — **PASS**

- Exemption is the single path `docs/project/maintainers/qualification/iiac_evidence` (`pyproject.toml extend-exclude`), consumed by both gates through `split_ruff_exempt`.
- Path matrix: `iiac_evidence/**` exempt; `iiac_evidence_x/`, `iiac_evidence.py`, `IIAC_EVIDENCE/`, `evol_x/misplaced_probe.py`, `scripts/**`, bare `iiac_evidence/` at repo root are linted.
- Staged-file simulation of `run_ruff_changed_paths.py --mode staged` and `run_pre_commit_gate.py`: the evidence probe is reported exempt, the misplaced file is passed to `ruff check` and `ruff format --check`. No skip variable is involved.
- `ruff check docs/project/maintainers/qualification` (directory discovery): the misplaced file produces E401/I001/F401×2, the file under `iiac_evidence` is not reported.
- The six B-4 changed Python files pass `ruff check` and `ruff format --check`. (A broader `ruff check scripts/ci` shows 7 errors and 5 unformatted files in unrelated pre-existing scripts; not B-4.)
- The 7 probe scripts and all of `iiac_evidence/**` are byte-identical to P1 and to `ca2b14782`.
- Informational (RA-B4-07): no path normalisation (`..`) — unreachable because git emits normalised paths; the exemption covers every `.py` under `iiac_evidence/`, not only the probes.

## 8. The four pre-existing failures

Reproduced at `ca2b14782` and at `9a8fa3634` (clones, LF checkout): **4 failed in each**, identical assertion text.

| Test | Root cause | Class |
|---|---|---|
| `test_iiac_activation_gate::test_act_n01_inactive_policy_fails_default_and_require` | expects the canonical verifier to fail; the policy is legitimately `IN_FORCE` | 1 stale assumption |
| `test_iiac_activation_gate::test_positive_fixture_passes_without_activating_canonical_policy` | asserts canonical `NOT_IN_FORCE` | 1 stale assumption |
| `test_iiac_negative_guards::test_n12_session_a_policy_not_in_force_by_default` | same | 1 stale assumption |
| `test_iiac_negative_guards::test_n09_implementer_must_not_publish_pass_filename_glob` | forbids any pass record by filename; Session C records are legitimate | 1 stale assumption (guard obsolete) |

Additional findings:
- **CRLF pollution (class 2/4, HF-02).** `test_n08` (activation record, `-text`) and `test_n05` (`OD_ENT_02_THREAT_MODEL.md`) overwrite canonical files and restore them with `write_text`, which writes CRLF on Windows. Reproduced in a clone: activation blob `c505dd5b…` → `c067ebc4…`, `verify_iiac_policy_activation` → `worktree activation record does not match HEAD`, preflight mode `INVALID`. Restored with `git checkout`. In the canonical repository the blob stayed `c505dd5b…` after every run I performed (I never ran `tests/unit/governance` there).
- **CI (class 4, HF-03).** `governance-fast` on HEAD (run `42104326792`): **6 failed, 120 passed** — the four above plus two `test_iiac_g0_inventory` tests failing on a depth-1 checkout (`not ancestor of HEAD`, `bad object`). Locally G0 passes. The handoff's "4" is the local count. No CI run exists for `ca2b14782`; the CI-only pair is inferred, not reproduced, as pre-B-4.
- No actual regression from B-4 in any of them. Separate governance test-hotfix ticket specification: `GOVERNANCE_TEST_HOTFIX_TICKET_SPEC.md`. Those tests were not modified.

## 9. Independent qualification results (exact)

| Run (where) | Command | Result |
|---|---|---|
| B-4 + Ruff exemption tests (canonical) | `pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py` | **59 passed**, 0 failed, 0 skipped (117 s) |
| EVOL-X qualification + unit/qualification (canonical) | `pytest tests/qualification/evol_x tests/unit/qualification -rs` | **384 passed**, 0 failed, 0 skipped (695 s); includes the 59 above |
| Governance unit suite (clone, LF checkout) | `pytest tests/unit/governance -rs` | **6 failed, 119 passed, 1 skipped** (496 s). 4 = the stale tests of §8. The other 2 (`test_n05`, `test_n08`) are artefacts of the clone: their verifier subprocesses import `scripts` from the canonical editable install, so they judged the canonical tree; rerun with `PYTHONPATH=<clone>` both pass (and leave the clone's activation record and threat model modified = HF-02). The two G0 tests that fail in CI pass here. Skip: symlink creation unavailable |
| Same canonical B-4 test in a **fresh default-config clone** (`core.autocrlf=true`) | `pytest …::test_canonical_repository_formal_closure_fails_closed_while_b6_unpublished` | **FAILED** (A-14; RA-B4-01) |
| Four named tests at `ca2b14782` / `9a8fa3634` (clones) | see §8 | 4 failed / 4 failed |
| `verify_iiac_policy_activation --require-in-force`, `verify_iiac_audit_manifest`, `verify_iiac_role_separation --strict-product-diff`, `verify_iiac_g0_inventory_completeness` (canonical) | | all PASS (G0: 17510 paths) |
| Preflight, canonical, `--gate b6-readiness` / default | | FAIL / FAIL; Gate A errors = the two A-18 entries only (LF worktree) |
| Adversarial experiments | `exp_real.py` (22 cases on real-evidence clones; `batch2` runs via `python exp_real.py b2` logic), `exp_fixture.py` (18 gate cases + 8 register strings), `exp_fixture2.py` (4 cases) | outcomes in §4-§7 |

The implementer's `505 passed, 4 failed, 1 skipped` is **not** an overall green run and was not reproduced as a single run; my numbers above are separate runs with accurate counts. The full platform suite was not rerun (no material impact outside the preflight, the CI helpers and their tests).

## 10. Findings that block

| ID | Sev | Blocks | One line |
|---|---|---|---|
| RA-B4-01 | HIGH | B-4 acceptance, B6 admission on CRLF checkouts | A-14 byte compare of the pass record fails in any default Windows checkout |
| RA-B4-02 | MEDIUM | B-4 acceptance, B6 admission | A-18 closes blockers by absence; no independent evidence reference; missing field passes |
| RA-B4-03 | MEDIUM | B-4 acceptance, formal closure | `decision_recorded` accepts `NOT APPROVED`, `REVOKED (previously APPROVED)`, … |
| RA-B4-04 | MEDIUM | B-4 acceptance, formal closure | Gate B accepts self-asserted B6 flags with no acceptance record |

Not blocking: RA-B4-05 (accepted limitation + hardening), RA-B4-06/07/08/09/10 (informational/hardening). Historical/not introduced by B-4: HF-01 (4 stale tests), HF-02 (pollution), HF-03 (CI depth). Full register: `EVOL_X_B4_INDEPENDENT_REAUDIT_FINDINGS.json`.

## 11. What is confirmed sound

Two-gate separation (Gate A never needs a B6 SHA; Gate B never needs its own closure SHA); genuine `IN_FORCE` validation; evidence bytes ↔ SHA-256 ↔ git blobs; exact-SHA ancestry and staleness; role/agent separation; PASS_SCOPED→PASS and limitation removal rejected; legacy human gates no longer required and legacy mode cannot pass; ECAP-P4 dual activation; preserved Session C/P2 evidence; Ruff exemption boundary; no product or IIAC-governance path touched.

## 12. Closure criteria and handoff to the certifier

1. Implementer corrects RA-B4-01..04 (RA-B4-05 immutability part and RA-B4-06 recommended) in one ticket, with a test that clones with `core.autocrlf=true`.
2. A delta re-audit by a separate `independent-quality-auditor` session re-runs the four reproductions and the full B-4 test file.
3. Only then may the certifier remove B-4 from `open_blockers_for_closure`, citing that re-audit; until the evidence-reference mechanism of RA-B4-02 exists, the certifier must record the citation in the P3 publication text.
4. This audit's PASS-equivalent is **not** issued; no PASS record is created here. The certifier must not treat this dossier as authority to close B-4, issue B6, promote FRZ rows or decide OD-EVOL-12 / OD-EVOLX-08.

## 13. Scope and limits of this audit

- Gate PASS is not a semantic guarantee (RA-B4-05). Operator identity is not verified (assurance `process_enforced`, LIM-10).
- Not run: the entire platform suite; CI at the corrective SHA (no run exists); Linux behaviour (inferred from CI-level facts only; nothing in the preflight is OS-specific besides the EOL issue above).
- Session id of this audit comes from the session scratchpad path and awaits operator attestation.
- Auditor outputs are untracked files in `iiac_evidence/EVOL-X/b4_reaudit/`; nothing was committed (`may_commit: false`). Disposable clones under `C:\ia\` were created for experiments and are not evidence.
