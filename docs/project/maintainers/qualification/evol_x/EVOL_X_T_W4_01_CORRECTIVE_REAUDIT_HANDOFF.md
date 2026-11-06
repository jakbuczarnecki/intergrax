# EVOL-X T-W4-01 — consolidated corrective (Part 1): independent delta re-audit handoff

**Ticket:** [`T-IGX-019-A`](../../../../../governance/tickets/T-IGX-019-A.md) (`Approved:` line left for the operator) · **Policy:** `IIAC-2026-11-05.1` (`IN_FORCE`)
**Role:** implementer (`may_commit: true`; cannot certify its own correction) · **Audit baseline:** `5afaad9eadfc1655f4ef1ab59eb022960a924648`
**Audited predecessor:** `6cf8c5a2247465a1321c239d0559d751383daec2` (RA-B4 corrective), `1e28b9e2b77bbc44171f80017fd56b4032aa196e` (T-IGX-018-A)
**`T_W4_01_PART1_IMPLEMENTATION_SHA`:** `742b47cfc096fc4878e49dce50313302169640ad` · evidence-only republication commit: `e0eeca8de` (parent of the implementation commit)
**EVOL-X status:** `NOT_CLOSED`. No B6 acceptance, no FRZ promotion, no OD-EVOL-12 / OD-EVOLX-08 / P3 decision, no change to `open_blockers_for_closure`, no closure.

This handoff is evidence for an **independent delta re-audit** by a *new* `independent-quality-auditor` session. It is not a PASS record,
does not resolve B-4 and does not authorise anything. Not done: **W4-F01** (needs an Operator decision), **W4-F08** (operator attestations).

## 1. Findings → changes

Changed code: [`scripts/qualification/evol_x_closure_publication_preflight.py`](../../../../../scripts/qualification/evol_x_closure_publication_preflight.py).
Changed tests: [`tests/unit/qualification/test_evol_x_closure_preflight_b4.py`](../../../../../tests/unit/qualification/test_evol_x_closure_preflight_b4.py)
(new `test_w4_*` section; existing fixtures adapted, see §3), [`tests/unit/governance/test_evol_x_real_repo_fresh_clone.py`](../../../../../tests/unit/governance/test_evol_x_real_repo_fresh_clone.py) (new).
Config: [`.gitattributes`](../../../../../.gitattributes) (three `-text` paths). Evidence-only: `IIAC_EVOLX_B4_EVIDENCE_PUBLICATION_BINDING.utf8.json` (new file).
Untouched (verify with `git diff 5afaad9ea..HEAD --stat`): `intergrax/`, `agents/`, `applications/`, `scripts/governance/*`, the activation record, the protocol, all existing evidence.

### RA-B4-05 — exact certified scope (`A-19` Gate A, `B-11` Gate B)

Baseline = the manifest and certification-record **blobs at `certification_publication_sha`** (P2). Nothing is decided by phrase search any more.

| Control | Rule |
|---|---|
| Baseline pin | `certification_publication_sha` must be the **oldest commit that added the certification record** (`git log --diff-filter=A`). Without this pin the baseline could be moved to a later, already-widened commit (also bypassed the RA-B4-02 blocker baseline — see §4, new finding **W4-X1**). |
| Limitations | multiset of `(id, exact text)` of the manifest `limitations` must equal the baseline multiset (order irrelevant); every dropped / added / reworded entry is an `A-19` line (id + 12-hex text digest); duplicate ids fail; entries without an id (e.g. *Evidence-only publication…*) are compared by exact text too. |
| `certification_verdict`, `frz_evol_dispositions` | exact equality with the baseline (case-sensitive; key set included). |
| Keys | certification record: any key absent from the baseline that is not a lifecycle key (`CERT_LIFECYCLE_KEYS`) fails; a `*scope*`/`*claim*`-named baseline key must keep its value. Manifest: no new keys; scope/claim-named keys must keep their value. |
| Promoted FRZ rows (`B-11`) | status + notes cells must equal `render_promoted_row(<baseline disposition>)` (`ELIGIBLE_PASS` → `PASS`/`—`; `ELIGIBLE_PASS_SCOPED(x)` → `PASS — SCOPED`/`x`; `ELIGIBLE_PASS_EVOL_X_SCOPED(x)` → `PASS (EVOL-X SCOPED)`/`x`), exactly 6 cells. Old `B-06` checks are kept. |
| B6 record (`B-11`) | only: the exact title `# EVOL-X IIAC B6 acceptance record`, allow-listed `Key: value` lines (`B6_RECORD_FIELDS`), blank lines and the signature line; fenced blocks, prose, extra fields fail; `Decision scope` must be exactly `B6 acceptance for EVOL-X (<verdict>, evidence-bound)`; `Decision ID` a bare identifier; `Accepted limitations` = baseline `<id>:<sha256 of exact text>` set (`accepted_limitation_tokens()`), stated once, no duplicates. |

There is **no amendment channel**: the specification allowed a signed `scope_amendments` record that narrows scope; that was not implemented (it would
create an authority path with no operator-record contract). Any deviation from the baseline needs a new P2 certification publication.

### W4-F03 / W4-F05 — blocker lifecycle

`blocker_ledger()` walks every published version of the certification record after the baseline (`git log --reverse <cert_sha>..HEAD -- <record>`);
each blocker ever listed gets an origin `(opened_at_sha, text)`. It must stay open or carry an evidence-bound resolution whose
`opened_at_sha` / `baseline_text_sha256` are those of its first publication; closure must follow that publication. Duplicate open ids are an explicit
`… more than once (duplicate id)` error (Gate A `A-18` and Gate B `B-09`), no longer merged by `setdefault`.

### W4-F04 — operator records

`operator_record_errors(..., not_before=)`: P3 (via `resolution_errors`) and OD-EVOL-12 / OD-EVOLX-08 records are rejected when the signature is older than the
certification publication commit; every record's `Decision scope` is rejected when it contains a negation (`scope_negation()`: `NOT`, `NO`, `NEVER`, `DENIED`,
`UNAUTHORIZED/UNAUTHORISED`, `REVOKED`, `REJECTED`, `DECLINED`, `WITHDRAWN`, `PROHIBITED`, `FORBIDDEN`, contractions; NFKC + zero-width stripped). The provenance limit
(signature line is process-enforced text) is unchanged and still documented in the module docstring.

### W4-F06 — byte-stable binding records

`.gitattributes`: `-text` for the binding, its UTF-8 republication and the delegation record, so `delegation_record_sha256 (2120852e…)` equals the checkout bytes everywhere.
The originals were **not edited** (`test_w4_f06_the_original_binding_and_delegation_record_are_never_edited` compares them with the audit baseline blobs).
`IIAC_EVOLX_B4_EVIDENCE_PUBLICATION_BINDING.utf8.json` is a separate evidence-only commit: identical field values, the single `0x97` decoded as `U+2014`, plus
`supersedes{path, sha256 of the original blob, transcoding}`. Working copies cloned before this change need `git checkout -- <path>` once (Git now treats the CRLF copy as modified).

### W4-F07 — real-repository fresh clones

`tests/unit/governance/test_evol_x_real_repo_fresh_clone.py` clones the committed `HEAD` of the real repository with `core.autocrlf=true` and `false` and runs the real
`verify_iiac_policy_activation --require-in-force` and the real preflight; it also replays the auditor's real-evidence tamper and widening probes inside the clone.
The two `autocrlf=false` checks are **strict `xfail`** (`W4_F01`): they document the open defect and turn red (XPASS) as soon as W4-F01 is repaired, forcing the marker's removal;
`test_w4_f01_is_the_only_reason_the_lf_clone_activation_fails` pins that `ACT-N08` is the sole LF failure.

## 2. Audit scenarios → controls

| Auditor probe | Now |
|---|---|
| S05-01..06 (LIM-01..05 appended / negated) | `A-19` limitation … dropped or reworded / added (scope widening) |
| S05-07 duplicate widened LIM-03 | `A-19` duplicate id + added entry |
| S05-08 reorder-only control | passes (3 permutations tested) |
| S05-09 / S05-10 FRZ disposition / verdict widened | `A-19` disposition / `certification_verdict` differs |
| S05-11 scope key added | `A-19` key absent from the certification publication |
| S05-12 certifier-authored limitations dropped | `A-19` limitation … dropped |
| S05-13 promoted row widened | `B-11` row must equal the rendered row |
| S05-14 B6 record `Accepted scope:` field | `B-11` unexpected field / not allow-listed |
| S02-01 late blocker added then deleted | `A-18` blocker B-7 added after the certification publication … disappeared without closure evidence |
| S02-02 / S02-02b / S02-02c duplicate open ids | `A-18` / `B-09` … (duplicate id) |
| S02-06 / S02-07 P3 pre-dated / negated scope | `A-18` … predates the certification publication / … negation |
| S03-01 / S03-02 OD scope negated / signed 2020 | `B-07` … negation / … predates the certification publication |
| S04-04 B6 `Accepted limitations` omits entries | `B-11` |
| S04-05 format-valid agent-authored signature | **unchanged** (documented provenance limit, `process_enforced`) |

## 3. Existing fixtures that had to change (so the re-auditor can check the weakening direction is nil)

* `DISPOSITIONS["FRZ-EVOL-07"]` now contains `qualification-layer deploy/rollback safety` (as the real disposition does); `PROMOTED_ROWS` are the exact rendered rows.
* The B6 fixture record uses the canonical title, the canonical scope string and `<id>:<sha256>` tokens (`_accepted_limitation_tokens`, an independent re-statement of the contract).
* B6 tests for `Accepted limitations` read `B-05` **or** `B-11`; two P3 and two OD record cases were added to their existing parametrisations.
* `_build_base(resolve=True)` and `_publish_b6(record_transform=)` gained keyword options; no existing call changed meaning.

## 4. Findings of this implementation for the re-auditor

* **W4-X1 (new, fixed here):** `certification_publication_sha` was not pinned to the real P2. Re-pointing it (manifest + record) at a later commit moved *both* the scope baseline and the RA-B4-02 blocker baseline. Tests: `test_w4_ra_b4_05_scope_baseline_cannot_be_moved…`, `test_w4_ra_b4_02_blocker_baseline_cannot_be_moved…`.
* **Residual:** `git log --diff-filter=A` trusts repository history; a force-rewritten history is outside the model (same as the rest of the preflight). The B6 title/scope are canonical strings chosen by this implementation (no operator template existed): the re-auditor should confirm they are acceptable as the contract.
* **Residual:** `render_promoted_row` formats were derived from the fixtures / the auditor's pass-record style (`PASS — SCOPED`); the operator will have to author promoted rows from that function.

## 5. Verification (Windows 11, `core.autocrlf=true`, Python 3.12.11, short `--basetemp D:/w4pt/…`)

| Check | Result |
|---|---|
| `test_evol_x_closure_preflight_b4.py`, pre-existing + adapted tests (`-k "not w4_"`) | 210 passed (baseline before the change: 209 passed) |
| same module, new `test_w4_*` tests (135, incl. the two baseline-move tests, W4-F03/F04/F06) | 133 passed in the full run; the 2 baseline-move tests + 26 W4-F04/F06/valid-B6 tests re-run after the final `ruff format`: 28 passed |
| `tests/unit/governance` + `test_evol_x_b4_ruff_evidence_exemption.py` | 143 passed, 1 skipped (symlinks), 2 xfailed (strict W4-F01 markers) |
| `tests/unit/governance/test_evol_x_real_repo_fresh_clone.py` at the implementation commit | 5 passed, 2 xfailed (`autocrlf=false` activation / preflight = W4-F01) |
| `ruff check` / `ruff format --check` on the three changed Python files | clean |
| Canonical preflight (`--gate b6-readiness`) | only the two real `A-18` open-blocker errors (B-4, B6-P3); no `A-19` |
| Auditor fixture probes (`auditor_probes_fixture.py`, 2 runs, ~60 scenarios) | all `CONTROL_OK` / `POSITIVE_OK` (S05-08 reorder control accepted) except **S04-05** (documented provenance limit, expected) and **S05-13** (harness error: the probe's text replacement no longer matches because the fixture row wording changed; the same scenario is covered by `test_w4_ra_b4_05_promoted_row_must_equal_the_template_rendered_from_the_baseline[FRZ-EVOL-07-…]`). One run crashed on a cp1250 `print` of a Cyrillic probe label (harness only) and was re-run with `PYTHONUTF8=1`. |
| Real-evidence probes (`real_scope_probe.py` with `SHA` = implementation SHA, clone of this commit) | R05-1/-2 (LIM-03, LIM-05 widened), R05-3 (certifier limitations dropped), R05-3b (FRZ-EVOL-07), R05-4 (verdict) each add `A-19`; R02-1 (all blockers deleted) and R02-2 (historical evidence) rejected |
| `real_scope_probe2.py`, `real_crlf_matrix.sh`, GitHub Actions run on the pushed SHA | **not run** (CI needs a push; the matrix script is bound to a bundle path of the auditor) — to be re-run by the re-auditor |
| Not verified | Linux behaviour (no CI run); W4-F01 LF activation still fails by design until Part 2 |

## 6. Re-audit instructions

1. `git diff 5afaad9ea..HEAD --stat`; confirm the forbidden paths are untouched and `IIAC_EVOLX_B4_EVIDENCE_PUBLICATION_BINDING.json` / the delegation record are byte-identical to `5afaad9ea`.
2. Re-run the auditor probes against this tree: `evidence/auditor_probes_fixture.py <worktree> <out.json>`, `real_scope_probe*.py` (set `SHA` to the implementation SHA), `real_crlf_matrix.sh`; every `FALSE_PASS` of the T-W4-01 register except S04-05 and the W4-F01 cases must now be a rejection.
3. Challenge §1 table rows and §4 residuals (try to widen scope through a path not listed; try a legitimate post-baseline operation that the new equality wrongly rejects).
4. W4-F01 must be handled by Part 2 after the Operator decision; Linux CI (`governance-fast`) stays red on the two ACT-N08 tests until then.
5. Acceptance needs a **separate** `certification-governance-agent` session; this implementer does not issue it.
