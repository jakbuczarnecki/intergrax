# EVOL-X B-4 — corrective implementation and independent re-audit handoff

**Ticket:** [`T-IGX-017-A`](../../../../../governance/tickets/T-IGX-017-A.md) · **Role:** implementer (`may_commit: true`) · **Policy:** `IIAC-2026-11-05.1` (`IN_FORCE` @ `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`)
**Start HEAD:** `ca2b147825ce6117cc7219e63a93d2872669cb49` · **Date:** 2026-11-05
**`B4_CORRECTIVE_IMPLEMENTATION_SHA`:** recorded in the binding commit that follows the implementation commit (see §9).

**Status:** implementation complete, **ready for independent re-audit**. This is not a self-PASS. This session issued no B6 acceptance, promoted no FRZ row, decided no operator item and did not close EVOL-X. Session C audit evidence and the P2 certification evidence are byte-unchanged.

## 1. Defect (B-4)

`scripts/qualification/evol_x_closure_publication_preflight.py` at the start HEAD:

- **False rejection.** It required superseded legacy gates (named OD-EVOL-11 human auditor, a manifest glob under `evol_x/`), and it required `B6_ACCEPTANCE_PUBLICATION_SHA` in the same check that should decide whether B6 may be accepted. That is circular: the SHA cannot exist before B6 is published.
- **False acceptance.** It accepted IIAC evidence by filename glob (`*EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md`, `*iiac*manifest*.json`) and trusted `status == IN_FORCE` text. It did not validate the manifest, digests, role separation, SHA ancestry, scope or findings. A forged or empty file with the right name would have satisfied the IIAC branch.

## 2. Root-cause matrix (false rejection ↔ false acceptance)

| # | Case | Old behavior | False rejection | False acceptance | Corrected rule (code) |
|---|---|---|---|---|---|
| 1 | OD-EVOL-11 external human identity | Fails while register says `auditor identity NOT NAMED` | **Yes** — IIAC migrates it to `auditor_session_id` + handoff | — | Not required in IIAC mode; independence enforced by A-09/A-16. Legacy only (L-05) |
| 2 | OD-EVOL-12 / OD-EVOLX-08 applicability | Required in the same pass as B6 readiness | **Yes** — they gate FRZ-08/10 text and parent closure, not B6 | — | Gate B only (B-07); certifier blocker `B-5` deferred to Gate B (A-18 vs B-09) |
| 3 | PASS record existence vs content | Glob match on filename | — | **Yes** — empty/forged file passes | Canonical path only; non-empty; byte-identical to auditor copy; digest = certification record = audit index; states stage, policy, verdict, SHAs (A-14) |
| 4 | Manifest schema and SHA integrity | Glob match `*iiac*manifest*.json` in `evol_x/` (wrong dir → also false rejection) | **Yes** — real manifest lives under `iiac_evidence/` | **Yes** — any JSON name passes | Canonical manifest; schema + `validate_manifest_dict`; companions `verify_iiac_audit_manifest` and `verify_iiac_role_separation --strict-product-diff` (A-03/A-04); every SHA is a commit with ancestry chain TES→FINAL→P1→P2→HEAD, non-self-referential (A-10) |
| 5 | `PASS_SCOPED` and its limitations | Not inspected | — | **Yes** — silent `PASS_SCOPED`→`PASS` or dropped limitation undetected | Verdict must equal the byte-preserved auditor draft and the pass record; every auditor limitation preserved as prefix; Session C restrictions pinned (A-06/A-07/A-08) |
| 6 | Auditor / implementer / certifier separation | Not inspected | — | **Yes** — self-certified manifest passes | Three distinct sessions and agent ids; canonical role ids; operator-attested handoff with signature line and no open checkbox (A-09/A-16) |
| 7 | B6 publication SHA presence and ancestry | Required always; presence only | **Yes** (circular, see 8) | **Yes** — any non-empty value, incl. historical internal SHA | Gate B: governance + ladder + certification record agree; strict descendant of `certification_publication_sha`; strict ancestor of HEAD; not a legacy row; phase-1 blob publishes `formal_b6_status: ACCEPTED` with SHA empty (B-01..B-04) |
| 8 | Circular B6 readiness ↔ closure | One gate | **Yes** | — | Two gates. Gate A never needs B6 SHA; Gate B never needs `FORMAL_CLOSURE_PUBLICATION_SHA` but validates it if present (B-08) |
| 9 | FRZ-EVOL-01..10 criterion-level acceptance | Not inspected | — | **Yes** — mass promotion or unscoped PASS undetected | Exactly one row per id; canonical requirement literal; PASS only with an `ELIGIBLE_PASS*` disposition; SCOPED stays SCOPED; LIM ids and mandatory scope text travel with the row (B-06) |
| 10 | Independent certification publication integrity | Not inspected | — | **Yes** | Audit index: 51 files digest/size, no unindexed file, bytes = P1 blobs; certification index: bytes = HEAD blobs; record ↔ manifest SHAs/digests/ids (A-12/A-13/A-17) |
| 11 | OD-EVOL-12 / OD-EVOLX-08 operator authority | Substring `**PENDING**` only | — | **Yes** — any edited status text passes | Register row must be SIGNED/APPROVED/CONFIRMED with ISO date; also OD-EVOL-05b/10/13 cited by FRZ dispositions (B-07). No decision is inferred from IIAC-OD-01 |
| 12 | Historical legacy records, non-retroactive evidence | Legacy B6 record accepted when IIAC inactive; no guard in IIAC mode | — | **Yes** — internal B6 / Protocol 03 automated record could stand in | Canonical IIAC paths only (A-14); legacy SHAs rejected as B6 (B-03); policy claiming `IN_FORCE` but failing the activation gate → `INVALID`, no legacy fallback (A-01); `NOT_IN_FORCE` → legacy path, IIAC evidence is not authority (L-00) |
| + | Policy genuinely `IN_FORCE` | Read `status` field | — | **Yes** — edited JSON passes | Canonical `evaluate_activation(..., require_in_force=True)` (A-01) |
| + | Stale audit | Not inspected | — | **Yes** | Ladder TES/FINAL = audited SHAs; no change under audited EVOL-X scope after its audit SHA (A-11) |
| + | Unresolved findings | Not inspected | — | **Yes** | Canonical `_findings_errors` + class `PROVEN_ARCHITECTURAL_DEFECT`/`IN_SCOPE_BLOCKER`, severity `BLOCKER`/`CRITICAL`/`HIGH` unresolved → fail; summary counts = 0 (A-15) |
| + | Superseded ECAP-P4 | Checked approvals only if `IN_FORCE` | — | Dual activation not detected | ECAP-P4 `IN_FORCE` alongside IIAC → fail (A-02) |

F-C-09 (`MISSING_EXECUTION_EVIDENCE`, procedural) is not treated as a blocking class: its closure criterion (operator-attested handoff, recorded ids) is enforced directly by A-09/A-16.

## 3. Two-gate contract

| | Gate A — B6 acceptance readiness | Gate B — formal EVOL-X closure readiness |
|---|---|---|
| Requires | A-01..A-18 (policy, manifest, scope, roles, SHAs, digests, indexes, pass record, findings, handoff, certification record, no B6-relevant open blocker) | Gate A **and** B-01..B-10 (published + bound B6, FRZ rows, operator decisions, valid ladder, no open blocker, no premature closure claim) |
| Does not require | `B6_ACCEPTANCE_PUBLICATION_SHA`, FRZ promotion, OD-EVOL-12/OD-EVOLX-08 | `FORMAL_CLOSURE_PUBLICATION_SHA` (validated if present) |
| CLI | `--gate b6-readiness` | `--gate formal-closure` (default) |

Gate A PASS never implies Gate B PASS (`PreflightResult.gate_b_pass` requires both). Gate B governance fields introduced by this contract: `formal_b6_status: "ACCEPTED"`, `b6_acceptance_policy_version`, `b6_acceptance_publication_sha` (two-phase), plus certification record `b6_acceptance: "ISSUED"` and `b6_acceptance_publication_sha`.

## 4. Session C scope restrictions retained (rejected if widened)

| Restriction | Enforcement |
|---|---|
| No complete platform-wide test ownership from F-C-03 | LIM-03 must keep `direct-import only`; FRZ-01/06 dispositions and rows must cite LIM-03 |
| FRZ-EVOL-07 qualification-layer only | LIM-05 `qualification-layer only`; disposition `LIM-05` + `NOT claimed`; row `qualification-layer` + `not claimed` |
| No production rollback enforcement claim | LIM-05 must keep `rollback is not gated` |
| M-09H `NOT_MEASURED` | LIM-02 must keep `M-09H NOT_MEASURED`; FRZ-08 disposition/row `M-09H not measured` |
| M-03 limited to self-surface arcs | LIM-04 must keep `self-surface arcs` |
| No `PASS_SCOPED` → `PASS` | manifest verdict = auditor draft = pass record; certification verdict keeps `SCOPED`; scoped FRZ rows stay `SCOPED` |

## 5. Ruff gate / evidence immutability

Two gates lint the 7 byte-preserved auditor probe scripts (`audit/evidence/probes/*.py`, 89 Ruff findings, 7 files not Ruff-formatted): the pre-commit gate (`run_pre_commit_gate.py`), skipped once for P1 `4dee0050d…` with `INTERGRAX_SKIP_PRE_COMMIT_GATE=1` under operator authorization, and the pre-push step `run_ruff_changed_paths.py --mode push`, which rejects the same paths whenever they are in the pushed range. **Root cause:** both gates pass explicit file paths to Ruff, and Ruff ignores `exclude` for explicit paths unless `force-exclude` is set. `force-exclude` would also re-apply Ruff's default excludes (`build`, `dist`, …) to tracked sources (e.g. `platform_proofs/**/scripts/build/*.py`), so it was rejected.

**Fix:** `[tool.ruff] extend-exclude = ["docs/project/maintainers/qualification/iiac_evidence"]` (SSOT) and `push_gate_config.split_ruff_exempt()` used by both gates. Only plain path entries are honored; glob entries are ignored (gates can only be stricter). No auditor file changed; tests pin the exemption to the evidence root and verify all 51 audit digests and the P1 blobs.

## 6. Changed files

| File | Change |
|---|---|
| `scripts/qualification/evol_x_closure_publication_preflight.py` | Rewritten: two-gate IIAC contract, legacy path kept for `NOT_IN_FORCE` only |
| `scripts/ci/push_gate_config.py` | `load_ruff_exempt_prefixes`, `ruff_exempt`, `split_ruff_exempt` |
| `scripts/ci/run_pre_commit_gate.py`, `scripts/ci/run_ruff_changed_paths.py` | Filter explicit Ruff paths through the exemption |
| `pyproject.toml` | `[tool.ruff] extend-exclude` (evidence root only) |
| `tests/unit/qualification/test_evol_x_closure_preflight_b4.py` | 51 tests: isolated git fixtures, positives + adversarial negatives + canonical check |
| `tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py` | 8 tests: exemption scope, gate wiring, evidence byte preservation |
| `governance/tickets/T-IGX-017-A.md`, this file | Ticket and handoff |

No product path (`intergrax/`, `agents/`, `applications/`), no IIAC material governance path (activation gate, `iiac_common`, manifest/role validators, protocol) and no audited EVOL-X scope path was changed, so the IIAC activation gate and the Session C audit stay valid (A-11 stays clean).

## 7. Evidence

Fixtures are temporary git repositories with the canonical relative layout and a full synthetic IIAC activation chain; nothing is written into the canonical repository.

| Area | Tests (all fail closed) |
|---|---|
| Positive | B6 readiness without closure (A PASS, B FAIL); complete formal closure fixture (A PASS, B PASS); CLI exit codes + JSON |
| Policy | `IN_FORCE` claim failing the activation gate; `NOT_IN_FORCE` legacy path; ECAP-P4 dual activation; evidence-root override on canonical repo |
| Pass record / manifest | missing, empty, forged pass record; missing, malformed, schema-invalid manifest; wrong stage, policy; non-commit SHA; broken ancestry; wrong digest (re-indexed); tampered dossier; uncommitted edit; unindexed file |
| Roles / staleness | self-certified session; self-certified agent; post-audit scope change; ladder TES moved; unresolved blocking finding; unsigned handoff; open B-4 blocker |
| Scope widening | `PASS_SCOPED`→`PASS`; LIM-02/03/04/05 rewritten; LIM-05 dropped; FRZ-07/08/01 disposition widened |
| Legacy | internal B6 record as pass record; historical SHA as B6 |
| Gate B | missing B6 SHA; self-referential B6; OD-EVOL-12/OD-EVOLX-08 pending; OD-EVOL-11 not required; FRZ-07 unscoped, FRZ-08 without M-09H, FRZ-06 without LIM-03, FRZ-03 OPEN; criterion literal rewritten; open B-5 blocker |
| Canonical | real repository: mode `IIAC`, Gate A errors ⊆ {A-18} while blockers are open, Gate B FAIL, all FRZ-EVOL rows `OPEN` |

Real-repository preflight after the change (exit 1, expected):

```text
Gate A - B6 acceptance readiness: FAIL
  - A-18 open certification blocker for B6 acceptance: B-4 legacy closure preflight requires independent corrective implementation and re-audit
  - A-18 open certification blocker for B6 acceptance: B6 acceptance (P3), FRZ promotion (P4) and formal closure (P5) not authorized in this publication
Gate B - formal EVOL-X closure readiness: FAIL  (B-00, B-01, B-02, B-06 ×10, B-07 OD-EVOL-12/OD-EVOLX-08, B-09)
```

All evidence checks (A-01..A-17) pass on the published Session C / P2 evidence. Gate A is held only by the certifier's own open-blocker list.

### 7.1 Consolidated qualification (worktree = start HEAD + this batch)

| Run | Result |
|---|---|
| `pytest tests/qualification/evol_x tests/unit/qualification tests/unit/governance` | **505 passed, 4 failed, 1 skipped** (21m39s). Includes the 290-node EVOL-X suite, the 59 new B-4 tests, IIAC activation / G0 / negative guards, FRZ canonical matrix, B6 attestation/custody tests |
| The 4 failures | `test_iiac_activation_gate::test_act_n01_inactive_policy_fails_default_and_require`, `::test_positive_fixture_passes_without_activating_canonical_policy`, `test_iiac_negative_guards::test_n09_implementer_must_not_publish_pass_filename_glob`, `::test_n12_session_a_policy_not_in_force_by_default`. **Pre-existing:** reproduced unchanged in a clean worktree at START_HEAD `ca2b1478…` (4 failed). They assert the pre-activation state (policy `NOT_IN_FORCE`, no IIAC PASS record files), which IIAC-OD-01 activation and P1 publication legitimately changed. Not modified here: they are IIAC negative guards and belong to a separate governance test-hotfix ticket |
| `verify_iiac_policy_activation --require-in-force` | exit 0 — `IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8…` |
| `verify_iiac_audit_manifest` (EVOL-X manifest) | PASS |
| `verify_iiac_role_separation --strict-product-diff` | PASS |
| `verify_iiac_g0_inventory_completeness` | PASS (17510 paths) |
| Pre-commit gate (`run_pre_commit_gate.py`, staged batch) | exit 0 — Ruff 6 files, diff-scoped gates OK, **no skip variable** |
| Repository state | IIAC `IN_FORCE`; certification record `stage_closure_status: NOT_CLOSED`, `b6_acceptance: NOT_ISSUED`; governance `b6_acceptance_publication_sha: null`, `evol_x_closed: false`; ladder `B6_ACCEPTANCE_PUBLICATION_SHA` PENDING; FRZ-EVOL-01..10 `OPEN`; OD-EVOL-12 / OD-EVOLX-08 `PENDING` |

## 8. Impact, residual risks, re-audit focus

- **Impact:** qualification tooling only. Changes the meaning of preflight PASS from "files present" to "evidence validated, per gate". No runtime behavior changes.
- **R-1 Text-bound restrictions.** Scope restrictions are checked by required phrases in limitations, dispositions and checklist rows. A reworded but equivalent limitation is rejected (fail-closed); a widening phrased while keeping the required words is not detected semantically.
- **R-2 Operator decisions** are verified as register state (status + date). Identity of the signer cannot be verified (`process_enforced`, LIM-10).
- **R-3 New Gate B fields** (`formal_b6_status`, `b6_acceptance_policy_version`, record `b6_acceptance`) are introduced by this contract; the certifier's P3 publication must follow it (two-phase).
- **R-4 Blocker parsing** relies on `B-<n>` prefixes in `open_blockers_for_closure`. Unprefixed entries block both gates.
- **R-5 Private-helper reuse.** The preflight reuses `_companion_errors`, `_findings_errors`, `_path_rejected` from `iiac_activation_gate` to avoid divergent logic; a rename there breaks the import (fail-closed at import time).
- **R-6 Audited scope list** is a constant mirroring `audit_scope.json` `in_scope`; a new audited path would need a code change.
- **Re-audit focus:** two-gate separation, absence of any silent pass path, fidelity of A-07/A-08 to Session C limits, B-03/B-04 two-phase semantics, and the Ruff exemption boundary.

## 9. SHAs

| Role | SHA |
|---|---|
| START_HEAD | `ca2b147825ce6117cc7219e63a93d2872669cb49` |
| `B4_CORRECTIVE_IMPLEMENTATION_SHA` | bound in the follow-up binding commit (this file, §9) |
