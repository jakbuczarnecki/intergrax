# W2.1-A CI disposition — per test node

Published copy of local evidence; not an IIAC certification.

- Recorded: 2026-11-27, Claude Code master orchestrator. This is **not** an IIAC certification.
- Subject: development 7d2cc3d28, the Wave 2 integration merge on top of base 32005c212.
- Evidence: job logs re-downloaded as UTF-8 into `.tmp/session/ec-w21/ci/job_<id>.log`. Node lists are in `nodes_<job>.txt`.
- The Antigravity originals in `.tmp/session/ec-campaign-master/w21-ci/` are UTF-16. They are kept as they are.

## Runs

| Run | SHA | Job | Result |
|---|---|---|---|
| Regression gate 45951546692 | 7d2cc3d28 | governance-fast 140348318639 | 17 failed, 1012 passed |
| Regression gate 45951546692 | 7d2cc3d28 | evol-x-b4-preflight 140348318606 | 10 failed, 485 passed, 2 skipped |
| Regression gate 45860721660 (historical) | aabd3925c | governance-fast 140050719713 | 15 failed, 1014 passed |
| Regression gate 45860721660 (historical) | aabd3925c | evol-x-b4 140050720251 | 10 failed, 485 passed, 2 skipped |
| Regression gate (historical) | 646e63953 | governance-fast 140001853368 | 15 failed (same set) |
| Enterprise audit 45951546779 | 7d2cc3d28 | 140348317813 | 5 ENT-A gate failures |
| Enterprise audit 45656931539 (historical) | ec8d85021 | 139356007961 | same 5 ENT-A gate failures |

**Wave 2 delta.** `32005c212..7d2cc3d28` touches 67 files. No change under `intergrax/`, `agents/`, `applications/`, `scripts/` or `.github/`. The only capability change is +63 doc lines in 4 architecture docs. **No production code changed.**

**Downstream jobs.** In every run inspected, governance-fast and evol-x-b4 fail, so every downstream job is skipped: gate-tests, governance-tier, wiring, closeout, ci-enterprise and diagnostics. **The full integrated suite has not run in CI at any of these SHAs.** That is a qualification gap, not a pass.

## governance-fast (17 nodes)

Classes: EXP-RED = expected-red before RB-2; CASCADE-MASKED = failure caused by an upstream gate, so the node's own assertion is not evaluated; HIST = historical baseline from before Wave 2; XPROG = cross-program defect.

| # | Node | Message | Class | Evidence |
|---|---|---|---|---|
| 1 | `test_iiac_activation_gate.py::test_canonical_policy_is_in_force_and_verifies` | IIAC activation FAIL: ACT-N11 product path changed | EXP-RED | ACT-N11 is expected to fail before RB-2 and is not a waiver trigger (MEP-AF2_v2.1.md:881, :953). Same failure at 646e63953 and aabd3925c. |
| 2 | `test_iiac_negative_guards.py::test_n12_policy_is_in_force_only_with_valid_activation_evidence` | assert 1 == 0 | EXP-RED | Same root as #1 |
| 3–4 | `test_evol_x_real_repo_fresh_clone.py::test_da01_real_repoint_*` (2) | A-01 policy IN_FORCE but activation gate fails (ACT-N11) | EXP-RED, A-01 cascade | `policy_mode()` returns INVALID |
| 5–8 | `test_evol_x_real_repo_fresh_clone.py::test_da02_real_sealed_record_keys…[4 params]` | A-01 | EXP-RED, A-01 cascade | as above |
| 9–11 | `test_evol_x_real_repo_fresh_clone.py::test_real_activation_verifies_in_a_fresh_autocrlf_{false,input,true}_clone` | IIAC policy activation: FAIL | EXP-RED | ACT-N11 |
| 12–13 | `test_evol_x_real_repo_fresh_clone.py::test_real_preflight_in_a_fresh_autocrlf_{false…matches_crlf, true…only_certifier_blockers}` | A-01 | EXP-RED, A-01 cascade | as above |
| 14 | `test_evol_x_real_repo_fresh_clone.py::test_real_preflight_flags_tampered_evidence_in_a_fresh_clone` | `assert 'A-14' in {'A-01'}` | **CASCADE-MASKED** | `evol_x_closure_publication_preflight.py:2679-2680`: when the mode is INVALID, the preflight returns before `gate_a()`. A-14 is never evaluated. |
| 15 | `test_evol_x_real_repo_fresh_clone.py::test_real_preflight_rejects_a_widened_certified_scope_in_a_fresh_clone` | any(A-19/LIM-03/FRZ-EVOL-07) False | **CASCADE-MASKED** | same early return; A-19 is never evaluated |
| 16 | `test_pdc_x_inventory.py::test_committed_inventory_is_current_and_closed_world` | INTEGRAX_INTELLIGENCE.md is UNMAPPED | **XPROG** (INTELLIGENCE-X) | Introduced by 0792036d5 (T-IGX-079-A, 2026-11-26). That commit is an ancestor of the Wave 2 base 32005c212 but not of aabd3925c. The failure is older than Wave 2; it first appeared at 7d2cc3d28 because 32005c212 had no CI run. |
| 17 | `test_pdc_x_inventory.py::test_every_item_has_exactly_one_known_category_and_status` | as #16 | **XPROG** | as #16. Fix candidate: T-IGX-090-E, see W2.1-B. |

**Residual risk on #14 and #15.** These tests do not currently check tamper detection or scope widening. Their real result will only be known once ACT-N11 clears at RB-2. After RB-2 they must be re-run on their own; they must not be assumed green.

## evol-x-b4-preflight (10 nodes) — all HIST

The node set is identical at 646e63953, aabd3925c and 7d2cc3d28 (`diff` of the node lists is empty).

**Onset.**
- Last success: 4a1242ea4 (2026-11-25 05:31Z).
- First failure: dd4ffe4bc (05:46Z).
- The first-parent merge in that window that brought B4 code in is `695109d86` (Merge claude/elegant-cannon-a61eed). It carried `b3b9a205a` from 2026-11-06, "W4-01 round 3 — bind inputs to HEAD by content" (T-IGX-019-A), together with 9c94506f9 and bc0e6e228.
- The stricter gate merged on 11-25 met EVOL-X evidence that had drifted after 11-06. This is not ACT-N11 and not Wave 2.

| Node (`tests/unit/qualification/test_evol_x_closure_preflight_b4.py::`) | Cause |
|---|---|
| `test_ra_b4_03_real_register_rows_still_classify_correctly` | OD-EVOL-12 register row classification |
| `test_w402_02_real_clone_is_clean_and_every_imported_gate_module_is_byte_bound` | preflight FAIL caused by the A-18/A-19 rows below |
| `test_w402_02_untracked_unrelated_files_outside_evidence_and_modules_stay_clean` | same aggregate preflight FAIL |
| `test_w402_03_open_blocker_text_must_equal_the_published_text` | A-18: open blocker B-5 text changed after publication |
| `test_w402_03_a_resolved_blocker_entry_may_carry_only_allow_listed_keys[B-4]` | A-18: unexpected `scope_note` key |
| `…[B6-P3]` | A-18: unexpected `scope_note` key |
| `test_w402_03_nested_resolution_objects_are_allow_listed_too` | A-18 allow-list (same `scope_note` drift) |
| `test_w402_03_a_committed_file_in_an_unindexed_directory_is_rejected` | A-19: `iiac_evidence/EVOL-X/side_channel/` is not indexed |
| `test_w402_03_a_committed_addendum_at_the_evidence_root_is_rejected` | A-19: `iiac_evidence/EVOL-X/SCOPE_ADDENDUM.md` sits outside the known evidence directories |
| `test_w402_04_commit_graph_and_alternates_are_documented_as_residuals` | "alternates" residual missing from the preflight module docstring |

The disposition is HIST: an EVOL-X evidence or register drift against the T-IGX-019-A gate. Who owns the fix is undecided between EVOL-X and AF-CLOSE; it goes to the AF-CLOSE handoff. Wave 2 does not own it, and the gate must not be relaxed.

## Enterprise audit (ENT-AUDIT-X)

| Item | 7d2cc3d28 | ec8d85021 (2026-11-25) | Class |
|---|---|---|---|
| ENT-A-PRD-01 | ERROR FAIL | ERROR FAIL | HIST |
| ENT-A-PRD-02 | TRACKED_DEBT FAIL | TRACKED_DEBT FAIL | HIST, tracked debt |
| ENT-A-REG-03 | TRACKED_DEBT FAIL | TRACKED_DEBT FAIL | HIST, tracked debt |
| ENT-A-TEN-01 | TRACKED_DEBT FAIL | TRACKED_DEBT FAIL | HIST, tracked debt |
| ENT-A-TEN-02 | TRACKED_DEBT FAIL | TRACKED_DEBT FAIL | HIST, tracked debt. Not new: it already failed at ec8d85021. Its acceptance test predates Wave 2 (e0f64ed7f). |

## Totals

| Class | Count |
|---|---|
| New regression caused by Wave 2 | **0** |
| Expected-red, ACT-N11 before RB-2 | 13 |
| Cascade-masked, unverified until RB-2 | 2 |
| Cross-program defect (INTELLIGENCE-X / PDC-X), fix candidate ready | 2 |
| Historical EVOL-X B4 drift | 10 |
| Historical ENT-AUDIT | 5 |
| Blocker for integrated qualification | 1: downstream jobs never executed |

These statuses do not support a G2 / R / FCS / M3 claim. CI is red, and must not be described as green or waived.
