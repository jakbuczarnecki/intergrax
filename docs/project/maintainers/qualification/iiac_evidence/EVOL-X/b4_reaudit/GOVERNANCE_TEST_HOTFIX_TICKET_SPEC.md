# Governance test-hotfix ticket specification (proposed — not created by the auditor)

**Proposed id:** `T-IGX-018-A` (number to be assigned by the operator; the auditor role has `may_commit: false` and does not create tickets).
**Title:** Governance tests: retire stale pre-activation assertions, stop canonical-file pollution, fix CI history depth.
**Origin:** EVOL-X B-4 independent re-audit, findings HF-01, HF-02, HF-03. **Not part of B-4**; none of these defects was introduced by `23feb2bbb`.

## Evidence

| ID | Test / config | Root cause | Class |
|---|---|---|---|
| HF-01a | `test_iiac_activation_gate.py::test_act_n01_inactive_policy_fails_default_and_require` | Runs the default verifier against the canonical record and expects exit != 0; the record is now `IN_FORCE` and valid (exit 0). | stale test assumption |
| HF-01b | `test_iiac_activation_gate.py::test_positive_fixture_passes_without_activating_canonical_policy` | Asserts canonical `status == "NOT_IN_FORCE"`. | stale test assumption |
| HF-01c | `test_iiac_negative_guards.py::test_n12_session_a_policy_not_in_force_by_default` | Same assertion on the canonical record. | stale test assumption |
| HF-01d | `test_iiac_negative_guards.py::test_n09_implementer_must_not_publish_pass_filename_glob` | Globs `*IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` under the qualification root and expects none; the Session C pass records are now legitimately published (`evol_x/` and `iiac_evidence/EVOL-X/audit/`). A filename guard cannot distinguish a forged record from a genuine one. | stale test assumption; guard obsolete |
| HF-02 | `test_iiac_negative_guards.py::test_n05_*` (lines 99-108) and `::test_n08_*` (lines 126-142) | Overwrite `OD_ENT_02_THREAT_MODEL.md` / `IIAC_POLICY_ACTIVATION_RECORD.json` in the canonical tree and restore with `write_text`, which converts LF to CRLF on Windows. The activation record is `-text`, so the restored file no longer equals its Git blob and the activation gate (and the EVOL-X preflight, A-01) fails until `git checkout`. | test pollution / filesystem side effect |
| HF-03 | `.github/workflows/unit-tests.yml` job `governance-fast` | Depth-1 checkout. `test_iiac_g0_inventory.py` (2 tests) and the activation gate need history (`not ancestor of HEAD`, `bad object`). | CI configuration |

CI run `42104326792` (HEAD `9a8fa3634`) governance-fast: 6 failed, 120 passed = the four above + the two G0 tests.

## Required changes

1. HF-01a-c: change the expectations to the post-activation contract. Use a fixture copy for the negative case (e.g. run the verifier with `--activation-record` pointing at a tmp copy with `status: NOT_IN_FORCE`, and `--expect-not-in-force` against that copy), and assert the canonical record verifies `--require-in-force` instead of asserting `NOT_IN_FORCE`.
2. HF-01d: replace the filename glob with a content guard: any `*IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` must be referenced by a manifest that passes `verify_iiac_audit_manifest` and `verify_iiac_role_separation`; unreferenced or invalid pass records fail the guard.
3. HF-02: never write to canonical paths. Copy the file to `tmp_path`, point the verifier at it, or restore with `write_bytes(original_bytes)`. Add a post-test assertion fixture (session scope) that `git hash-object` of the activation record equals the HEAD blob.
4. HF-03: `actions/checkout` with `fetch-depth: 0` for `governance-fast`.
5. Do not touch IIAC material governance paths (activation gate, `iiac_common`, manifest/role validators, protocol) — only tests and the workflow.

## Acceptance

- `pytest tests/unit/governance -q` reports 0 failed locally (Windows) and in CI; after the run `git status --short` is empty and the activation-record blob is unchanged.
- Negative controls kept: a forged `IN_FORCE` record, an unindexed pass record, a mismatched digest each still fail.
- Independent review by a session other than the implementer.

## Out of scope

EVOL-X preflight changes (see RA-B4-01..04), Session C and P2 evidence, product paths.
