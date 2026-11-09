# P5 IIAC audit report — HIST product batch (pre-activation)

| Field | Value |
|---|---|
| Campaign | HIST-01..04 |
| Auditor session | `hist-p5-iiac-aud` (process_enforced; same worktree as merge orchestration, read-only verification) |
| Tree SHA | `3c91694c645ff0d97d104adf0c5a564b9a4c5b3c` (`hist-cert/p5-product-merge` = `origin/development` + merge `origin/hist-cert/product-batch`) |
| Product integration SHA | `427646f41de0df992cd54c0e157de8f36b560f39` |
| P4 product audit | `468b17576` @ `hist-cert/aud-product-report` |

## Scope

- Product-only delta C1–C4; no material-path edits; no T-IGX-021-B/C.
- Repository IIAC gate unit tests + canonical verifier.

## Commands and results

| Step | Exit | Notes |
|---|---|---|
| `pytest` `test_iiac_activation_gate.py` + `test_iiac_negative_guards.py` | **1** (2 fails, 39 pass, 1 skip) | Expected on unactivated product merge: `test_canonical_policy_is_in_force_and_verifies`, `test_n12_policy_is_in_force_only_with_valid_activation_evidence` fail because ACT-N11 lists 12 product paths |
| `verify_iiac_policy_activation --require-in-force` | **1** | Lists exactly the 12 C1–C4 paths (matches P4 AUDIT_REPORT expectation) |

Logs: `iiac_unit_tests.log`, `verify_require_in_force.log`, `summary.txt` in this directory.

## Verdict

**PASS_SCOPED (pre-activation audit package)** — evidence correctly shows ACT-N11 red until Operator publishes new activation. **Not** a substitute for P5-SIGN-ACT or P5-VERIFY green.

## Blocker

| ID | Owner | Action |
|---|---|---|
| P5-SIGN-ACT | Operator | Publish new IIAC activation record + literal `operator @` line for post-merge audit SHA |
| ENT consent | ENT-AUDIT-X owner | Optional counter-signature on shared re-activation file |
