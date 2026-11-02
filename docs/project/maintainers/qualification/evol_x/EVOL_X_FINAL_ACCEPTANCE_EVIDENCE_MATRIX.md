# EVOL-X — final acceptance evidence matrix (security + gates)

**Authority:** [EVOL_X_COMPLETE_CLOSURE_AND_CERTIFICATION_MASTER_PLAN.md](./EVOL_X_COMPLETE_CLOSURE_AND_CERTIFICATION_MASTER_PLAN.md).  
**Contract tests:** `tests/unit/qualification/test_evol_x_b6_attestation_contract.py` (29 tests).

| Scenario | Test / control | Oracle | Evidence | Acceptance threshold |
|----------|----------------|--------|----------|----------------------|
| Signature forgery | `test_forged_signature_is_rejected` | verify → invalid | pytest log | exit 0 test PASS |
| Wrong trust root | `test_unknown_trust_root_is_rejected` | unknown kid | pytest | fail-closed |
| Revoked key | `test_revoked_signing_key_is_rejected` | revocation list | pytest | fail-closed |
| Implementer-controlled signing | `test_implementer_controlled_attestor_is_rejected` | attestor principal check | pytest | fail-closed |
| Unauthorized principal | `test_wrong_attestor_is_rejected` | attestor id mismatch | pytest | fail-closed |
| Compromised runner | Operational: runner not admin-controlled by implementer | infra inventory | [EVOL_X_AUTO_B6_INFRASTRUCTURE_INVENTORY.json](./b6_technical_qualification/EVOL_X_AUTO_B6_INFRASTRUCTURE_INVENTORY.json) | attestor principal isolated |
| Mutable verifier | Code review + signed security review | hash-pinned scripts | security sign-off record | reviewer signature |
| Workflow substitution | GHA branch protection + external workflow owner | audit log | security review | no implementer-only control |
| Policy downgrade | `test_in_force_flag_without_approvals_is_insufficient` | governance conjunction | pytest | NOT_ELIGIBLE |
| Evidence tampering | Mode A digest checks | `evol_x_b6_independent_verification.py` | Mode A log | INTEGRITY_OK |
| Wrong TES | semantic mutation tests | bound SHA mismatch | pytest parametrized | fail-closed |
| Wrong FINAL | `--evidence-ref` adversarial | orchestrator exit 2 | W7-01 report | fail-closed |
| Stale CI | CI record pin `41337077334` | ladder + attestation fields | CI qual record | exact run id |
| Replay attack | `test_replayed_attestation_is_rejected` | replay ledger | pytest | fail-closed |
| Incomplete ledger | replay ledger schema tests | append-only | contract tests | reject duplicate id |
| Missing security review | `evaluate_formal_eligibility` | blockers list | governance JSON | `MISSING_SECURITY_REVIEW` until signed |
| Missing maintainer approval | governance `NOT_IN_FORCE` | attestation verify | governance JSON | POLICY_NOT_IN_FORCE |
| Self-acceptance | `test_self_acceptance_is_rejected` | attestor ≠ implementer | pytest | fail-closed |
| Cross-stage reuse | attestation scope fields | EVOL-X only in document | schema | reject foreign stage |
| Unauthorized FRZ promotion | attestation must not claim FRZ PASS | schema + tests | pytest | fail-closed |
| False EVOL-X CLOSED | `test_published_records_keep_formal_b6_pending` | governance `evol_x_closed: false` | repo scan | no premature CLOSED |
| Fail-open errors | `test_unsigned_record_is_rejected` | verify | pytest | fail-closed |
| Test cred as formal B6 | `test_valid_test_credential_cannot_authorize_formal_b6` | credential class | pytest | NOT_ELIGIBLE |
| Production claim on test custody | `test_production_claim_on_test_custody_is_not_eligible` | custody domain | pytest | NOT_ELIGIBLE |
| Insufficient authority | `test_insufficient_authority_blocks_formal_eligibility` | blockers | pytest | NOT_ELIGIBLE |
| Altered Mode results | `test_altered_mode_results_are_rejected` | attestation binding | pytest | fail-closed |
| Stale validity window | `test_stale_window_when_clock_is_outside_validity` | notBefore/notAfter | pytest | fail-closed |
| Technical PASS ≠ acceptance | `test_technical_pass_does_not_publish_acceptance` | eligibility | pytest | NOT_ELIGIBLE |
| Mode A integrity | G-05 | orchestrator Mode A | provenance doc | INTEGRITY_OK |
| Mode B replay 290 | G-06 v2 | pytest @ TES | Mode B log | 290 PASS |
| VL3 population | G-01 | CI 41337077334 | CI record | 290/290 |
| M-09H scoped N/A | G-03 / F-008 | metrics | FINAL bundle | NOT_MEASURED allowed |
| FRZ-EVOL-07 qual scope | G-08 | S7 + assessor | FRZ dossier | Outcome A at W7-03 |
| 102-ID mechanical | G-14 | 102 matrix | b6_technical_qualification | auditor attestation @ B6 |

**Independent security review gate:** scenarios marked operational require [EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md](./EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md) — **not** satisfied by agent pytest alone.
