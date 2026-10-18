# D03-A-PLATFORM-POLICY-CONFIG-01

| Field | Value |
|---|---|
| Status | **READY FOR AUDIT** |
| Date | 2026-10-18 |
| Baseline | `c41c788f15a480dc59ffcb68717044ce5a30adaf` |
| Parent | `PROD-Q-R1-R7-Q-D03-A-CONDITIONAL-INFRASTRUCTURE-DIRECTION` — remains OPEN / BLOCKED |
| Predecessor shape | Readiness §15. This record is the implementation evidence, not a rewrite of that document |
| LOCAL-01 | Unchanged behavior. `intergrax/runtime/governance/local_history_continuity.py` was not edited |
| Production admission delta | **0** |
| FRZ | **not promoted**. New `FRZ-*` PASS = **0** |

Cursor may report **READY FOR AUDIT** only. This record does not close D03-A, D03-B, D03-C, IMPL-01, R7-Q, PROD-Q, or any global `FRZ-*`. It is not cloud infrastructure proof, IAM proof, a production retention period, or a replaceability demonstration.

## What the commit adds

`HistoryContinuityActivation` now requires `policy: HistoryContinuityActivationPolicy`. The policy is frozen, `extra="forbid"`, and carries:

- `history_scope`: only `environment` may activate; `tenant` is rejected
- `administrators`: primary and witness identifiers, present, non-blank, unambiguous, and distinct under case-fold
- `runtime_credential_mode`: only `local_no_runtime_credential` may activate; static long-lived, administrator, and workload-federation modes are rejected
- `separated_capabilities`: exactly `WRITE`, `INITIALIZE`, `RECONCILE`, `INVALIDATE`, and `RECOVER`, each once

No secret, token, or connection field exists. Retention stays the existing `HistoryContinuityRetentionPolicy` on the activation. The writer still cannot lower the duration, lower the ceiling, or clear the hold. No production duration was chosen.

`compose_history_continuity_authority` remains the only composition. Missing activation, or activation that fails `history_continuity_activation_is_admitted`, returns `InactiveHistoryContinuityAuthority`. Malformed configuration that Pydantic rejects never reaches that function. The existing issuer/resolver handler was not widened. A principal tenant that does not match the environment resolver is still not activated. Environment scope does not authorize another tenant's initialize.

## What this commit does not do

- No change to `GovernedProviderInvocationDispatchGate`, production lifecycle wiring, `ProviderInvocationStore`, or GCA
- No change to local CAS, fencing, witness, recovery, or chain verification
- No cloud SDK, IAM federation, or credential acquisition
- No second composition function and no second adapter selector
- No claim that distinct configured identifiers are real administrative independence
- No claim that the ceiling is legal retention or WORM

## Dependency graph

```text
ProductionControlPlaneGovernance
    |
    v
compose_history_continuity_authority
    |
    v
HistoryContinuityActivation + HistoryContinuityActivationPolicy
    |
    +-- missing / rejected --> InactiveHistoryContinuityAuthority
    |
    +-- admitted + resolver match --> _open_local_history_continuity
                                          |
                                          v
                                   LocalDeterministicHistoryContinuity
                                          |
                                          v
                         ControlPlaneMutationAuthorizationBoundary
```

`ContinuityPermissionGrant` remains a caller claim. The boundary remains the authorizer.

## ECAP change delta

| Step | Result |
|---|---|
| A1 | Contract field and policy models; one composition predicate; targeted tests; this record and roadmap status. No new public execution symbol |
| A2 | Composition imports the contract predicate and still opens only the existing local adapter. No new configuration registry |
| A3 | CREATE gate and production lifecycle remain non-callers. Checked by reading those three files, not by a repository scan |
| A4 | Semantic owner unchanged: `ProductionControlPlaneGovernance`. Composition owner unchanged: `compose_history_continuity_authority` |
| A5 | Production-effective reachability: **no** |
| A6 | Families below. No `FRZ-*` status change |
| A7 | Historical PASS is not extended. LOCAL-01 acceptance is not reopened |
| A8 | Targeted nodes below. The existing module's repository discovery test was not run |

## Tenant Isolation Audit

```text
tenant scope applicable: YES
canonical tenant identity: unchanged; RequestIdentity.tenant_id and the environment resolver
tenant owner: unchanged
propagation path: existing resolver check, then existing operation scope check
state isolation: one authority instance remains one application environment
provider/config isolation: tenant-specific history_scope cannot activate
evidence/trace isolation: not a new trace path
async/recovery continuity: local recovery code was not edited
cross-tenant path: environment activation does not admit another tenant's initialize
fail-closed behavior: mismatch returns Inactive or TENANT_MISMATCH
adversarial evidence: targeted tenant-scope rejection and tenant-b initialize denial
result: BLOCKED
```

BLOCKED means this child does not extend historical `FRZ-TEN-*` PASS and does not prove a cloud anchor. The local configuration invariant is covered by the targeted tests.

## Tests

`pytest-timeout` is not installed. The process was capped at 60 seconds externally.

Recorded run, matching this tree: **13 passed in 0.71s**. Exit was the pytest summary, not a timeout.

```text
uv run pytest -p no:xdist
  tests/unit/governance/test_d03_a_platform_policy_config_01.py::<11 nodes>
  tests/unit/governance/test_d03_a_local_01_elcr_iirw.py::test_missing_activation_and_cross_tenant_owner_deny
  tests/unit/governance/test_d03_a_local_01_elcr_iirw.py::test_retention_hard_limits_and_writer_cannot_weaken
```

Targeted Ruff check on the changed Python files: **All checks passed**.

Not executed, and not a global regression PASS:

- `test_initialize_and_advance_are_monotonic`
- `test_stale_revision_fence_scope_and_permissions`
- `test_quarantine_reconcile_revocation_and_fence`
- `test_concurrent_cas_has_one_winner`
- `test_crash_before_witness_recovers_without_commit`
- `test_crash_after_witness_recover_commits`
- `test_primary_rollback_denies_without_unsealing`
- `test_witness_rollback_denies_mismatch`
- `test_forged_and_unauthorized_continuity_permissions`
- `test_altered_witness_metadata_denies_without_unsealing`
- `test_inconsistent_recovery_does_not_commit_or_unseal`
- `test_direct_construction_and_production_paths_stay_closed` (repository `rglob`)
- `test_large_history_routine_verification_stays_bounded`
- `test_historical_corruption_and_omission_use_full_chain_verification`
- `test_current_head_witness_mismatch_is_bounded`
- `test_binding_digest_is_canonical_and_separator_unambiguous`

## Applicable FRZ

Checklist status is unchanged. Evidence here is scoped code plus the targeted tests. It is not freeze evidence.

| Family | Checklist status | This child |
|---|---|---|
| `FRZ-OWN-01`..`05` | OPEN | Same governance owner and same composition function |
| `FRZ-BND-01`..`06` | OPEN | No new layer edge. CREATE gate is not a caller |
| `FRZ-CTR-01`..`06` | OPEN | Additive typed contract. No vendor types |
| `FRZ-TYP-01`..`06` | OPEN | Frozen models, `extra="forbid"`. No semantic `Any` |
| `FRZ-PLG-01`..`07` | OPEN | No new provider registration or self-activation |
| `FRZ-CFG-01`..`08` | PASS | Historical PASS. Not extended. Missing or rejected policy does not open the adapter |
| `FRZ-TEN-01`..`12` | PASS | Historical PASS. Not extended. Local scope and mismatch checks only |
| `FRZ-SEC-01`, `FRZ-SEC-04` | OPEN | No production auth bypass was added. IAM is not implemented |
| `FRZ-RPL-01`..`04` | OPEN | RPL-A..H not claimed |

## Findings

| Finding | Classification |
|---|---|
| Configured administrator identifiers are not an inspected IAM graph | **IN-SCOPE BLOCKER** for parent ratification. Not claimed by this child |
| OD-D03-A-06 production duration remains unresolved | **IN-SCOPE BLOCKER** for a production duration. This child does not set one |
| OD-D03-A-07 prices, capacity, and achieved SLO remain unresolved | **IN-SCOPE BLOCKER** for budget and SLO |
| D03-B, D03-C, IMPL-01, R7-Q, PROD-Q, and global `FRZ-*` stay open | **TRACKED FREEZE DEBT** |
| Cloud sandbox, disaster recovery, and real administrative independence were not executed | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** |
| Sixteen LOCAL-01 nodes, including repository discovery, were not run | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** for any claim of full LOCAL-01 regression. Targeted PASS is not that claim |

No finding in this child is self-certified as enterprise compliance.
