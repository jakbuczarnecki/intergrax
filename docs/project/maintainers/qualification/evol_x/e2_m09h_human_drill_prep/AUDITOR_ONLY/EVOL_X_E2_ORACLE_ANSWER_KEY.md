# EVOL-X E2 — Auditor-only oracle answer key

**Audience:** Operator post-submission review · Independent B6 auditor · **NOT** the blind evaluator.

**Canonical Mode A oracle:** committed `expected_cause_identities` in the rebound registry @ EVIDENCE_FINAL HEAD (staging copy: [../artifacts/evol_m09_drill_case_registry.v1.rebind_staging.json](../artifacts/evol_m09_drill_case_registry.v1.rebind_staging.json)).

## Case oracle (after rebound @ TES `dcd7ae6e…`)

| Case | Seed path | input_digest @ TES | expected_cause_identities |
|------|-----------|-------------------|---------------------------|
| C1 | `intergrax/contracts/execution_identity.py` | `91f6d5308145ee4757b09502e76673dfef4651cb9da3ec02419fd281b87b6db3` | `9328b3d65d4e54c1ccfa9298d440915c5f7de2b15cbc557148fb518ba4bf25ed` |
| C2 | `testing_support/change_safety/graph/impact.py` | `dda029e107db5ef691c1a94284d0b609e860115e5f65bb2b5a5e8295ae1dcaf6` | `a2e34afb13fa8f11947bde48275e209909f88923b3f4c51b5c6676bb8b48c112` |
| C3 | `testing_support/change_safety/contracts/enums.py` | `1feef84e94866e940f6c0453fc82c9da04d798926dd9025d2d8c86be7fc58a50` | `b41609cafc86c65456508ab2ac5defbc45159e7014aebd6793d730f9780ff97b` |

## Correctness rule (Mode A)

For each case, `attributed_cause_identities` in `evol_m09_human_drill.v2.json` must equal the table above **exactly** (order-sensitive tuple). Record submitted incorrect answers as-is; set `correct: true` only on match.

## Pre-flight operator warning (F-EVOLX-E2-CAUSE-PREFLIGHT)

Independent replay tooling (`assess_path_change_impact` + `stable_digest_hex([kind, node_id])` over DIRECT nodes @ TES) **does not reproduce** the table hashes from emitted diagnostic JSON alone. Before human execution, operator must either:

- obtain operator-signed confirmation that the attribution rule used in drill matches the frozen registry oracle, **or**
- STOP and open a qualification defect (do not fabricate human PASS).

Document pre-flight output under `.tmp/session/evol-x-e2-m09h/preflight-cause-oracle.log`.

## population_digest @ rebound registry

Compute with `m09h_population_digest` in `testing_support/change_safety/evidence/p03.py` over rebound registry bytes.

## TECH bundle note (historical)

Published TECH registry still has `bound_test_execution_sha = 6fec9ec…`; FINAL supplement must replace bytes in-bundle with rebound staging — do not rewrite TECH SHA history.
