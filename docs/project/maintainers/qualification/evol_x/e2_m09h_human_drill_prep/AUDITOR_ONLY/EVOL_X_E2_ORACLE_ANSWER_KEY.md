# EVOL-X E2 — Auditor-only oracle answer key

**Audience:** Operator post-submission review · Independent B6 auditor · **NOT** the blind evaluator.

**Derivation rule (normative):** [`EVOL_X_M09H_CAUSE_IDENTITY.md`](../EVOL_X_M09H_CAUSE_IDENTITY.md)

## Case oracle (@ rebound registry / E-CORR-M09H-ORACLE)

| Case | Seed path | input_digest @ TES | expected_cause_identities (DIRECT module `stable_digest_hex([kind, node_id])`) |
|------|-----------|-------------------|--------------------------------------------------------------------------------|
| C1 | `intergrax/contracts/execution_identity.py` | `91f6d5308145ee4757b09502e76673dfef4651cb9da3ec02419fd281b87b6db3` | `3fc41379bf4835887ef1758cc1aedf2f966e2280cb58f9b14da09bbfa48b091d` |
| C2 | `testing_support/change_safety/graph/impact.py` | `dda029e107db5ef691c1a94284d0b609e860115e5f65bb2b5a5e8295ae1dcaf6` | `f1decda53e2306a25418e70ed30046d491f36639e30cb0bfb52be7db1d61f83e` |
| C3 | `testing_support/change_safety/contracts/enums.py` | `1feef84e94866e940f6c0453fc82c9da04d798926dd9025d2d8c86be7fc58a50` | `82364099afe3fa41f42a9f0860ce54bebf4c53ec9f740f04859277ac6069ca55` |

Historical TECH registry hashes (`9328b3d6…`, etc.) are **invalid** — superseded evidence only.

## Correctness rule (Mode A)

`attributed_cause_identities` must equal the table exactly (order-sensitive).
