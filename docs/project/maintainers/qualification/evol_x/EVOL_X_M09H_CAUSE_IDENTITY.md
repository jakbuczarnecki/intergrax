# EVOL-X M-09H — Cause identity (normative)

**Policy:** OD-EVOL-08-M1 · **Implementation:** `testing_support/change_safety/diagnostics/m09_cause_identity.py`

## Semantic meaning

For a drill seed `changed_path` @ `test_execution_sha`, the **attributed root cause** is the **DIRECT**
semantic-graph node representing the changed Python module. The node’s stable identifier is
`node_id` (for example `module:testing_support.change_safety.graph.impact`) with graph kind
`DIRECT`.

The **cause identity** submitted by the evaluator (and stored in
`expected_cause_identities` / `attributed_cause_identities`) is:

```text
stable_digest_hex([kind, node_id])
```

using `testing_support/change_safety/contracts/canonical_json.py` (sorted-keys JSON, SHA-256 hex).

This reuses the existing closed-world graph node identity — not an ad-hoc hash of file bytes or
dossier content.

## Human procedure (blind)

1. From the diagnostic packet, read `impact.affected` (or operator-provided `describe_attribution_evidence`).
2. Identify the **DIRECT** node whose `node_id` equals `module:<module_id_for_path(seed)>`.
3. Compute `stable_digest_hex([kind, node_id])` with the canonical helper or documented serialization.
4. Submit that 64-character hex string as `attributed_cause_identities[0]`.

The evaluator **does not** need the committed registry or auditor answer key to compute the digest
once the correct node is identified.

## Registry oracle

`evol_m09_drill_case_registry.v1.json` `expected_cause_identities` MUST equal the derivation above
@ `bound_test_execution_sha`. Mode A final certification rejects registry / diagnostic mismatch
(`testing_support/change_safety/evidence/p03.py`).

## Historical defect

Pre-E-CORR-M09H-ORACLE registry hashes were not tied to this rule (finding
`F-EVOLX-E2-CAUSE-PREFLIGHT`). Superseded TECH bundles remain historical; active TECH after
correction republish binds a recomputed registry @ NEW_TEST_EXECUTION_SHA.
