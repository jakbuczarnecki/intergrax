# EVOL-X W4 — Workflow evolution (WM-01..15) and deployment safety (DM-01..07) matrices

**Status:** IMPLEMENTATION-DEFINED (Batch D, 2026-11-01). The W1 documents `EVOL_X_WORKFLOW_EVOLUTION_MATRIX_W1.md` and `EVOL_X_DEPLOYMENT_SAFETY_MATRIX_W1.md` referenced by the master plan **do not exist in the repository**; the matrix rows were derived from OD-EVOL-02 (policy i), OD-EVOL-05a/b/c, OD-EVOL-06, N-02, N-04, and `EVOL_X_ENFORCEMENT_BOUNDARY_W1.md`. **Requires independent auditor review (W7-01)** — this document is not self-certification.

SSOT data: `tests/qualification/evol_x/data/workflow_evolution_matrix.v1.json`, `deployment_safety_matrix.v1.json`, `deployment_na_register.v1.json`, `requalification_triggers.v1.json`, `frozen_core_registry.v1.json`.

## Workflow matrix (first-match, fail-closed catch-all WM-15)

Inputs: `has_inflight`, `adopt_in_place`, `pin_evidence`, `fingerprint_changed`, `graph_spec_bumped`, `migration_declared`, `superseded_readable`.
Policy invariant (exhaustively tested over all 128 inputs): an in-flight execution adopting a changed definition **in place** is never `SAFE_IN_PLACE`/`VERSIONED_DEPLOY`/`MIGRATE`.
Every row must be reachable (coverage test).

## Deployment matrix (first-match, fail-closed catch-all)

Inputs: `operation`, `evidence_present`, `contract_compat`, `persisted_schema_compatible`, `durable_forward_only_applied`, `inflight_pins`, `drain_possible`. Revision facts come from a sidecar `schema_snapshot_manifest.v1` (OD-EVOL-06). Assessor only (OD-EVOL-05b Option 1: **no** production port; W4-04 not executed). NA-01..04 are recorded out-of-scope (`deployment_na_register.v1.json`).

## Triggers T1..T9

T1 graph digest · T2 inventory digest · T3 ruleset version · T4 extractor manifest · T5 selection policy + conservative exclusions · T6 VL3 execution manifest · T7 owner rules · T8 exclusion register · T9 frozen-core registry. Missing digest on either side counts as stale.

## Frozen-core path mapping

Governance "Frozen Surface Registry" rows are mapped to path prefixes in `frozen_core_registry.v1.json`; unclassified non-extension paths are fail-safe Class C (governance fail-safe rule).
