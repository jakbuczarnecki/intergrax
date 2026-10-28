# EBH-5 — ALLOWED_FIXED_SUBSTRATE_DECISIONS

| Field | Value |
| --- | --- |
| **Status** | Ratification records for EBH-5 — **not** global FRZ PASS |
| **SSOT** | [`tests/qualification/ebh_5/_fixed_substrate_registry.py`](../../../tests/qualification/ebh_5/_fixed_substrate_registry.py) |
| **Evidence SHA** | [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md) |

## Decisions (G06-02)

| substrate_id | technology | scope_paths | contract_if_any | ratification_authority | adr_ref | multi_instance_constraints | review_date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `runtime-sqlite-persistence-bundle` | SQLite file-backed runtime stores | `intergrax/runtime/**/sqlite*.py`; agents persistence | Per-store port contracts | Platform maintainer + OD-EBH5-04 FIXED | STATE-X SX-F*; SQLite annex | Single-node / profile-scoped | 2026-10-28 — EBH-5 review cycle |
| `single-node-profile-v1-sqlite-triad` | SQLite GCA/PIS/AD | `single_node_production_profile_v1.py` | SingleNodeProductionProfileV1Config | PROD-Q qualification chain | PROD-Q ADR; DIST-SCALE scoped | Strict single-node | 2026-10-28 — EBH-5 review cycle |
| `lab-adaptive-sqlite` | SQLite adaptive stores | lab factory + `runtime/adaptive/*` | Lab factory paths (GAP-01) | Lab harness owner — H_REFERENCE | EBH-5 GAP-01; OD-EBH5-04 | Lab only | 2026-10-28 — EBH-5 review cycle |

Inventory `FIXED SUBSTRATE — RATIFIED WITH EVIDENCE` concern_ids must map to `runtime-sqlite-persistence-bundle` or `single-node-profile-v1-sqlite-triad` (see SSOT gate).

## Operator review (OD-EBH5-04)

Runtime SQLite ports remain **FIXED SUBSTRATE** unless operator promotes specific ports to `extensibility_promised=yes` with full RPL L3 proof.
