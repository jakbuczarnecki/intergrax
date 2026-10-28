# EBH-5 — ALLOWED_FIXED_SUBSTRATE_DECISIONS

| Field | Value |
| --- | --- |
| **Status** | Operator decisions **2026-10-28** recorded — **not** global FRZ PASS |
| **SSOT** | [`tests/qualification/ebh_5/_fixed_substrate_registry.py`](../../../tests/qualification/ebh_5/_fixed_substrate_registry.py) |
| **Operator packet** | [`EBH_5_OD_EBH5_02_SAC_FREEZE_PACKET.md`](EBH_5_OD_EBH5_02_SAC_FREEZE_PACKET.md) |

## Decisions (G06-02)

| substrate_id | decision_status | canonical_contract | ratification_authority | decision_evidence_sha |
| --- | --- | --- | --- | --- |
| `runtime-sqlite-persistence-bundle` | **RATIFIED — SCOPED** @ 2026-10-28 | Per-store port contracts | OD-EBH5-04-A operator scoped | STATE-X / SQLite annex + OD-EBH5-04-A |
| `single-node-profile-v1-sqlite-triad` | **RATIFIED — SCOPED PROD-Q** | SingleNodeProductionProfileV1Config | PROD-Q chain | `83a376b3…` scoped baseline |
| `lab-adaptive-sqlite` | **RATIFIED — LAB ONLY** @ 2026-10-28 | Lab adaptive stores | OD-EBH5-04-B operator lab-only | GAP-01 + OD-EBH5-04-B |

Legacy columns (scope, ADR, constraints) remain in Python SSOT.

## Decisions table (legacy summary)

| substrate_id | technology | scope_paths | contract_if_any | ratification_authority | adr_ref | multi_instance_constraints | review_date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `runtime-sqlite-persistence-bundle` | SQLite file-backed runtime stores | `intergrax/runtime/**/sqlite*.py`; agents persistence | Per-store port contracts | Platform maintainer + OD-EBH5-04 FIXED | STATE-X SX-F*; SQLite annex | Single-node / profile-scoped | 2026-10-28 — EBH-5 review cycle |
| `single-node-profile-v1-sqlite-triad` | SQLite GCA/PIS/AD | `single_node_production_profile_v1.py` | SingleNodeProductionProfileV1Config | PROD-Q qualification chain | PROD-Q ADR; DIST-SCALE scoped | Strict single-node | 2026-10-28 — EBH-5 review cycle |
| `lab-adaptive-sqlite` | SQLite adaptive stores | lab factory + `runtime/adaptive/*` | Lab factory paths (GAP-01) | Lab harness owner — H_REFERENCE | EBH-5 GAP-01; OD-EBH5-04 | Lab only | 2026-10-28 — EBH-5 review cycle |

Inventory `FIXED SUBSTRATE — RATIFIED WITH EVIDENCE` concern_ids must map to `runtime-sqlite-persistence-bundle` or `single-node-profile-v1-sqlite-triad` (see SSOT gate).

## Operator review (OD-EBH5-04)

Runtime SQLite ports remain **FIXED SUBSTRATE** unless operator promotes specific ports to `extensibility_promised=yes` with full RPL L3 proof.
